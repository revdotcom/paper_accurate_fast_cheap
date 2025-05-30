# Copyright (c) 2024, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

from huggingface_hub import PyTorchModelHubMixin

from mamba_ssm.modules.mamba2 import Mamba2


class Mamba2Bidirectional(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=32, #64, # make sure: d_model * expand / headdim = multiple of 8 (https://github.com/state-spaces/mamba/issues/351#issuecomment-2167091940)
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.mamba_forward = Mamba2(
            d_model,
            d_state,
            d_conv,
            conv_init,
            expand,
            headdim, #64, # make sure: d_model * expand / headdim = multiple of 8 (https://github.com/state-spaces/mamba/issues/351#issuecomment-2167091940)
            d_ssm,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
            ngroups,
            A_init_range,
            D_has_hdim,
            rmsnorm,
            norm_before_gate,
            dt_min,
            dt_max,
            dt_init_floor,
            dt_limit,
            bias,
            conv_bias,
            # Fused kernel and sharding options
            chunk_size,
            use_mem_eff_path,
            layer_idx,  # Absorb kwarg for general module
            process_group,
            sequence_parallel,
            device,
            dtype,
        )
            
        self.mamba_backward = Mamba2(
            d_model,
            d_state,
            d_conv,
            conv_init,
            expand,
            headdim, #64, # make sure: d_model * expand / headdim = multiple of 8 (https://github.com/state-spaces/mamba/issues/351#issuecomment-2167091940)
            d_ssm,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
            ngroups,
            A_init_range,
            D_has_hdim,
            rmsnorm,
            norm_before_gate,
            dt_min,
            dt_max,
            dt_init_floor,
            dt_limit,
            bias,
            conv_bias,
            # Fused kernel and sharding options
            chunk_size,
            use_mem_eff_path,
            layer_idx,  # Absorb kwarg for general module
            process_group,
            sequence_parallel,
            device,
            dtype,
        )

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None, mask_pad=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        
        out_forward = self.mamba_forward(u, seqlen, seq_idx, cu_seqlens, inference_params)

        u_fliped = torch.flip(u, [1])
        out_backward = self.mamba_backward(u_fliped, seqlen, seq_idx, cu_seqlens, inference_params)
        out_backward_fliped = torch.flip(out_backward, [1])
        out = (out_forward + out_backward_fliped)/2

        return out