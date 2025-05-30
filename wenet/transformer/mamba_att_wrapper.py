import os
from typing import Tuple, Optional, List
import torch
from torch import nn

class MambaAttWrapper(torch.nn.Module):

    def __init__(self,
                 head_size: int,
                 dim_att: int,
                 num_blocks : int,
                 rnn_att_version: str = "mamba2", # mamba mamba_torch mamba2
                 rnn_att_direction: str = "bi", # uni bi
                 layer_id: int = 1):
        super().__init__()
        self.head_size = head_size
        self.dim_att = dim_att
        self.num_blocks = num_blocks
        self.rnn_att_version = rnn_att_version
        self.rnn_att_direction = rnn_att_direction
        self.layer_id = layer_id

        if self.rnn_att_version=="mamba":
            from mamba_ssm import Mamba
        elif self.rnn_att_version=="mamba_torch":
            from mamba_ssm.modules.mamba_simple_torchscript import Mamba
        elif self.rnn_att_version=="mamba2":
            if self.rnn_att_direction=="uni":
                # from mamba_ssm import Mamba2 as Mamba
                from mamba_ssm.modules.mamba2 import Mamba2 as Mamba
            elif self.rnn_att_direction=="bi":
                # from mamba_ssm.modules.mamba2_bidirectional import Mamba2Bidirectional as Mamba
                from wenet.transformer.mamba2_bidirectional import Mamba2Bidirectional as Mamba
        
        self.mamba = Mamba(self.dim_att, headdim=self.head_size)
        self._init_weights(self.mamba, layer_id)

    # x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb, att_cache)
    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
            pos_emb: torch.Tensor = torch.empty(0),
            cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        x_att = self.mamba(query)
        new_att_cache = cache

        return x_att, new_att_cache

    # https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
    def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
    ):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)
