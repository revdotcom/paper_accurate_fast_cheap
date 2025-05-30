import os
from typing import Tuple, Optional, List
import torch

from wenet.rwkv_v6.rwkv_wrapper import RWKV_TmixWrapper

class RWKV_TmixWrapper_bidirectional_direction_dropout_both(torch.nn.Module):

    def __init__(self,
                 head_size: int,
                 dim_att: int,
                 num_blocks : int,
                 rnn_att_version: str,
                 rnn_att_direction: str,
                 ctx_len: int = 2048,
                 do_bfloat16: bool = True,
                 layer_id: int = 1):
        super().__init__()

        self.layer_id = layer_id
        self.num_blocks = num_blocks
        self.bi_layers_actives = []
        self.bi_active = True
        self.alt_decoding = False
        if os.environ.get("RWKV_ALT_DECODING", "0") == "1":
            print("RWKV_ALT_DECODING enabled")
            self.alt_decoding = True

        if os.environ.get("RWKV_BIDIRECTIONAL_LAYERS"):
            self.bi_layers_actives = [int(x) for x in os.environ.get("RWKV_BIDIRECTIONAL_LAYERS").split(",")]
            # print(f"RWKV_BIDIRECTIONAL_LAYERS: {self.bi_layers_actives}")
            if layer_id not in self.bi_layers_actives:
                print(f"RWKV_BIDIRECTIONAL_LAYERS: {layer_id} not in {self.bi_layers_actives}"	)
                self.bi_active = False
        self.rwkv_wrapper_forward = RWKV_TmixWrapper(head_size, dim_att, num_blocks, rnn_att_version, rnn_att_direction, ctx_len, do_bfloat16, layer_id)
        self.rwkv_wrapper_backward = RWKV_TmixWrapper(head_size, dim_att, num_blocks, rnn_att_version, rnn_att_direction, ctx_len, do_bfloat16, layer_id)

        # print("direction drop both init")

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

        x = query

        if self.training:
            p =0.2
            keep = torch.distributions.bernoulli.Bernoulli(1-p).sample((1,))
            if keep==1:
                x_att_forward, new_att_cache_forward = self.rwkv_wrapper_forward(x, x, x, mask, pos_emb, cache)
                x_fliped = torch.flip(x, [1])
                x_att_backward, new_att_cache_backward = self.rwkv_wrapper_backward(x_fliped, x_fliped, x_fliped, mask, pos_emb, cache)
                x_att_backward_fliped = torch.flip(x_att_backward, [1])
                out = (x_att_forward + x_att_backward_fliped)/2
            else:
                choose = torch.distributions.bernoulli.Bernoulli(1-p).sample((1,))
                if choose<=0.5:
                    x_att_forward, new_att_cache_forward = self.rwkv_wrapper_forward(x, x, x, mask, pos_emb, cache)
                    out = x_att_forward
                else:
                    x_fliped = torch.flip(x, [1])
                    x_att_backward, new_att_cache_backward = self.rwkv_wrapper_backward(x_fliped, x_fliped, x_fliped, mask, pos_emb, cache)
                    x_att_backward_fliped = torch.flip(x_att_backward, [1])
                    out = x_att_backward_fliped
        elif self.bi_active:
            # print(f"layerid={self.layer_id} bi_active: {self.bi_active}")
            x_att_forward, new_att_cache_forward = self.rwkv_wrapper_forward(x, x, x, mask, pos_emb, cache)
            x_fliped = torch.flip(x, [1])
            x_att_backward, new_att_cache_backward = self.rwkv_wrapper_backward(x_fliped, x_fliped, x_fliped, mask, pos_emb, cache)
            x_att_backward_fliped = torch.flip(x_att_backward, [1])
            out = (x_att_forward + x_att_backward_fliped)/2
        elif self.alt_decoding and self.layer_id % 2 == 0:
            # print(f"layerid={self.layer_id} alt, left only")
            # left only
            x_att_forward, new_att_cache_forward = self.rwkv_wrapper_forward(x, x, x, mask, pos_emb, cache)
            out = x_att_forward
        elif self.alt_decoding and self.layer_id % 2 == 1:
            # print(f"layerid={self.layer_id} alt, right only")
            # right only
            x_fliped = torch.flip(x, [1])
            x_att_backward, new_att_cache_backward = self.rwkv_wrapper_backward(x_fliped, x_fliped, x_fliped, mask, pos_emb, cache)
            x_att_backward_fliped = torch.flip(x_att_backward, [1])
            out = x_att_backward_fliped
        else:
            # print(f"layerid={self.layer_id} no alt, left")
            x_att_forward, new_att_cache_forward = self.rwkv_wrapper_forward(x, x, x, mask, pos_emb, cache)
            out = x_att_forward

        # new_att_cache = new_att_cache_forward  # Is that the best choice? Does it matter for bidirectional RNN?
        new_att_cache = cache

        return out, new_att_cache
