import os
from typing import Tuple, Optional, List
import torch

from wenet.rwkv_v6.rwkv_wrapper import RWKV_TmixWrapper

class RWKV_TmixWrapper_bidirectional(torch.nn.Module):

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

        self.do_bfloat16 = do_bfloat16
        self.rwkv_wrapper_forward = RWKV_TmixWrapper(head_size, dim_att, num_blocks, rnn_att_version, rnn_att_direction, ctx_len, do_bfloat16, layer_id)
        self.rwkv_wrapper_backward = RWKV_TmixWrapper(head_size, dim_att, num_blocks, rnn_att_version, rnn_att_direction, ctx_len, do_bfloat16, layer_id)

        # we'll handle the conversion inside the bidirectional wrapper code itself
        # but we still need to initialize the classes with its original value 
        # since this changes how the weights are represented
        self.rwkv_wrapper_forward.do_bfloat16 = False
        self.rwkv_wrapper_backward.do_bfloat16 = False
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

        if self.do_bfloat16:
            query = query.to(dtype=torch.bfloat16)

        x = query
        x_fliped = torch.flip(x, [1])

        x_att_forward, new_att_cache_forward = self.rwkv_wrapper_forward(x, x, x, mask, pos_emb, cache)
        x_att_backward, new_att_cache_backward = self.rwkv_wrapper_backward(x_fliped, x_fliped, x_fliped, mask, pos_emb, cache)
        x_att_backward_fliped = torch.flip(x_att_backward, [1])
        out = (x_att_forward + x_att_backward_fliped) / 2

        # torch.cuda.synchronize()
        #out = (x_att_forward + x_att_backward_fliped)/2
        # out = x_att_backward_fliped

        if self.do_bfloat16:
            out = out.float()
        else:
            out = out


        new_att_cache = new_att_cache_forward  # Is that the best choice? Does it matter for bidirectional RNN?
        # new_att_cache = cache

        return out, new_att_cache

    def forward_parallel(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
            pos_emb: torch.Tensor = torch.empty(0),
            cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.do_bfloat16:
            query = query.to(dtype=torch.bfloat16)

        # Create streams
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()

        # Create CUDA events
        event_s1 = torch.cuda.Event(blocking=False)
        event_s2 = torch.cuda.Event(blocking=False)


        with torch.cuda.stream(s1):
            x_att_forward, new_att_cache_forward = self.rwkv_wrapper_forward(query, query, query, mask, pos_emb, cache)
            x_att_forward = x_att_forward / 2
            event_s1.record()

        with torch.cuda.stream(s2):
            x_fliped = torch.flip(query, [1])
            x_att_backward, new_att_cache_backward = self.rwkv_wrapper_backward(x_fliped, x_fliped, x_fliped, mask, pos_emb, cache)
            x_att_backward_fliped = torch.flip(x_att_backward, [1])
            x_att_backward_fliped  = x_att_backward_fliped / 2
            event_s2.record()

        with torch.cuda.stream(torch.cuda.default_stream()):
            # Wait for events to ensure x_att_forward and x_att_backward_fliped are ready
            event_s1.wait()
            event_s2.wait()
            # the / 2 is already done
            out = (x_att_forward + x_att_backward_fliped)

        if self.do_bfloat16:
            out = out.float()
        else:
            out = out


        new_att_cache = new_att_cache_forward  # Is that the best choice? Does it matter for bidirectional RNN?
        # new_att_cache = cache

        return out, new_att_cache
