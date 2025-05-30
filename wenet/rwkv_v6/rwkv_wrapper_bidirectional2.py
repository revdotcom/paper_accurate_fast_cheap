import os
from typing import Tuple, Optional, List
import torch
from time import sleep

from wenet.rwkv_v6.rwkv_wrapper import RWKV_TmixWrapper
import nvtx

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

        # Create persistent streams and events
        self.layer_id = layer_id
        self.stream1 : torch.cuda.Stream = None
        self.stream2 : torch.cuda.Stream = None
        self.flipped_buffer_set = False
        self.flipped_buffer = None
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

        # return self.forward_regular(query, key, value, mask, pos_emb, cache)
        # return self.forward_speed0(query, key, value, mask, pos_emb, cache)
        return self.forward_flip(query, None, None, None, None, None)

    def forward_regular(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
            pos_emb: torch.Tensor = torch.empty(0),
            cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Regular code """

        x = query
        if self.do_bfloat16:
            x = x.to(dtype=torch.bfloat16)

        x_fliped = torch.flip(x, [1])

        with nvtx.annotate("Stream 1: Forward Computation"):
            x_att_forward, new_att_cache_forward = self.rwkv_wrapper_forward(x, x, x, mask, pos_emb, cache)

        with nvtx.annotate("Stream 2: back Computation"):
            x_att_backward, _ = self.rwkv_wrapper_backward(x_fliped, x_fliped, x_fliped, mask, pos_emb, cache)
            x_att_backward_fliped = torch.flip(x_att_backward, [1])

        with nvtx.annotate("Merge Computation"):
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

    def forward_flip(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
            pos_emb: torch.Tensor = torch.empty(0),
            cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Regular code """
        # if self.layer_id ==0 :
        #     print("flip")

        B, T, N = query.size()
        if not self.flipped_buffer_set:
            self.flipped_buffer_set = True
            self.flipped_buffer = torch.zeros((B, T, N), dtype=torch.bfloat16, device=query.device)


        x = query
        self.flipped_buffer.resize_(B, T, N)

        if self.do_bfloat16:
            x = x.to(dtype=torch.bfloat16)

        x_att_forward, new_att_cache_forward = self.rwkv_wrapper_forward(x, x, x, None, None, None)
        # x_fliped = x[:, torch.arange(x.size(1) - 1, -1, -1), :]
        # self.flipped_buffer.copy_(x[:, torch.arange(T - 1, -1, -1), :])
        self.flipped_buffer[:, :, :] = x[:, torch.arange(T - 1, -1, -1), :]

        # x_fliped = torch.flip(x, [1])
        x_att_backward, _ = self.rwkv_wrapper_backward(self.flipped_buffer, self.flipped_buffer, self.flipped_buffer, None, None, None)
        # x_att_backward_fliped = torch.flip(x_att_backward, [1])

        # we don't need the content of flipped buffer anymore, so let's use it 
        # to flip the x_att_backward
        # self.flipped_buffer.copy_(x_att_backward)
        self.flipped_buffer[:, :, :] = x_att_backward[:, torch.arange(T - 1, -1, -1), :]
        # x_att_backward_fliped = self.flipped_buffer

        # if self.layer_id ==0 :
        #     print(f"x_att_backward_fliped: {x_att_backward_fliped.shape}, x shape : {x.shape}")

        out = (x_att_forward + self.flipped_buffer) / 2
        # out = (x_att_forward + x_att_backward_fliped) / 2

        if self.do_bfloat16:
            out = out.float()
        else:
            out = out


        new_att_cache = new_att_cache_forward  # Is that the best choice? Does it matter for bidirectional RNN?
        # new_att_cache = cache

        return out, new_att_cache

    def forward_speed0(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
            pos_emb: torch.Tensor = torch.empty(0),
            cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Forward optimized with streams. This works in terms of output results, but is slower then the original version"""

        device = query.device

        if self.stream1 is None:
            self.stream1 = torch.cuda.Stream(device=device)
            self.stream2 = torch.cuda.Stream(device=device)

        x = query
        if self.do_bfloat16:
            x = x.to(dtype=torch.bfloat16)

        x_fliped = torch.flip(x, [1])
        # Ensure we're in sync before starting new computation
        torch.cuda.synchronize(device)

        x_att_backward, new_att_cache_backward = None, None
        x_att_forward, new_att_cache_forward = None, None

        with nvtx.annotate("Stream 1: Forward Computation"):
            with torch.cuda.stream(self.stream1) as sc1:
                x_att_forward, new_att_cache_forward = self.rwkv_wrapper_forward(x, x, x, mask, pos_emb, cache)
                # x_att_forward = x_att_forward / 2.0

        with nvtx.annotate("Stream 2: back Computation"):
            with torch.cuda.stream(self.stream2) as sc2:
                x_att_backward, _ = self.rwkv_wrapper_backward(x_fliped, x_fliped, x_fliped, mask, pos_emb, cache)
                x_att_backward_fliped = torch.flip(x_att_backward, [1])
                # x_att_backward_fliped = x_att_backward_fliped / 2.0
        # Synchronize before merging to ensure all computations are complete
        # self.event_s1.synchronize()
        # self.event_s2.synchronize()

        # self.stream1.synchronize(device)
        # self.stream2.synchronize(device)

        torch.cuda.synchronize(device)
        with nvtx.annotate("Merge Computation"):
            out = (x_att_forward + x_att_backward_fliped)/ 2.0

        if self.do_bfloat16:
            out = out.float()

        # if torch.isnan(out).sum().item() > 0:
        #     print(f"NaN detected in output: {torch.isnan(out).sum().item()}, x_att_forward: {torch.isnan(x_att_forward).sum().item()}, x_att_backward_fliped: {torch.isnan(x_att_backward_fliped).sum().item()}")
        # else:
        #     print("OK") 

        new_att_cache = new_att_cache_forward

        return out, new_att_cache



    def forward_opt2(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Some other version of forward optimized"""

        x = query
        if self.do_bfloat16:
            x = x.to(dtype=torch.bfloat16)

        x_fliped = torch.flip(x, [1])

        # Define CUDA streams
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        event1 = torch.cuda.Event()
        event2 = torch.cuda.Event()
        # Results placeholders
        x_att_forward = None
        x_att_backward_fliped = None
        new_att_cache_forward = None

        # Forward computation on stream1
        with torch.cuda.stream(stream1):
            x_att_forward, new_att_cache_forward = self.rwkv_wrapper_forward(x, x, x, mask, pos_emb, cache)
            event1.record()

        # Backward computation on stream2
        with torch.cuda.stream(stream2):
            x_att_backward, _ = self.rwkv_wrapper_backward(x_fliped, x_fliped, x_fliped, mask, pos_emb, cache)
            x_att_backward_fliped = torch.flip(x_att_backward, [1])
            event2.record()

        # Wait for streams to finish
        event1.wait()
        event2.wait()
        if torch.isnan(x_att_forward).sum().item() > 0:
            print(f"NaN detected in x_att_forward: {torch.isnan(x_att_forward).sum().item()}")
        if torch.isnan(x_att_backward_fliped).sum().item() > 0:
            print(f"NaN detected in x_att_backward_fliped: {torch.isnan(x_att_backward_fliped).sum().item()}")

        # Merge results
        out = (x_att_forward + x_att_backward_fliped) / 2

        if self.do_bfloat16:
            out = out.float()

        if torch.isnan(out).sum().item() > 0:
            print(f"NaN detected in output: {torch.isnan(out).sum().item()}, x_att_forward: {torch.isnan(x_att_forward).sum().item()}, x_att_backward_fliped: {torch.isnan(x_att_backward_fliped).sum().item()}")
        else:
            print("OK") 
        new_att_cache = new_att_cache_forward  # Cache selection for bidirectional RNN
        return out, new_att_cache