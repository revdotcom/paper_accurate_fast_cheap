import os
from typing import Tuple, Optional, List
import torch

class RWKV_TmixWrapper(torch.nn.Module):

    #     def __init__(self, layer_id, n_layer, n_embd, n_head, head_size, dim_att, chunk_len:int = 128, precision:int = 64, max_ctx_len:int = 4096):
            # global RWKV_NO_CUDA
            # RWKV_NO_CUDA = False
            # encoder_selfattn_layer_args = (
            #     1, num_blocks, 
            #     output_size,
            #     attention_heads,
            #     attention_heads * 8,
            #     attention_heads * attention_heads * 8,
            # )

    def __init__(self,
                 head_size: int,
                 dim_att: int,
                 num_blocks : int,
                 rnn_att_version: str = None,
                 rnn_att_direction: str = None,
                 ctx_len: int = 2048,
                 do_bfloat16: bool = True,
                 layer_id: int = 1):
        super().__init__()
        self.head_size = head_size
        self.dim_att = dim_att
        self.num_blocks = num_blocks
        self.rnn_att_version = rnn_att_version
        self.rnn_att_direction = rnn_att_direction
        self.ctx_len = ctx_len
        self.do_bfloat16 = do_bfloat16
        self.layer_id = layer_id

        self.n_head = dim_att // head_size 
        self.head_size = head_size
        self.n_embd = dim_att # these two quantities are the same, at least for that module
        
        # here, we set the os variables...
        os.environ["RWKV_MY_TESTING"] = "x060"
        os.environ["RWKV_HEAD_SIZE_A"] = f"{self.head_size}"
        os.environ["RWKV_JIT_ON"] = "1"
        os.environ["RWKV_CTXLEN"] = f"{ctx_len}"
        os.environ["RWKV_TRAIN_TYPE"] = ""
        #for k in os.environ.keys():
        #    if 'RWKV_' in k:
        #        print(f"{k=}, {os.environ[k]}")

        from wenet.rwkv_v6.src.model import RWKV_Tmix_x060c as RwkvBlock
        self.tmix_block =  RwkvBlock(head_size = self.head_size, n_layers=self.num_blocks, n_embd=self.n_embd, dim_att=self.dim_att, layer_id=self.layer_id)
        if self.do_bfloat16:
            self.tmix_block = self.tmix_block.to(dtype=torch.bfloat16)

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

        query_dtype = query.dtype
        if self.do_bfloat16:
            query = query.to(dtype=torch.bfloat16)


        # x_att_16 = self.self_attn(x16, need_state=False)
        x_att_16 = self.tmix_block(query)
        # this will be a noop if we're not using bfloat16
        if self.do_bfloat16:
            #x_att = x_att_16.float()
            x_att = x_att_16.to(dtype=query_dtype)
        else:
            x_att = x_att_16

        new_att_cache = cache

        return x_att, new_att_cache
