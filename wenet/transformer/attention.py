# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Multi-Head Attention layer definition."""

import math
from typing import Optional, Tuple, List

import torch
from torch import nn

import torch.nn.functional as F

INF_VAL = 10000.0

class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 key_bias: bool = True, 
                 layer_id: int = 0):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(
        self,
        value: torch.Tensor,
        scores: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
    ) -> torch.Tensor:
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        mask = mask.to(value.device)
        n_batch = value.size(0)
        # NOTE(xcsong): When will `if mask.size(2) > 0` be True?
        #   1. onnx(16/4) [WHY? Because we feed real cache & real mask for the
        #           1st chunk to ease the onnx export.]
        #   2. pytorch training
        if mask.size(2) > 0:  # time2 > 0
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            # For last chunk, time2 might be larger than scores.size(-1)
            mask = mask[:, :, :, :scores.size(-1)]  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0)  # (batch, head, time1, time2)
        # NOTE(xcsong): When will `if mask.size(2) > 0` be False?
        #   1. onnx(16/-1, -1/-1, 16/0)
        #   2. jit (16/-1, -1/-1, 16/0, 16/4)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (x.transpose(1, 2).contiguous().view(n_batch, -1,
                                                 self.h * self.d_k)
             )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`

        """
        q, k, v = self.forward_qkv(query, key, value)

        # NOTE(xcsong):
        #   when export onnx model, for 1st chunk, we feed
        #       cache(1, head, 0, d_k * 2) (16/-1, -1/-1, 16/0 mode)
        #       or cache(1, head, real_cache_t, d_k * 2) (16/4 mode).
        #       In all modes, `if cache.size(0) > 0` will alwayse be `True`
        #       and we will always do splitting and
        #       concatnation(this will simplify onnx export). Note that
        #       it's OK to concat & split zero-shaped tensors(see code below).
        #   when export jit  model, for 1st chunk, we always feed
        #       cache(0, 0, 0, 0) since jit supports dynamic if-branch.
        # >>> a = torch.ones((1, 2, 0, 4))
        # >>> b = torch.ones((1, 2, 3, 4))
        # >>> c = torch.cat((a, b), dim=2)
        # >>> torch.equal(b, c)        # True
        # >>> d = torch.split(a, 2, dim=-1)
        # >>> torch.equal(d[0], d[1])  # True
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache,
                                                 cache.size(-1) // 2,
                                                 dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.
        new_cache = torch.cat((k, v), dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache

    def forward_output_weights(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
                pos_emb: torch.Tensor = torch.empty(0),
                cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`

        """
        q, k, v = self.forward_qkv(query, key, value)

        # NOTE(xcsong):
        #   when export onnx model, for 1st chunk, we feed
        #       cache(1, head, 0, d_k * 2) (16/-1, -1/-1, 16/0 mode)
        #       or cache(1, head, real_cache_t, d_k * 2) (16/4 mode).
        #       In all modes, `if cache.size(0) > 0` will alwayse be `True`
        #       and we will always do splitting and
        #       concatnation(this will simplify onnx export). Note that
        #       it's OK to concat & split zero-shaped tensors(see code below).
        #   when export jit  model, for 1st chunk, we always feed
        #       cache(0, 0, 0, 0) since jit supports dynamic if-branch.
        # >>> a = torch.ones((1, 2, 0, 4))
        # >>> b = torch.ones((1, 2, 3, 4))
        # >>> c = torch.cat((a, b), dim=2)
        # >>> torch.equal(b, c)        # True
        # >>> d = torch.split(a, 2, dim=-1)
        # >>> torch.equal(d[0], d[1])  # True
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(
                cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.
        new_cache = torch.cat((k, v), dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), new_cache, scores


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 key_bias: bool = True,
                 layer_id: int = 0):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate, key_bias)
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x, zero_triu: bool = False):
        """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """

        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
                               device=x.device,
                               dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0],
                                 x.size()[1],
                                 x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        # NOTE(xcsong):
        #   when export onnx model, for 1st chunk, we feed
        #       cache(1, head, 0, d_k * 2) (16/-1, -1/-1, 16/0 mode)
        #       or cache(1, head, real_cache_t, d_k * 2) (16/4 mode).
        #       In all modes, `if cache.size(0) > 0` will alwayse be `True`
        #       and we will always do splitting and
        #       concatnation(this will simplify onnx export). Note that
        #       it's OK to concat & split zero-shaped tensors(see code below).
        #   when export jit  model, for 1st chunk, we always feed
        #       cache(0, 0, 0, 0) since jit supports dynamic if-branch.
        # >>> a = torch.ones((1, 2, 0, 4))
        # >>> b = torch.ones((1, 2, 3, 4))
        # >>> c = torch.cat((a, b), dim=2)
        # >>> torch.equal(b, c)        # True
        # >>> d = torch.split(a, 2, dim=-1)
        # >>> torch.equal(d[0], d[1])  # True
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache,
                                                 cache.size(-1) // 2,
                                                 dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.
        new_cache = torch.cat((k, v), dim=-1)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        # JDF: come back to this!!!
        # Remove rel_shift since it is useless in speech recognition,
        # and it requires special attention for streaming.
        # matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask), new_cache

class LimitedRelPositionMultiHeadedAttention(RelPositionMultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 key_bias: bool = True,
                 att_context_size=[500,500],
                 global_tokens=0,
                 global_tokens_spacing=1,
                 global_attn_separate=False,
                 layer_id: int = 0):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate, key_bias, layer_id)
        self.att_context_size = att_context_size
        self.global_tokens = global_tokens
        self.global_tokens_spacing = global_tokens_spacing
        self.global_attn_separate = global_attn_separate

        if self.global_attn_separate:
            # Note: NeMo implementation has option to use bias on all three,
            # this uses the WeNet convention of only optionally adding bias
            # to the key weights
            self.global_q = nn.Linear(n_feat, n_feat)
            self.global_k = nn.Linear(n_feat, n_feat, bias=key_bias)
            self.global_v = nn.Linear(n_feat, n_feat)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        """
        q, k, v = self.forward_qkv(query, key, value)
        n_batch, _, T, _ = q.size()

        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache,
                                                 cache.size(-1) // 2,
                                                 dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
        #   non-trivial to calculate `next_cache_start` here.
        new_cache = torch.cat((k, v), dim=-1)

        w = max(self.att_context_size[0], self.att_context_size[1])
        if w <= 0:
            raise ValueError("When using local attention, context size must be set > 0")
        pad_len = (2 * w - T % (2 * w)) % (2 * w)  # pad time to 2w
        q = F.pad(q, (0, 0, 0, pad_len))  # (batch, head, time, size)
        k = F.pad(k, (0, 0, 0, pad_len))  # (batch, head, time, size)
        v = F.pad(v, (0, 0, 0, pad_len))  # (batch, head, time, size)
        mask = F.pad(mask, (0, pad_len), value=1.0)
        pos_emb = F.pad(pos_emb, (0, 0, 0, pad_len)) #JDF: had to add this

        #JDF this is a very silly wenet thing to transpose and then transpose back
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)
        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        
        #matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        diagonal_matrix_ac = self.sliding_chunks_matmul_qk(
            q_with_bias_u, k, w, padding_value=0.0
        )  # (batch, head, time, 2w + 1)
        
        # compute matrix b and matrix d
        # (batch, head, time1, time2)
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)
        p = p.expand(q_with_bias_v.size(0), -1, -1, -1)
        
        diagonal_matrix_bd = self.sliding_chunks_matmul_qk(
            q_with_bias_v, p, w, padding_value=0.0
        )  # (batch, head, time, 2w + 1)
        #print(diagonal_matrix_bd.shape)
        #print(diagonal_matrix_bd[0, 0, 0, :])
        #print(diagonal_matrix_bd[0, 0, 1, :])

        #scores = (matrix_ac + matrix_bd) / math.sqrt(
        #    self.d_k)  # (batch, head, time1, time2)
        start_pos = w - self.att_context_size[0]
        end_pos = w + self.att_context_size[1]

        diagonal_matrix_ac[:, :, :, : self.att_context_size[0]] += diagonal_matrix_bd[
            :, :, :, : self.att_context_size[0]
        ]
        diagonal_matrix_ac[:, :, :, -(self.att_context_size[1] + 1) :] += diagonal_matrix_bd[
            :, :, :, self.att_context_size[0] :
        ]
        scores = diagonal_matrix_ac / math.sqrt(self.d_k)
        # (batch, head, time, 2w + 1)

        # mask invalid positions
        scores[:, :, :, :start_pos] = -10000. #INF_VAL
        scores[:, :, :, end_pos + 1 :] = -10000. #INF_VAL
        #print(scores.shape)
        #print(scores[0, 0, 0, :])

        # This implementation is fast and takes very little memory because num_heads x hidden_size = 1
        # from (bsz x seq_len) to (bsz x num_heads x seqlen x hidden_size)
        mask = mask.unsqueeze(dim=-1) #.unsqueeze(dim=1)
        # cast to float/half then replace 1's with -inf
        float_mask = mask.type_as(scores).masked_fill(mask, -10000.) #INF_VAL)
        ones = float_mask.new_ones(size=float_mask.size())  # tensor of ones
        # diagonal mask with zeros everywhere and -inf inplace of padding
        
        d_mask = self.sliding_chunks_matmul_qk(ones, float_mask, w, padding_value=0.0)
        # (batch, head, time, 2w + 1)

        #scores += d_mask
        if self.global_tokens > 0:

            if self.global_attn_separate:
                global_q = self.global_q(query).view(n_batch, -1, self.h, self.d_k)
                global_k = self.global_k(key).view(n_batch, -1, self.h, self.d_k)
                global_v = self.global_v(value).view(n_batch, -1, self.h, self.d_k)
                global_q = global_q.transpose(1, 2)
                global_k = global_k.transpose(1, 2)
                global_v = global_v.transpose(1, 2)
                global_q = F.pad(global_q, (0, 0, 0, pad_len))  # (batch, head, time, size)
                global_k = F.pad(global_k, (0, 0, 0, pad_len))  # (batch, head, time, size)
                global_v = F.pad(global_v, (0, 0, 0, pad_len))  # (batch, head, time, size)
            else:
                global_q, global_k, global_v = q, k, v
            
            global_q /= math.sqrt(self.d_k)

            # assign which tokens are global
            is_index_global_attn = torch.zeros_like(mask.transpose(1,2))
            is_index_global_attn[
                :, : self.global_tokens * self.global_tokens_spacing : self.global_tokens_spacing
            ] = 1.0

            # compute global attn indices
            (
                max_num_global_attn_indices,
                is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero,
            ) = self._get_global_attn_indices(is_index_global_attn=is_index_global_attn)

            # calculate global attn probs with global keys
            # (batch, time, head, max_num_global_attn_indices)
            global_key_attn = self._compute_global_key_attn(
                query=global_q.transpose(1, 2),
                key=global_k.transpose(1, 2),
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            ).transpose(1, 2)
            
            # concat to local_attn_probs
            # (batch, time, head, max_num_global_attn_indices + 2*w)
            scores = torch.cat((global_key_attn, scores), dim=-1)
            
            # free memory
            del global_key_attn

        #print(scores.shape)
        #print(scores[0, 0, 0, :])
        attn = torch.softmax(scores, dim=-1)#.masked_fill(mask, 0.0)
        #print(attn.shape)
        #print(attn.nonzero(as_tuple=True))
        p_attn = self.dropout(attn)
        # (batch, head, time, 2w + 1)

        if self.global_tokens > 0:
            # compute sum of global and local attn
            out = self._compute_attn_output_with_global_indices(
                value=v,
                attn_probs=p_attn,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                w=w,
            )
        else:
            # compute local attn only
            out = self.sliding_chunks_matmul_pv(p_attn, v, w)

        out = out.reshape(n_batch, -1, self.h * self.d_k)[:, :T]

        if self.global_tokens > 0:
            out_global_to_all = self._compute_out_global_to_all(
                query=global_q,
                key=global_k,
                value=global_v,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                is_index_masked=mask,
            )

            # overwrite values with global attention
            out_global_to_all = out_global_to_all.reshape(out.shape[0], -1, out.shape[2])
            out[:, : self.global_tokens * self.global_tokens_spacing : self.global_tokens_spacing, :] = out_global_to_all

        ret = self.linear_out(out)
        #print()
        
        return ret, new_cache

    def _get_global_attn_indices(self, is_index_global_attn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute global attention indices.

        Args:
            is_index_global_attn (torch.Tensor): (batch, time) A boolean tensor indicating if an index is a global attention index.

        Returns:
            max_num_global_attn_indices (int): Maximum number of global attention indices in the batch.
            is_index_global_attn_nonzero (tuple): Indices of global attention (non-zero elements).
            is_local_index_global_attn_nonzero (tuple): Indices of non-padding values within global attention indices.
            is_local_index_no_global_attn_nonzero (tuple): Indices of padding values within global attention indices.
        """
        # Calculate the number of global attention indices in the batch
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1)

        # Find the maximum number of global attention indices in the batch
        max_num_global_attn_indices = num_global_attn_indices.max()

        # Get the indices of global attention (non-zero elements)
        #is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)
        is_index_global_attn_nonzero = torch.nonzero(is_index_global_attn, as_tuple=True)

        # Create a helper tensor to find the local indices of global attention
        is_local_index_global_attn = torch.arange(
            max_num_global_attn_indices, device=is_index_global_attn.device
        ) < num_global_attn_indices.unsqueeze(dim=-1)

        # Find the non-padding values within global attention indices
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(as_tuple=True)

        # Find the padding values within global attention indices
        is_local_index_no_global_attn_nonzero = (is_local_index_global_attn == 0).nonzero(as_tuple=True)

        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )

    def _compute_global_key_attn(
        self,
        key: torch.Tensor,
        query: torch.Tensor,
        max_num_global_attn_indices: int,
        is_index_global_attn_nonzero: tuple,
        is_local_index_global_attn_nonzero: tuple,
        is_local_index_no_global_attn_nonzero: tuple,
    ) -> torch.Tensor:
        """
        Compute the attention probabilities using only global key vectors.

        Args:
            key (torch.Tensor): (batch, time, head, head_dim) The key vectors.
            query (torch.Tensor): (batch, time, head, head_dim) The query vectors.
            max_num_global_attn_indices (int): Maximum number of global attention indices in the batch.
            is_index_global_attn_nonzero (tuple): Indices of global attention (non-zero elements).
            is_local_index_global_attn_nonzero (tuple): Non-padding values within global attention indices.
            is_local_index_no_global_attn_nonzero (tuple): Padding values within global attention indices.

        Returns:
            attn_probs_from_global_key (torch.Tensor): (batch, time, head, max_num_global_attn_indices) The computed attention probabilities using only global key vectors.
        """
        batch_size = key.shape[0]

        # create only global key vectors
        key_only_global = key.new_zeros(batch_size, max_num_global_attn_indices, self.h, self.d_k)

        key_only_global[is_local_index_global_attn_nonzero] = key[is_index_global_attn_nonzero]

        if query.size(1) == key_only_global.size(2):
            qT = query.transpose(1,2)
        else:
            qT = query
        
        attn_probs_from_global_key = torch.einsum("blhd,bshd->blhs", (qT, key_only_global))

        # need to transpose since ONNX export only supports consecutive indexing: https://pytorch.org/docs/stable/onnx.html#writes-sets
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)
        attn_probs_from_global_key[
            is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :
        ] = torch.finfo(attn_probs_from_global_key.dtype).min
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)

        return attn_probs_from_global_key

    def _compute_attn_output_with_global_indices(
        self,
        value: torch.Tensor,
        attn_probs: torch.Tensor,
        max_num_global_attn_indices: int,
        is_index_global_attn_nonzero: tuple,
        is_local_index_global_attn_nonzero: tuple,
        w: int,
    ) -> torch.Tensor:
        """
        Compute the attention output with global indices.

        Args:
            value (torch.Tensor): (batch, head, time, head_dim) The value vectors for global attention.
            attn_probs (torch.Tensor): (batch, time, head, 2w) The attention probabilities.
            max_num_global_attn_indices (int): Maximum number of global attention indices in the batch.
            is_index_global_attn_nonzero (tuple): Indices of global attention (non-zero elements).
            is_local_index_global_attn_nonzero (tuple): Non-padding values within global attention indices.
            w (int): Local context size
        Returns:
            torch.Tensor: (batch, time, head x head_dim) The attention output of all tokens attending to global.
        """
        batch_size, time = attn_probs.shape[0], attn_probs.shape[2]

        value = value.transpose(1, 2)

        # get value vectors for global only
        value_vectors_only_global = value.new_zeros(batch_size, max_num_global_attn_indices, self.h, self.d_k)
        value_vectors_only_global[is_local_index_global_attn_nonzero] = value[is_index_global_attn_nonzero]

        # cut local attn probs to global only
        attn_probs_only_global = attn_probs.narrow(-1, 0, max_num_global_attn_indices)
        # compute attn output only global
        attn_output_only_global = torch.matmul(
            attn_probs_only_global.clone(), value_vectors_only_global.transpose(1, 2).clone()
        ).transpose(1, 2)

        # reshape attn probs
        attn_probs_without_global = attn_probs.narrow(
            -1, max_num_global_attn_indices, attn_probs.size(-1) - max_num_global_attn_indices
        ).contiguous()

        # compute attn output with global
        attn_output_without_global = self.sliding_chunks_matmul_pv(attn_probs_without_global, value.transpose(1, 2), w)

        return attn_output_only_global + attn_output_without_global

    def _compute_out_global_to_all(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        max_num_global_attn_indices: int,
        is_local_index_global_attn_nonzero: tuple,
        is_index_global_attn_nonzero: tuple,
        is_local_index_no_global_attn_nonzero: tuple,
        is_index_masked: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the attention output of global tokens attending to all.

        Args:
            query (torch.Tensor): (batch, head, time, head_dim) The queries for global attention.
            key (torch.Tensor): (batch, head, time, head_dim) The keys for global attention.
            value (torch.Tensor): (batch, head, time, head_dim) The values for global attention.
            max_num_global_attn_indices (int): Maximum number of global attention indices in the batch.
            is_local_index_global_attn_nonzero (tuple): Non-padding values within global attention indices.
            is_index_global_attn_nonzero (tuple): Indices of global attention (non-zero elements).
            is_local_index_no_global_attn_nonzero (tuple): Padding values within global attention indices.
            is_index_masked (torch.Tensor): (batch, time) A boolean tensor indicating if an index is masked.

        Returns:
            global_attn_output (torch.Tensor): (batch, max_num_global_attn_indices, head x head_dim)
            The attention output of global tokens attending to all.
        """

        batch_size = key.shape[0]
        seq_len = key.shape[2]

        global_k = key.reshape(batch_size * self.h, -1, self.d_k)
        global_v = value.reshape(batch_size * self.h, -1, self.d_k)

        global_q = query.transpose(1, 2)
        global_q_from_global = global_q.new_zeros(batch_size, max_num_global_attn_indices, self.h, self.d_k)
        global_q_from_global[is_local_index_global_attn_nonzero] = global_q[is_index_global_attn_nonzero]
        global_q_from_global = global_q_from_global.transpose(0, 1).reshape(batch_size * self.h, -1, self.d_k)

        # compute attn scores
        global_attn_scores = torch.bmm(global_q_from_global, global_k.transpose(1, 2))
        global_attn_scores = global_attn_scores.view(batch_size, self.h, max_num_global_attn_indices, seq_len)

        # need to transpose since ONNX export only supports consecutive indexing: https://pytorch.org/docs/stable/onnx.html#writes-sets
        global_attn_scores = global_attn_scores.transpose(1, 2)
        global_attn_scores[
            is_local_index_no_global_attn_nonzero[0], is_local_index_no_global_attn_nonzero[1], :, :
        ] = torch.finfo(global_attn_scores.dtype).min
        global_attn_scores = global_attn_scores.transpose(1, 2)

        global_attn_scores = global_attn_scores.masked_fill(
            is_index_masked.transpose(2, 3),
            torch.finfo(global_attn_scores.dtype).min,
        )

        global_attn_scores = global_attn_scores.view(batch_size * self.h, max_num_global_attn_indices, seq_len)

        # compute global attn probs
        if self.training:
            global_attn_probs_float = nn.functional.softmax(global_attn_scores, dim=-1, dtype=torch.float32)
        else:
            global_attn_probs_float = nn.functional.softmax(global_attn_scores, dim=-1)

        global_attn_probs = self.dropout(global_attn_probs_float)

        # global attn output
        global_attn_output = torch.bmm(global_attn_probs, global_v)
        global_attn_output = global_attn_output.view(batch_size, self.h, max_num_global_attn_indices, self.d_k)

        global_attn_output = global_attn_output[
            is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]
        ]

        global_attn_output = global_attn_output.reshape(global_attn_output.shape[0], -1)

        return global_attn_output

    # Longformer implementation for overlap case
    #
    def _skew(self, x: torch.Tensor, direction: List[int], padding_value: float) -> torch.Tensor:
        """Convert diagonals into columns (or columns into diagonals depending on `direction`

        Args:
            x (torch.Tensor): (batch x head, chunk_count, 2w, 2w)
            direction (List[int]): padding directions
            padding_value (float): value to pad with

        Returns:
            output (torch.Tensor): (batch x head, chunk_count, 2w, 2w + 1)

        """
        x_padded = F.pad(x, direction, value=padding_value)
        #x_padded = x_padded.view(*x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2))
        x_padded = x_padded.view(x_padded.size()[0], x_padded.size(1), x_padded.size(-1), x_padded.size(-2))
        return x_padded

    def _skew2(self, x: torch.Tensor, padding_value: float) -> torch.Tensor:
        """Shift every row 1 step to right converting columns into diagonals

        Args:
            x (torch.Tensor): (batch x head, chunks_count + 1, w, 2w + 1)
            padding_value (float): value to pad with

        Returns:
            output (torch.Tensor): (batch x head, chunks_count + 1, w, 3w)
        """
        # X = B x C x M x L
        B, C, M, L = x.size()
        x = F.pad(x, (0, M + 1), value=padding_value)  # B x C x M x (L+M+1)
        x = x.view(B, C, -1)  # B x C x ML+MM+M
        x = x[:, :, :-M]  # B x C x ML+MM
        x = x.view(B, C, M, M + L)  # B x C, M x L+M
        x = x[:, :, :, :-1]
        return x

    def _chunk_overlap(self, x: torch.Tensor, w: int) -> torch.Tensor:
        """Convert into overlapping chunks.

        Args:
            x (torch.Tensor): # (batch x head, time, size)
            w (int): Chunk overlap size

        Returns:
            output (torch.Tensor): # (batch x head, chunk_count, 2w, size)
        """

        # non-overlapping chunks of size = 2w
        x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))

        # use `as_strided` to make the chunks overlap with an overlap size = w
        chunk_size = list(x.size())
        chunk_size[1] = chunk_size[1] * 2 - 1

        chunk_stride = list(x.stride())
        chunk_stride[1] = chunk_stride[1] // 2
        return x.as_strided(size=chunk_size, stride=chunk_stride)

    def _get_invalid_locations_mask(self, w: int, device: torch.device):

        diagonals_list = []
        for j in range(-w, 1):
            diagonal_mask = torch.zeros(w, device='cpu', dtype=torch.uint8)
            diagonal_mask[:-j] = 1
            diagonals_list.append(diagonal_mask)

        mask = torch.stack(diagonals_list, dim=-1)
        mask = mask[None, None, :, :]

        #ending_mask = mask.flip(dims=(2, 3)).bool().to(device)
        #return mask.bool().to(device), ending_mask

        ending_mask = mask.flip(dims=(2, 3)).to(torch.bool).to(device)
        return mask.to(torch.bool).to(device), ending_mask

    def mask_invalid_locations(
        self,
        input_tensor: torch.Tensor,
        w: int,
    ):
        """
        Mask locations invalid for the sliding window attention

        Args:
            input_tensor (torch.Tensor): # (batch x head, time, size)
            w (int): Chunk overlap size
        """
        beginning_mask, ending_mask = self._get_invalid_locations_mask(w, input_tensor.device)
        seq_len = input_tensor.size(2)
        beginning_input = input_tensor[:, :, :w, : w + 1]
        beginning_mask = beginning_mask[:, :, :seq_len].expand(beginning_input.size())
        beginning_input.masked_fill_(beginning_mask, -float('inf'))

        ending_input = input_tensor[:, :, -w:, -(w + 1) :]
        ending_mask = ending_mask[:, :, -seq_len:].expand(ending_input.size())
        ending_input.masked_fill_(ending_mask, -float('inf'))

    def sliding_chunks_matmul_qk(self, q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float) -> torch.Tensor:
        """Matrix multiplication of query x key tensors using with a sliding window attention pattern.
        This implementation splits the input into overlapping chunks of size 2w
        with an overlap of size w

        Args:
            q (torch.Tensor): (batch, head, time, size)
            k (torch.Tensor): (batch, head, time, size)
            w (int): Chunk overlap size
            padding_value (float): Value to pad with

        Returns:
            output (torch.Tensor): (batch, head, time, 2w + 1)
        """
        bsz, num_heads, seqlen, head_dim = q.size()
        assert seqlen % (w * 2) == 0
        assert q.size() == k.size()

        chunks_count = seqlen // w - 1
        # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size w * 2
        q = q.reshape(bsz * num_heads, seqlen, head_dim)
        k = k.reshape(bsz * num_heads, seqlen, head_dim)

        chunk_q = self._chunk_overlap(q, w)  # (batch x head, chunk_count, 2w, size)
        chunk_k = self._chunk_overlap(k, w)  # (batch x head, chunk_count, 2w, size)

        # matrix multipication
        # bcxd: bsz*num_heads x chunks x 2w x head_dim
        # bcyd: bsz*num_heads x chunks x 2w x head_dim
        # bcxy: bsz*num_heads x chunks x 2w x 2w
        chunk_attn = torch.einsum('bcxd,bcyd->bcxy', (chunk_q, chunk_k))  # multiply
        # (batch x head, chunk_count, 2w, 2w)

        # convert diagonals into columns
        diagonal_chunk_attn = self._skew(chunk_attn, direction=(0, 0, 0, 1), padding_value=padding_value)
        # (batch x head, chunk_count, 2w, 2w + 1)

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (w * 2 + 1) columns. The first (w) columns are the w lower triangles (attention from a word to
        # w previous words). The following column is attention score from each word to itself, then
        # followed by w columns for the upper triangle.

        diagonal_attn = diagonal_chunk_attn.new_empty((bsz * num_heads, chunks_count + 1, w, w * 2 + 1))
        # (batch x head, chunk_count + 1, w, 2w + 1)

        # copy parts from diagonal_chunk_attn into the compined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, : w + 1]
        diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, : w + 1]
        # - copying the lower triangle
        diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, -(w + 1) : -1, w + 1 :]
        diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, : w - 1, 1 - w :]

        # separate bsz and num_heads dimensions again
        diagonal_attn = diagonal_attn.view(bsz, num_heads, seqlen, 2 * w + 1)
        # (batch, head, time, 2w + 1)

        self.mask_invalid_locations(diagonal_attn, w)

        return diagonal_attn

    def sliding_chunks_matmul_pv(self, prob: torch.Tensor, v: torch.Tensor, w: int):
        """Same as sliding_chunks_matmul_qk but for prob and value tensors.

        Args:
            prob (torch.Tensor): (batch, head, time, size)
            v (torch.Tensor): (batch, head, time, size)
            w (int): Chunk overlap size

        Returns:
            output (torch.Tensor): (batch, time, head, size)
        """
        bsz, num_heads, seqlen, head_dim = v.size()
        chunks_count = seqlen // w - 1
        # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size 2w
        chunk_prob = prob.reshape(bsz * num_heads, seqlen // w, w, 2 * w + 1)
        # (batch x head, chunks_count + 1, w, 2w + 1)

        # group bsz and num_heads dimensions into one
        v = v.reshape(bsz * num_heads, seqlen, head_dim)
        # (batch x head, time, size)

        # pad seqlen with w at the beginning of the sequence and another w at the end
        padded_v = F.pad(v, (0, 0, w, w), value=-1)
        # (batch x head, time + 2w, size)

        # chunk padded_v into chunks of size 3w and an overlap of size w
        chunk_v_size = (bsz * num_heads, chunks_count + 1, 3 * w, head_dim)
        chunk_v_stride = padded_v.stride()
        chunk_v_stride = chunk_v_stride[0], w * chunk_v_stride[1], chunk_v_stride[1], chunk_v_stride[2]
        chunk_v = padded_v.as_strided(size=chunk_v_size, stride=chunk_v_stride)
        # (batch x head, chunks_count + 1, 3w, size)

        skewed_prob = self._skew2(chunk_prob, padding_value=0)
        # (batch x head, chunks_count + 1, w, 3w)
        context = torch.einsum('bcwd,bcdh->bcwh', (skewed_prob, chunk_v))
        # (batch x head, chunks_count + 1, w, size)

        return context.view(bsz, num_heads, seqlen, head_dim).transpose(1, 2)

class ESPnetRelPositionMultiHeadedAttention(MultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (legacy version).
    Details can be found in https://github.com/espnet/espnet/pull/2816.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
    n_head (int): The number of heads.
    n_feat (int): The number of features.
    dropout_rate (float): Dropout rate.
    zero_triu (bool): Whether to zero the upper triangular part of attention matrix.                                                                                                                                                            
    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False, layer_id:int = 0):
        """Construct an ESPnetRelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, time2).

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
                               device=x.device,
                               dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(x.size()[0],
                                 x.size()[1],
                                 x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: Optional[torch.Tensor],
                pos_emb: torch.Tensor):
        """Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor (#batch, time1, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask)
