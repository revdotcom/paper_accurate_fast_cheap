# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#               2024 Alan (alanfangemail@gmail.com)
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
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Encoder definition with lora."""

from typing import Optional, List
import logging

import torch

from wenet.transformer.convolution import ConvolutionModule
from wenet.transformer.encoder import TransformerEncoder, ConformerEncoder, LanguageSpecificConformerEncoder
from wenet.transformer.encoder_layer import TransformerEncoderLayer
from wenet.transformer.encoder_layer import ConformerEncoderLayer, LanguageSpecificConformerEncoderLayer
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward

from wenet.utils.class_utils import (
    WENET_MLP_CLASSES,
    WENET_ACTIVATION_CLASSES,
)
from wenet.finetune.lora.utils import WENET_LORA_ATTENTION_CLASSES


class LoRATransformerEncoder(TransformerEncoder):
    """Transformer encoder module with lora."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "abs_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        key_bias: bool = True,
        activation_type: str = "relu",
        gradient_checkpointing: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        lora_list: Optional[List[str]] = None,
    ):
        """ Construct TransformerEncoder

        See Encoder for the meaning of each parameter.
        """
        super().__init__(input_size, output_size, attention_heads,
                         linear_units, num_blocks, dropout_rate,
                         positional_dropout_rate, attention_dropout_rate,
                         input_layer, pos_enc_layer_type, normalize_before,
                         static_chunk_size, use_dynamic_chunk, global_cmvn,
                         use_dynamic_left_chunk, key_bias,
                         activation_type, gradient_checkpointing)
        activation = WENET_ACTIVATION_CLASSES[activation_type]()
        mlp_class = PositionwiseFeedForward
        self.encoders = torch.nn.ModuleList([
            TransformerEncoderLayer(
                output_size,
                WENET_LORA_ATTENTION_CLASSES["selfattn"](
                    attention_heads, output_size, attention_dropout_rate,
                    key_bias, lora_rank, lora_alpha, lora_dropout, lora_list),
                mlp_class(output_size, linear_units, dropout_rate, activation),
                dropout_rate,
                normalize_before
            ) for _ in range(num_blocks)
        ])


class LoRAConformerEncoder(ConformerEncoder):
    """Conformer encoder module with lora."""

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        key_bias: bool = True,
        gradient_checkpointing: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        lora_list: Optional[List[str]] = None,
        num_langs: int = 0,
    ):
        """Construct ConformerEncoder

        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            selfattention_layer_type (str): Encoder attention layer type,
                the parameter has no effect now, it's just for configure
                compatibility.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
            key_bias: whether use bias in attention.linear_k, False for whisper models.
        """
        super().__init__(
            input_size, output_size, attention_heads, linear_units, num_blocks,
            dropout_rate, positional_dropout_rate, attention_dropout_rate,
            input_layer, pos_enc_layer_type, normalize_before,
            static_chunk_size, use_dynamic_chunk, global_cmvn,
            use_dynamic_left_chunk, positionwise_conv_kernel_size,
            macaron_style, selfattention_layer_type, activation_type,
            use_cnn_module, cnn_module_kernel, causal, cnn_module_norm, key_bias,
            gradient_checkpointing, lora_rank, lora_alpha, lora_dropout, lora_list, num_langs)
        activation = WENET_ACTIVATION_CLASSES[activation_type]()

        # self-attention module definition
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            key_bias,
            lora_rank,
            lora_alpha,
            lora_dropout,
            lora_list,
        )
        # feed-forward module definition
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        # convolution module definition
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal)

        num_regular_blocks = num_blocks if self.num_langs == 0 else num_blocks - 2
        
        mlp_class = PositionwiseFeedForward
        self.encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                WENET_LORA_ATTENTION_CLASSES[selfattention_layer_type](
                    *encoder_selfattn_layer_args),
                mlp_class(*positionwise_layer_args),
                mlp_class(*positionwise_layer_args) if macaron_style else None,
                ConvolutionModule(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
            ) for _ in range(num_regular_blocks)
        ])

        if self.num_langs > 1:
            # TODO : JPR : consider having more than one layer at the begining and at the end ?

            logging.info(f"Tranforming the ConformerEncoder to a LanguageSpecificConformerEncoder because num_langs is {self.num_langs}")
            self.encoders.insert(0, LanguageSpecificConformerEncoderLayer(
                output_size,
                WENET_LORA_ATTENTION_CLASSES[selfattention_layer_type](
                    *encoder_selfattn_layer_args),
                mlp_class(*positionwise_layer_args),
                mlp_class(
                    *positionwise_layer_args) if macaron_style else None,
                ConvolutionModule(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                num_langs))

            self.encoders.append(LanguageSpecificConformerEncoderLayer(
                output_size,
                WENET_LORA_ATTENTION_CLASSES[selfattention_layer_type](
                    *encoder_selfattn_layer_args),
                mlp_class(*positionwise_layer_args),
                mlp_class(
                    *positionwise_layer_args) if macaron_style else None,
                ConvolutionModule(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                num_langs))


class LoRALanguageSpecificConformerEncoder(LanguageSpecificConformerEncoder):

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        pos_enc_layer_type: str = "rel_pos",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        positionwise_conv_kernel_size: int = 1,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        key_bias: bool = True,
        gradient_checkpointing: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 8,
        lora_dropout: float = 0.0,
        lora_list: Optional[List[str]] = None,
        num_langs: int = 3,
    ):
        """Construct LanguageSpecificConformerEncoder

        Args:
            input_size to use_dynamic_chunk, see in BaseEncoder
            positionwise_conv_kernel_size (int): Kernel size of positionwise
                conv1d layer.
            macaron_style (bool): Whether to use macaron style for
                positionwise layer.
            selfattention_layer_type (str): Encoder attention layer type,
                the parameter has no effect now, it's just for configure
                compatibility.
            activation_type (str): Encoder activation function type.
            use_cnn_module (bool): Whether to use convolution module.
            cnn_module_kernel (int): Kernel size of convolution module.
            causal (bool): whether to use causal convolution or not.
            key_bias: whether use bias in attention.linear_k, False for whisper models.
        """
        print("HI NAT in LoRA LSL conformer encoder, gradient checkpointing is ", gradient_checkpointing)
        super().__init__(
            input_size, output_size, attention_heads, linear_units, num_blocks,
            dropout_rate, positional_dropout_rate, attention_dropout_rate,
            input_layer, pos_enc_layer_type, normalize_before,
            static_chunk_size, use_dynamic_chunk, global_cmvn,
            use_dynamic_left_chunk, positionwise_conv_kernel_size,
            macaron_style, selfattention_layer_type, activation_type,
            use_cnn_module, cnn_module_kernel, causal, cnn_module_norm,
            key_bias, gradient_checkpointing, lora_rank, lora_alpha, lora_dropout, lora_list, num_langs)
        activation = WENET_ACTIVATION_CLASSES[activation_type]()

        # self-attention module definition
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            key_bias,
            lora_rank,
            lora_alpha,
            lora_dropout,
            lora_list,
        )
        # feed-forward module definition
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        # convolution module definition
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal)

        mlp_class = PositionwiseFeedForward

        # TODO : JPR : consider having more than one layer at the begining and at the end ?
        self.encoders.insert(0, LanguageSpecificConformerEncoderLayer(
            output_size,
            WENET_LORA_ATTENTION_CLASSES[selfattention_layer_type](
                *encoder_selfattn_layer_args),
            mlp_class(*positionwise_layer_args),
            mlp_class(
                *positionwise_layer_args) if macaron_style else None,
            ConvolutionModule(
                *convolution_layer_args) if use_cnn_module else None,
            dropout_rate,
            normalize_before,
            num_langs))

        self.encoders.append(LanguageSpecificConformerEncoderLayer(
            output_size,
            WENET_LORA_ATTENTION_CLASSES[selfattention_layer_type](
                *encoder_selfattn_layer_args),
            mlp_class(*positionwise_layer_args),
            mlp_class(
                *positionwise_layer_args) if macaron_style else None,
            ConvolutionModule(
                *convolution_layer_args) if use_cnn_module else None,
            dropout_rate,
            normalize_before,
            num_langs))
