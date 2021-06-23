# coding=utf-8
# Copyright NVIDIA and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" SegFormer model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

SEGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "nvidia/segformer-b0-fine-tuned-ade-512-512": "https://huggingface.co/nvidia/segformer-b0-fine-tuned-ade-512-512/resolve/main/config.json",
    # See all SegFormer models at https://huggingface.co/models?filter=segformer
}


class SegFormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.SegFormerModel`.
    It is used to instantiate a SegFormer model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the SegFormer `nvidia/segformer-b0-fine-tuned-ade-512-512 <https://huggingface.co/nvidia/segformer-b0-fine-tuned-ade-512-512>`__ 
    architecture.

    Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
    to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
    for more information.


    Args:
        image_size (:obj:`int`, `optional`, defaults to 224):
            The size (resolution) of each image.
        num_channels (:obj:`int`, `optional`, defaults to 3):
            The number of input channels.
        num_encoder_blocks (:obj:`int`, `optional`, defaults to 4):
            The number of encoder blocks (i.e. stages in the Mix Transformer encoder). 
        depths (:obj:`List[int]`, `optional`, defaults to [2, 2, 2, 2]):
            The number of layers in each encoder block.
        sr_ratios (:obj:`List[int]`, `optional`, defaults to [8, 4, 2, 1]):
            Sequence reduction ratios in each encoder block.
        hidden_sizes (:obj:`List[int]`, `optional`, defaults to [32, 64, 160, 256]):
            Dimension of each of the encoder blocks.
        downsampling_rates (:obj:`List[int]`, `optional`, defaults to [1, 4, 8, 16]):
            Downsample rate of the image resolution before each encoder block.
        patch_sizes (:obj:`List[int]`, `optional`, defaults to [7, 3, 3, 3]):
            Patch size before each encoder block.
        strides (:obj:`List[int]`, `optional`, defaults to [4, 2, 2, 2]):
            Stride before each encoder block.
        encoder_attention_heads (:obj:`List[int]`, `optional`, defaults to [1, 2, 4, 8]):
            Number of attention heads for each attention layer in each block of the Transformer encoder.
        mlp_ratio (:obj:`List[int]`, `optional`, defaults to [4, 4, 4, 4]):
            Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the encoder blocks.
        activation_function (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        drop_path_rate (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for stochastic depth, used in the blocks of the Transformer encoder. 
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        decoder_hidden_size (:obj:`int`, `optional`, defaults to 256):
            The dimension of the decoder.
        init_std (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

        Example::

        >>> from transformers import SegFormerModel, SegFormerConfig

        >>> # Initializing a SegFormer nvidia/segformer-b0-fine-tuned-ade-512-512 style configuration
        >>> configuration = SegFormerConfig()

        >>> # Initializing a model from the nvidia/segformer-b0-fine-tuned-ade-512-512 style configuration
        >>> model = SegFormerModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "segformer"
    def __init__(
        self,
        image_size=224,
        num_channels=3,
        num_encoder_blocks=4,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[32, 64, 160, 256],
        downsampling_rates=[1, 4, 8, 16],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        encoder_attention_heads=[1, 2, 4, 8],
        mlp_ratio=[4, 4, 4, 4],
        encoder_ffn_dim=4096,
        activation_function="gelu",
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        drop_path_rate=0.1,
        layer_norm_eps=1e-6,
        decoder_hidden_size=256,
        init_std=0.02,
        is_encoder_decoder=True,
        scale_embedding=False,
        **kwargs
    ):
        super().__init__(
            is_encoder_decoder=is_encoder_decoder,
            **kwargs
        )

        self.image_size = image_size
        self.num_channels = num_channels
        self.num_encoder_blocks = num_encoder_blocks
        self.depths = depths
        self.sr_ratios = sr_ratios
        self.hidden_sizes = hidden_sizes
        self.downsampling_rates = downsampling_rates
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.encoder_attention_heads = encoder_attention_heads
        self.mlp_ratio = mlp_ratio
        self.encoder_ffn_dim = encoder_ffn_dim
        self.activation_function = activation_function
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.drop_path_rate = drop_path_rate
        self.layer_norm_eps = layer_norm_eps
        self.decoder_hidden_size = decoder_hidden_size
        self.init_std = init_std
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True