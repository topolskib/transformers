# coding=utf-8
# Copyright 2021 Google AI and The HuggingFace Inc. team. All rights reserved.
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
""" CvT model configuration """

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

CVT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/cvt-13-224-224": "https://huggingface.co/microsoft/cvt-13-224-224/resolve/main/config.json",
    # See all CvT models at https://huggingface.co/models?filter=cvt
}


class CvtConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.CvtModel`. It is used to
    instantiate an CvT model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the CvT `microsoft/cvt-13-224-224
    <https://huggingface.co/google/microsoft/cvt-13-224-224>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"selu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        image_size (:obj:`int`, `optional`, defaults to :obj:`224`):
            The size (resolution) of each image.
        patch_size (:obj:`int`, `optional`, defaults to :obj:`16`):
            The size (resolution) of each patch.
        num_channels (:obj:`int`, `optional`, defaults to :obj:`3`):
            The number of input channels.


    Example::

        >>> from transformers import CvtModel, CvtConfig

        >>> # Initializing a CvT microsoft/cvt-13-224-224 style configuration
        >>> configuration = CvtConfig()

        >>> # Initializing a model from the microsoft/cvt-13-224-224 style configuration
        >>> model = CvtModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "cvt"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=False,
        # additional properties
        in_chans=3,
        init="trunc_norm",
        num_classes=1000,
        num_stages=3,
        patch_size=[7, 3, 3],
        patch_stride=[4, 2, 2],
        patch_padding=[2, 1, 1],
        dim_embed=[64, 192, 384],
        num_heads=[1, 3, 6],
        depth=[1, 4, 16],
        mlp_ratio=[4.0, 4.0, 4.0],
        attn_drop_rate=[0.0, 0.0, 0.0],
        drop_rate=[0.0, 0.0, 0.0],
        drop_path_rate=[0.0, 0.0, 0.1],
        qkv_bias=[True, True, True],
        cls_token=[False, False, True],
        pos_embed=[False, False, False],
        qkv_proj_method=["dw_bn", "dw_bn", "dw_bn"],
        kernel_qkv=[3, 3, 3],
        padding_kv=[1, 1, 1],
        stride_kv=[2, 2, 2],
        padding_q=[1, 1, 1],
        stride_q=[1, 1, 1],
        act_layer="gelu",
        norm_layer="layer_norm",
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        # additional properties
        self.in_chans = in_chans
        self.num_stages = num_stages
        self.num_classes = num_classes
        self.init = init
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.dim_embed = dim_embed
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.attn_drop_rate = attn_drop_rate
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.qkv_bias = qkv_bias
        self.cls_token = cls_token
        self.pos_embed = pos_embed
        self.qkv_proj_method = qkv_proj_method
        self.kernel_qkv = kernel_qkv
        self.padding_kv = padding_kv
        self.stride_kv = stride_kv
        self.padding_q = padding_q
        self.stride_q = stride_q
        self.act_layer = act_layer
        self.norm_layer = norm_layer