# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" VitDet model configuration"""


from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import get_aligned_output_features_output_indices


logger = logging.get_logger(__name__)

VITDET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/vit-det-base": "https://huggingface.co/facebook/vit-det-base/resolve/main/config.json",
}


class VitDetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VitDetModel`]. It is used to instantiate an
    VitDet model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the VitDet
    [google/vitdet-base-patch16-224](https://huggingface.co/google/vitdet-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        mlp_ratio (`int`, *optional*, defaults to 4):
            Ratio of mlp hidden dim to embedding dim.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        drop_path_rate (`float`, *optional*, defaults to `0.0`):
            Stochastic depth rate.
        window_block_indices (`List[int]`, *optional*):
            List of indices of blocks that should have window attention instead of regular global self-attention.
        residual_block_indices (`List[int]`, *optional*):
            List of indices of blocks that should have an extra residual block after the MLP.
        use_absolute_position_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to add absolute position embeddings to the patch embeddings.
        use_relative_position_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to add relative position embeddings to the attention maps.

    Example:

    ```python
    >>> from transformers import VitDetConfig, VitDetModel

    >>> # Initializing a VitDet vitdet-base-patch16-224 style configuration
    >>> configuration = VitDetConfig()

    >>> # Initializing a model (with random weights) from the vitdet-base-patch16-224 style configuration
    >>> model = VitDetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "vitdet"

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        mlp_ratio=4,
        hidden_act="gelu",
        dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        drop_path_rate=0.0,
        window_block_indices=[],
        residual_block_indices=[],
        use_absolute_position_embeddings=True,
        use_relative_position_embeddings=False,
        out_features=None,
        out_indices=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.mlp_ratio = mlp_ratio
        self.hidden_act = hidden_act
        self.dropout_prob = dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.drop_path_rate = drop_path_rate
        self.window_block_indices = window_block_indices
        self.residual_block_indices = residual_block_indices
        self.use_absolute_position_embeddings = use_absolute_position_embeddings
        self.use_relative_position_embeddings = use_relative_position_embeddings

        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_hidden_layers + 1)]
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
