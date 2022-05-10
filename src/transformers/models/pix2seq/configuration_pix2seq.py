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
""" Pix2Seq model configuration"""

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

PIX2SEQ_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/pix2seq-vit-base": "https://huggingface.co/google/pix2seq-vit-base/resolve/main/config.json",
}



class Pix2SeqConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Pix2SeqModel`]. It is used to instantiate an
    Pix2Seq model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Pix2Seq
    [google/pix2seq-base-patch16-224](https://huggingface.co/google/pix2seq-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        patch_size (`int`, *optional*, defaults to `16`):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to `3`):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        use_cls_token (`bool`, *optional*, defaults to `False`):
            Whether to use a CLS token.
        positional_encoding (`str`, *optional*, defaults to `sin_cos`):
            Position encoding type to use. Either "learned" or (2D) "sin_cos".
        position_encoding_decoder (`str`, *optional*, defaults to `learned`):
            Position encoding type to use in the decoder. Either "learned" or (1D) "sin_cos".
        max_position_embeddings (`int`, *optional*, defaults to `512`):
            The maximum sequence length in the decoder.
        num_decoder_layers (`int`, *optional*, defaults to `6`):
            The number of layers in the decoder.
        num_attention_heads_decoder (`int`, *optional*, defaults to `16`):
            The number of attention heads in the decoder.

    Example:

    ```python
    >>> from transformers import Pix2SeqModel, Pix2SeqConfig

    >>> # Initializing a Pix2Seq pix2seq-base-patch16-224 style configuration
    >>> configuration = Pix2SeqConfig()

    >>> # Initializing a model from the pix2seq-base-patch16-224 style configuration
    >>> model = Pix2SeqModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "pix2seq"

    def __init__(
        self,
        num_channels=3,
        image_size=640,
        patch_size=16,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        qkv_bias=True,
        use_cls_token=False,
        positional_encoding="sin_cos",
        dec_proj_mode="mlp",
        drop_path_rate=0.1,
        vocab_size=3000,
        dim_decoder=512,
        positional_encoding_decoder="learned",
        max_position_embeddings=512,
        num_decoder_layers=6,
        num_attention_heads_decoder=16,
        output_bias = True,
        use_cache = True,
        tie_word_embeddings=False,
        is_encoder_decoder=True,
        bos_token_id=10,
        **kwargs
    ):
        super().__init__(is_encoder_decoder=is_encoder_decoder, bos_token_id=bos_token_id, **kwargs)

        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.use_cls_token = use_cls_token
        self.positional_encoding = positional_encoding
        self.dec_proj_mode = dec_proj_mode
        self.drop_path_rate = drop_path_rate
        # decoder attributes
        self.dim_decoder = dim_decoder
        self.vocab_size = vocab_size
        self.positional_encoding_decoder = positional_encoding_decoder
        self.max_position_embeddings = max_position_embeddings
        self.num_decoder_layers = num_decoder_layers
        self.num_attention_heads_decoder = num_attention_heads_decoder
        self.output_bias = output_bias
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
