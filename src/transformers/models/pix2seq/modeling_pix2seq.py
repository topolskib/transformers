# coding=utf-8
# Copyright 2022 Google AI and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Pix2Seq model."""


import collections.abc
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_pix2seq import Pix2SeqConfig


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "Pix2SeqConfig"
_FEAT_EXTRACTOR_FOR_DOC = "ViTFeatureExtractor"

# Base docstring
_CHECKPOINT_FOR_DOC = "google/pix2seq-vit-base"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]


PIX2SEQ_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/pix2seq-vit-base",
    # See all Pix2Seq models at https://huggingface.co/models?filter=pix2seq
]


# Copied from transformers.models.vit.modeling_vit.to_2tuple
def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


def get_angles(pos, i, dim):
    angle_rates = 1 / torch.pow(10000.0, (2 * (i // 2).float()) / dim)
    return pos.float() * angle_rates.float()


def positional_encoding(coords, dim):
    """
    Args:
        coords:
            Coordinates in (batch_size, dimension)

        Returns:
            Position embeddings of shape (bsz, size, dim).
    """
    angle_rads = get_angles(coords.unsqueeze(-1), torch.arange(start=0, end=dim)[None, None, :], dim)

    # apply sin to even indices in the array; 2i
    angle_rads1 = torch.sin(angle_rads[:, :, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads2 = torch.cos(angle_rads[:, :, 1::2])

    pos_encoding = torch.cat([angle_rads1, angle_rads2], -1)

    return pos_encoding.float()


def get_1d_position_codes(seqlen, out_dim, normalization_max=6.2831852):
    """Get 1d positional embedding with sin/cos codes.

    Args:
        seqlen:
            An `int` specifying the length of the sequence.
        out_dim:
            An `int` specifying the output dimension of the encoding.
        normalization_max:
            Normalize coordinates between [0, normalization_max]. If None, raw coordinates from 0 to seqlen will be
            used.

    Returns:
        positional code of shape (1, seqlen, out_dim)
    """
    coords = torch.arange(seqlen, dtype=torch.float)
    if normalization_max is not None:
        coords = coords / (seqlen - 1) * normalization_max
    coords = positional_encoding(coords, out_dim)
    return coords


def get_2d_position_codes(height, width, out_dim, normalization_max=6.2831852):
    """Get 2d positional embedding with sin/cos codes.

    Args:
        height:
            An `int` specifying the height of the 2d image / feature map.
        width:
            An `int` specifying the width of the 2d image / feature map.
        out_dim:
            An `int` specifying the output dimension of the encoding. Must be divisible by 2.
        normalization_max:
            ?ormalize coordinates between [0, normalization_max]. If None, raw coordinates from 0 to height/width will
            be used.

    Returns:
        positional code of shape (1, height, width, out_dim)
    """
    y_coords = torch.arange(start=0, end=height, dtype=torch.float)
    if normalization_max is not None:
        y_coords = y_coords / (height - 1) * normalization_max
    y_coords = positional_encoding(y_coords, out_dim // 2)
    y_coords = y_coords.unsqueeze(2)
    y_coords = torch.cat([y_coords, torch.zeros_like(y_coords)], -1)

    x_coords = torch.arange(start=0, end=width, dtype=torch.float)
    if normalization_max is not None:
        x_coords = x_coords / (width - 1) * normalization_max
    x_coords = positional_encoding(x_coords, out_dim // 2)
    x_coords = x_coords.unsqueeze(1)
    x_coords = torch.cat([torch.zeros_like(x_coords), x_coords], -1)

    return y_coords + x_coords


class Pix2SeqPatchPositionEmbeddings(nn.Module):
    """
    Position embeddings to be added to the patch tokens.
    """

    def __init__(self, config: Pix2SeqConfig, dim):
        super().__init__()

        n_rows = n_cols = config.image_size // config.patch_size

        if config.positional_encoding == "learned":
            self.embeddings = nn.Parameter(torch.randn(n_rows * n_cols, dim))
        elif config.positional_encoding == "sin_cos":
            sin_cos = get_2d_position_codes(n_rows, n_cols, dim, normalization_max=6.2831852)
            self.embeddings = torch.reshape(sin_cos, [n_rows * n_cols, dim])
        else:
            raise ValueError("Unknown positional encoding:", config.positional_encoding)

    def forward(self):
        return self.embeddings


class Pix2SeqSequencePositionEmbeddings(nn.Module):
    """
    Position embeddings to be added to the text tokens.
    """

    def __init__(self, config: Pix2SeqConfig):
        super().__init__()

        dim = config.dim_decoder

        if config.positional_encoding_decoder == "learned":
            self.embeddings = nn.Parameter(torch.randn(config.max_position_embeddings, dim))
        elif config.positional_encoding_decoder == "sin_cos":
            sin_cos = get_1d_position_codes(config.max_position_embeddings, dim, normalization_max=6.2831852)
            self.embeddings = torch.reshape(sin_cos, [config.max_position_embeddings, dim])
        else:
            raise ValueError("Unknown positional encoding:", config.positional_encoding)

    def forward(self):
        return self.embeddings


class Pix2SeqEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config: Pix2SeqConfig) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if config.use_cls_token else None
        self.patch_embeddings = PatchEmbeddings(
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )
        # num_patches = self.patch_embeddings.num_patches
        # self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        self.position_embeddings = Pix2SeqPatchPositionEmbeddings(config, dim=config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        # print("Shape of patch embeddings:", embeddings.shape)

        batch_size, seq_len, _ = embeddings.size()

        if self.config.use_cls_token:
            # add the [CLS] token to the embedded patch tokens
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings().unsqueeze(0)

        # print("First values after position embeddings:", embeddings[0,:3,:3])
        # print("Last values after position embeddings:", embeddings[0,-3:,:3:])
        # print("Sum of embeddings:", embeddings.sum())

        embeddings = self.dropout(embeddings)

        return embeddings


class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.

    """

    def __init__(
        self,
        patch_size: Union[int, Tuple[int, int]] = 16,
        num_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.projection = nn.Conv2d(
            num_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding="valid"
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape

        embeddings = self.projection(pixel_values)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = self.layer_norm(embeddings)

        return embeddings


class Pix2SeqEncoderLayer(nn.Module):
    def __init__(self, config: Pix2SeqConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layernorm_before = nn.LayerNorm(self.embed_dim)
        self.self_attn = Pix2SeqAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads_encoder,
            dropout=config.attention_probs_dropout_prob,
        )
        self.drop_path = Pix2SeqDropPath(config.drop_path_rate)

        # MLP
        self.layernorm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, 4 * self.embed_dim)
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(self.embed_dim * 4, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        layer_head_mask: torch.FloatTensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(num_attention_heads_encoder,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layernorm_before(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            queries=hidden_states,
            keys=hidden_states,
            values=hidden_states,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + self.drop_path(hidden_states)

        # MLP
        residual = hidden_states

        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + self.drop_path(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Pix2SeqEncoder(nn.Module):
    def __init__(self, config: Pix2SeqConfig) -> None:
        super().__init__()
        self.config = config

        self.embeddings = Pix2SeqEmbeddings(config)

        self.layer = nn.ModuleList([Pix2SeqEncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        pixel_values: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if hidden_states is None:
            hidden_states = self.embeddings(pixel_values)

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask=layer_head_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, layer_head_mask=layer_head_mask, output_attentions=output_attentions
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# Copied from transformers.models.convnext.modeling_convnext.drop_path
def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep=True):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks). This is the same as the
    DropConnect impl I created for EfficientNet, etc networks, however, the original name is misleading as 'Drop
    Connect' is a different form of dropout in a separate paper... See discussion:
    https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the layer and
    argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.convnext.modeling_convnext.ConvNextDropPath
class Pix2SeqDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Pix2SeqProjectionMLP(nn.Module):
    """
    Pix2Seq projection MLP, only used if config.dec_proj_mode == "mlp".
    """

    def __init__(self, config: Pix2SeqConfig, mlp_ratio=4) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(config.dim_decoder, eps=config.layer_norm_eps)
        self.dense1 = nn.Linear(config.dim_decoder, config.dim_decoder * mlp_ratio)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout()
        self.dense2 = nn.Linear(config.dim_decoder * mlp_ratio, config.dim_decoder)
        self.drop_path = Pix2SeqDropPath(config.drop_path_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.layernorm(hidden_states)

        # print("First values of hidden states after layernorm:", hidden_states[0,:3,:3])

        hidden_states = self.dense1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)

        # print("First values of hidden states as residual:", hidden_states[0,:3,:3])

        hidden_states = residual + self.drop_path(hidden_states)

        return hidden_states


class Pix2SeqProjection(nn.Module):
    """
    A class to prepare the final hidden states of the encoder to be used by the decoder.

    3 options are possible:
    - linear (just a linear layer)
    - linear_p (linear layer + position embeddings)
    - mlp (position embeddings + MLP)
    """

    def __init__(self, config: Pix2SeqConfig) -> None:
        super().__init__()
        self.projection = nn.Linear(config.hidden_size, config.dim_decoder)
        self.layernorm = nn.LayerNorm(config.dim_decoder)

        if config.dec_proj_mode in ["linear_p", "mlp"]:
            self.position_embeddings = Pix2SeqPatchPositionEmbeddings(config, dim=config.dim_decoder)
        if config.dec_proj_mode == "mlp":
            self.projection_mlp = Pix2SeqProjectionMLP(config)

        self.config = config

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.projection(hidden_states)
        hidden_states = self.layernorm(hidden_states)

        # print("First values of hidden states after linear projection + ln:", hidden_states[0,:3,:3])

        # Add (optional) positional embedding to encoded visual units.
        if self.config.dec_proj_mode != "linear":
            position_embeddings = self.position_embeddings().unsqueeze(0)
            if self.config.use_cls_token:
                hidden_states = hidden_states + torch.cat(
                    [torch.zeros_like(position_embeddings[:, :1]), position_embeddings], 1
                )
            else:
                hidden_states = hidden_states + position_embeddings

            # print("First values of hidden states after position embeddings:", hidden_states[0,:3,:3])
            # print("Last values of hidden states after position embeddings:", hidden_states[0,-3:,-3:])

            if self.config.dec_proj_mode == "mlp":
                hidden_states = self.projection_mlp(hidden_states)
            else:
                assert self.config.dec_proj_mode == "linear_p"

        # "First values of hidden states after projection MLP:", hidden_states[0,:3,:3])

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTPreTrainedModel with ViT->Pix2Seq,vit->pix2seq
class Pix2SeqPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Pix2SeqConfig
    base_model_prefix = "pix2seq"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module: Pix2SeqEncoder, value: bool = False) -> None:
        if isinstance(module, Pix2SeqEncoder):
            module.gradient_checkpointing = value


class Pix2SeqAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        queries,
        keys,
        values,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, _ = queries.size()

        # get query proj
        query_states = self.q_proj(queries) * self.scaling
        key_states = self._shape(self.k_proj(keys), -1, bsz)
        value_states = self._shape(self.v_proj(values), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class Pix2SeqDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.dim_decoder

        self.self_attn = Pix2SeqAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads_decoder,
            dropout=config.attention_probs_dropout_prob,
            is_decoder=True,
        )

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.drop_path = Pix2SeqDropPath(config.drop_path_rate)
        self.encoder_attn = Pix2SeqAttention(
            self.embed_dim,
            config.num_attention_heads_decoder,
            dropout=config.attention_probs_dropout_prob,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # MLP
        self.layernorm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, 4 * self.embed_dim)
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(self.embed_dim * 4, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        print_values=False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(num_attention_heads_encoder,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`torch.FloatTensor)`):
                tensor of shape (batch size, seq len, hidden size): cached past hidden states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = keys = values = self.self_attn_layer_norm(hidden_states)

        # this corresponds to "x_for_cache" in the original implementation
        present_key_value = hidden_states

        if past_key_value is not None:
            # Augment kv_ln with cache in (bsz, c_size, d).
            # q_size, k_size = tf.shape(x)[1], tf.shape(cache)[1]
            # mask_self = tf.concat([tf.ones([1, 1, q_size, k_size]), mask_self], -1)
            keys = values = torch.cat([past_key_value, hidden_states], axis=1)
            #  print("Shape of past key value:", past_key_value.shape)
            # print("Shape of keys:", keys.shape)

        hidden_states, self_attn_weights = self.self_attn(
            queries=hidden_states,
            keys=keys,
            values=values,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        # if print_values:
        #     print("Hidden states after self-attention:", hidden_states[0,:3,:3])

        hidden_states = residual + self.drop_path(hidden_states)

        # Cross-Attention Block
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # if print_values:
        #     print("Hidden states after cross-attention layer norm:", hidden_states[0,:3,:3])

        cross_attn_weights = None
        if encoder_hidden_states is not None:
            hidden_states, cross_attn_weights = self.encoder_attn(
                queries=hidden_states,
                keys=encoder_hidden_states,
                values=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                output_attentions=output_attentions,
            )

            # if print_values:
            #     print("Hidden states after cross-attention:", hidden_states[0,:3,:3])

            hidden_states = residual + self.drop_path(hidden_states)

        # MLP
        # if print_values:
        #     print("Hidden states before MLP:", hidden_states[0,:3,:3])

        residual = hidden_states

        hidden_states = self.layernorm(hidden_states)

        # if print_values:
        #     print("Hidden states after MLP layernorm:", hidden_states[0,:3,:3])

        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + self.drop_path(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            # print("Caching:", present_key_value.shape)
            outputs += (present_key_value,)

        return outputs


class Pix2SeqDecoder(Pix2SeqPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_decoder_layers* layers. Each layer is a [`Pix2SeqDecoderLayer`].

    The decoder uses shared input and output embeddings by default.

    Past_key_values: tensor of shape (number of decoder layers, batch size, sequence length, hidden size).

    Args:
        config: Pix2SeqConfig
        embed_tokens (nn.Embedding): optional output embedding.
    """

    def __init__(self, config: Pix2SeqConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.padding_idx = config.pad_token_id

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.dim_decoder, self.padding_idx)

        self.embed_positions = Pix2SeqSequencePositionEmbeddings(config)

        self.layers = nn.ModuleList([Pix2SeqDecoderLayer(config) for _ in range(config.num_decoder_layers)])

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values.shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # add position embeddings
        seq_len = inputs_embeds.shape[1]
        hidden_states = (
            inputs_embeds + self.embed_positions()[past_key_values_length : past_key_values_length + seq_len]
        )

        print("Decoder input ID:", input_ids)
        print("Past key values length:", past_key_values_length)
        print("Shape of inputs_embeds:", hidden_states.shape)
        print("First values of inputs_embeds:", hidden_states[0, :3, :3])

        # print("Hidden states after embedding them:", hidden_states.shape)
        # print("First values of inputs_embeds with pos embeddings:", hidden_states[0,0,:3])

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                    )

        if past_key_values is not None:
            print("Shape of past key values:", past_key_values.shape)
            print("Shape of hidden_states:", hidden_states.shape)

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # take cache of ith layer
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                # if idx == 0:
                #     print(f"Hidden states before layer {idx}", hidden_states[0,:3,:3])
                #     print(f"Encoder hidden states before layer {idx}", encoder_hidden_states[0,:3,:3])

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    print_values=idx == 0,
                )
            hidden_states = layer_outputs[0]

            # if idx == 0:
            #         print(f"Hidden states after layer {idx}", hidden_states[0,:3,:3])

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # print("Final hidden states of decoder:", hidden_states[0,:3,:3])

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # cache = tensor of shape (number of decoder layers, batch size, seq length, hidden size)
        next_decoder_cache = torch.stack(next_decoder_cache, dim=0)

        if past_key_values is not None:
            next_cache = torch.cat([past_key_values, next_decoder_cache], dim=2)
        else:
            next_cache = next_decoder_cache
        print("Cache out:", next_cache.shape)
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


PIX2SEQ_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Pix2SeqConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

PIX2SEQ_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`ViTFeatureExtractor`]. See
            [`ViTFeatureExtractor.__call__`] for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Pix2Seq Model transformer outputting raw hidden-states without any specific head on top.",
    PIX2SEQ_START_DOCSTRING,
)
class Pix2SeqModel(Pix2SeqPreTrainedModel):
    def __init__(self, config: Pix2SeqConfig):
        super().__init__(config)
        self.config = config

        self.encoder = Pix2SeqEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.projection = Pix2SeqProjection(config)

        self.decoder = Pix2SeqDecoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(PIX2SEQ_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqLMOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_encoder_layers x num_heads]
        # and head_mask is converted to shape [num_encoder_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_encoder_layers)

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                pixel_values,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]
        if encoder_hidden_states.shape[-1] != self.config.dim_decoder:
            # project encoder hidden states to match dimension of the decoder
            # TODO make it more efficient by only computing this once at generation time
            sequence_output = self.layernorm(encoder_outputs[0])
            sequence_output = self.projection(sequence_output)
            encoder_hidden_states = sequence_output

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class Pix2SeqForConditionalGeneration(Pix2SeqPreTrainedModel):
    def __init__(self, config: Pix2SeqConfig):
        super().__init__(config)
        self.config = config

        self.model = Pix2SeqModel(config)
        self.output_layernorm = nn.LayerNorm(config.dim_decoder)
        self.lm_head = nn.Linear(config.dim_decoder, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        print("First values of decoder outputs:", outputs[0][0, :3, :3])

        decoder_outputs = self.output_layernorm(outputs[0])

        lm_logits = self.lm_head(decoder_outputs)

        print("First values of final logits:", lm_logits[:, -1, :][0, :3])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
            print("Decoder input ids:", decoder_input_ids)

        return {
            "pixel_values": None,  # encoder_outputs is defined. pixel_values not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }
