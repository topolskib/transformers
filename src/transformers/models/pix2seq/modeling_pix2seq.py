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
import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import embedding, nn

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
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


def get_angles(pos, i, dim):
  angle_rates = 1 / torch.pow(10000., (2 * (i//2).float()) / dim)
  return pos.float() * angle_rates.float()


def positional_encoding(coords, dim):
    """
    Args:
        coords:
            Coordinates in (batch_size, dimension)
            
        Returns: 
            Position embeddings of shape (bsz, size, dim).
    """
    angle_rads = get_angles(coords.unsqueeze(-1),
                            torch.arange(start=0, end=dim)[None, None, :],
                            dim)

    # apply sin to even indices in the array; 2i
    angle_rads1 = torch.sin(angle_rads[:, :, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads2 = torch.cos(angle_rads[:, :, 1::2])

    pos_encoding = torch.cat([angle_rads1, angle_rads2], -1)

    return pos_encoding.float()


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
            ?ormalize coordinates between [0, normalization_max]. If None, raw coordinates from 0 to height/width will be used.
    
    Returns:
        positional code of shape (1, height, width, out_dim)
    """
    y_coords = torch.arange(start=0, end=height, dtype=torch.float)
    if normalization_max is not None:
        y_coords = y_coords / (height - 1) * normalization_max
    y_coords = positional_encoding(y_coords, out_dim//2)
    y_coords = y_coords.unsqueeze(2)
    y_coords = torch.cat([y_coords, torch.zeros_like(y_coords)], -1)

    x_coords = torch.arange(start=0, end=width, dtype=torch.float)
    if normalization_max is not None:
        x_coords = x_coords / (width - 1) * normalization_max
    x_coords = positional_encoding(x_coords, out_dim//2)
    x_coords = x_coords.unsqueeze(1)
    x_coords = torch.cat([torch.zeros_like(x_coords), x_coords], -1)
    
    return y_coords + x_coords


class Pix2SeqPositionEmbeddings(nn.Module):
    def __init__(self, config: Pix2SeqConfig, dim):
        super().__init__()
        
        n_rows = n_cols = config.image_size // config.patch_size
        
        if config.positional_encoding == 'learned':
            self.embeddings = nn.Parameter(shape=(n_rows * n_cols, dim))
        elif config.positional_encoding == 'sin_cos':
            sin_cos = get_2d_position_codes(
                n_rows, n_cols, dim, normalization_max=6.2831852)
            self.embeddings = torch.reshape(sin_cos, [n_rows * n_cols, dim])
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
        self.position_embeddings = Pix2SeqPositionEmbeddings(config, dim=config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        batch_size, seq_len, _ = embeddings.size()

        if self.config.use_cls_token:
            # add the [CLS] token to the embedded patch tokens
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings().unsqueeze(0)

        print("First values after position embeddings:", embeddings[0,:3,:3])
        print("Last values after position embeddings:", embeddings[0,-3:,:3:])
        print("Sum of embeddings:", embeddings.sum())

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
        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding="valid")
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        
        print("First values of pixel values:", pixel_values[0,:3,0,0])

        print("Shape of kernel of conv2d layer:", self.projection.weight.data.shape)
        print("First values of kernel of conv2d layer:", self.projection.weight.data[:3,0,2,0])
        print("First values of bias of conv2d layer:", self.projection.bias.data[:3])
        
        embeddings = self.projection(pixel_values)

        # shape (batch_size, hidden_size, num_patches, num_patches)
        
        print("Shape of patch embeddings:", embeddings.shape)
        print("First values of patch embeddings:", embeddings[0,:3,0,0])
        
        embeddings = embeddings.flatten(2).transpose(1, 2)

        embeddings = self.layer_norm(embeddings)

        print("Shape of embeddings after layernorm:", embeddings.shape)
        print("First values of embeddings after layer norm:", embeddings[0,:3,:3])

        return embeddings


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention with ViT->Pix2Seq
class Pix2SeqSelfAttention(nn.Module):
    def __init__(self, config: Pix2SeqConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->Pix2Seq
class Pix2SeqSelfOutput(nn.Module):
    """
    The residual connection is defined in Pix2SeqLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: Pix2SeqConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->Pix2Seq
class Pix2SeqAttention(nn.Module):
    def __init__(self, config: Pix2SeqConfig) -> None:
        super().__init__()
        self.attention = Pix2SeqSelfAttention(config)
        self.output = Pix2SeqSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate with ViT->Pix2Seq
class Pix2SeqIntermediate(nn.Module):
    def __init__(self, config: Pix2SeqConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTOutput with ViT->Pix2Seq
class Pix2SeqOutput(nn.Module):
    def __init__(self, config: Pix2SeqConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTLayer with ViT->Pix2Seq
class Pix2SeqLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: Pix2SeqConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = Pix2SeqAttention(config)
        self.intermediate = Pix2SeqIntermediate(config)
        self.output = Pix2SeqOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in Pix2Seq, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in Pix2Seq, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->Pix2Seq
class Pix2SeqEncoder(nn.Module):
    def __init__(self, config: Pix2SeqConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([Pix2SeqLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

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
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

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

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Pix2SeqProjectionMLP(nn.Module):
    """
    Pix2Seq projection MLP, only used if config.dec_proj_mode == "mlp".
    """
    def __init__(self, config: Pix2SeqConfig) -> None:
        super().__init__()
        self.layernorm = nn.LayerNorm(config.dim_att_dec)
        self.dense1 = nn.Linear(config.dim_att_dec, config.dim_att_dec)
        self.dropout = nn.Dropout()
        self.dense2 = nn.Linear(config.dim_att_dec, config.dim_att_dec)
        self.drop_path = Pix2SeqDropPath(config.drop_path_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input = hidden_states
        
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)

        hidden_states = input + self.drop_path(hidden_states)

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
        self.projection = nn.Linear(config.hidden_size, config.dim_att_dec)
        self.layernorm = nn.LayerNorm(config.dim_att_dec)
        
        if config.dec_proj_mode in ['linear_p', 'mlp']:
            self.position_embeddings = Pix2SeqPositionEmbeddings(config, dim=config.dim_att_dec)
        if config.dec_proj_mode == 'mlp':
            self.projection_mlp = Pix2SeqProjectionMLP(config)

        self.config = config

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.projection(hidden_states)
        hidden_states = self.layernorm(hidden_states)

        # Add (optional) positional embedding to encoded visual units.
        if self.config.dec_proj_mode != 'linear':
            position_embeddings = self.position_embeddings().unsqueeze(0)
            if self.config.use_cls_token:
                hidden_states = hidden_states + torch.cat(
                    [torch.zeros_like(position_embeddings[:, :1]), position_embeddings], 1)
            else:
                hidden_states = hidden_states + position_embeddings
        
            if self.config.dec_proj_mode == 'mlp':
                hidden_states = self.projection_mlp(hidden_states)
            else:
                assert self.config.dec_proj_mode == 'linear_p'

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

        self.embeddings = Pix2SeqEmbeddings(config)
        self.encoder = Pix2SeqEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # self.projection = Pix2SeqProjection(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> PatchEmbeddings:
        return self.embeddings.patch_embeddings

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
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        # sequence_output = self.projection(sequence_output)

        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=None,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
