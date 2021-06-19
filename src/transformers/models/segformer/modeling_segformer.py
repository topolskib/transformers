# coding=utf-8
# Copyright 2021 NVIDIA The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch SegFormer model. """

import math
import os

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import (
    BaseModelOutput,
)
from ...modeling_utils import (
    PreTrainedModel,
    SequenceSummary,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ...utils import logging
from .configuration_segformer import SegFormerConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "nvidia/segformer-b0"
_CONFIG_FOR_DOC = "SegFormerConfig"

SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "nvidia/segformer-b0",
    # See all SegFormer models at https://huggingface.co/models?filter=segformer
]


class SegFormerOverlapPatchEmbeddings(nn.Module):
    """Construct the patch embeddings from an image."""

    def __init__(self, image_size, patch_size, stride, num_channels, hidden_size):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.height, self.width = image_size[0] // patch_size[0], image_size[1] // patch_size[1]
        self.num_patches = self.height * self.width
        self.proj = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values):

        x = self.proj(pixel_values)
        _, _, height, width = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.layer_norm(x)
        return x


class SegFormerSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in SegFormerModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

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


class SegFormerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SegFormerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = SegFormerSelfAttention(config)
        self.output = SegFormerSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class SegFormerIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class SegFormerOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class SegFormerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = SegFormerAttention(config)
        self.intermediate = SegFormerIntermediate(config)
        self.output = SegFormerOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class SegFormerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        cur = 0
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths))]  # stochastic depth decay rule

        #self.layer = nn.ModuleList([SegFormerEncoderLayer(config) for _ in range(config.num_hidden_layers)])

        for i in range(config.num_encoder_blocks):
            # patch embeddings
            embedding_name = f"patch_embeddings_{i}"
            setattr(self, embedding_name, SegFormerOverlapPatchEmbeddings(image_size=config.image_size // config.downsampling_rates[i], 
                                                                          patch_size=config.patch_sizes[i], stride=config.strides[i], 
                                                                          num_channels=config.hidden_sizes[i-1] if i != 0 else config.num_channels, 
                                                                          hidden_size=config.hidden_sizes[i]))
            # Transformer block
            if i != 0:
                cur += config.depths[i-1] 
            block_name = f"block{i}"
            setattr(self, block_name, nn.ModuleList([SegFormerEncoderLayer(hidden_size=config.hidden_sizes[i], num_heads=config.encoder_attention_heads[i], 
                        mlp_ratio=mlp_ratios[0], drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], sr_ratio=sr_ratios[i])
                        for j in range(config.depths[i])]))
            norm_name = f"layer_norm{i}"
            setattr(self, norm_name, nn.LayerNorm(config.hidden_sizes[i]))

    def forward(
        self,
        pixel_values,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        batch_size = pixel_values.shape[0]
        
        features = []
        x = pixel_values
        for i in range(config.num_encoder_blocks):
            # first, obtain patch embeddings
            embedding_name = f"patch_embeddings_{i}"
            embedding_layer = getattr(self, embedding_name)
            x, height, width = embedding_layer(x)
            # second, send embeddings through blocks
            block_name = f"block{i}"
            block = getattr(self, block_name)
            for i, blk in enumerate(block):
                x = blk(x, height, width)
            # third, apply layer norm and reshape
            norm_name = f"layer_norm{i}"
            norm_layer = getattr(self, norm_name)
            x = norm_layer(x)
            x = x.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
            features.append(x)
        
        # for i, layer_module in enumerate(self.layer):
        #     if output_hidden_states:
        #         all_hidden_states = all_hidden_states + (hidden_states,)

        #     layer_head_mask = head_mask[i] if head_mask is not None else None

        #     if getattr(self.config, "gradient_checkpointing", False) and self.training:

        #         def create_custom_forward(module):
        #             def custom_forward(*inputs):
        #                 return module(*inputs, output_attentions)

        #             return custom_forward

        #         layer_outputs = torch.utils.checkpoint.checkpoint(
        #             create_custom_forward(layer_module),
        #             hidden_states,
        #             attention_mask,
        #             layer_head_mask,
        #         )
        #     else:
        #         layer_outputs = layer_module(
        #             hidden_states,
        #             attention_mask,
        #             layer_head_mask,
        #             output_attentions,
        #         )

        #     hidden_states = layer_outputs[0]
        #     if output_attentions:
        #         all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class SegFormerDecoder(SegFormerPreTrainedModel):
    """
    MLP decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`SegFormerDecoderLayer`

    Args:
        config: SegFormerConfig
    """

    def __init__(self, config: SegFormerConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop

        self.layers = nn.ModuleList([SegFormerDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        features,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            features (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                ...
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warning("`use_cache = True` is incompatible with `config.gradient_checkpointing = True`. Setting `use_cache = False`...")
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
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_layer_head_mask=(cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@add_start_docstrings(
    "The bare SegFormer encoder-decoder model outputting raw hidden-states without any specific head on top.",
    SEGFORMER_START_DOCSTRING,
)
class SegFormerModel(SegFormerPreTrainedModel):
    def __init__(self, config: SegFormerConfig):
        super().__init__(config)

        # hierarchical Transformer encoder
        self.encoder = SegFormerEncoder(config)
        # MLP decoder
        self.decoder = SegFormerDecoder(config)

        self.init_weights()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(SEGFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        pixel_values,
        head_mask=None,
        encoder_outputs=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        #
        
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                hidden_states,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )