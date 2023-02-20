# coding=utf-8
# Copyright 2023 Microsoft Research Asia and the HuggingFace Inc. team.
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
""" PyTorch LayoutReader model."""


import math

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_layoutreader import LayoutReaderConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LayoutReaderConfig"
_CHECKPOINT_FOR_DOC = "microsoft/layoutreader-base"

LAYOUTREADER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/layoutreader-base",
    # See all LayoutReader models at https://huggingface.co/models?filter=layoutreader
]


LayoutReaderLayerNorm = nn.LayerNorm


class LayoutReaderEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super(LayoutReaderEmbeddings, self).__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)

        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size)

        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, bbox, token_type_ids=None, position_ids=None, task_idx=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        position_embeddings = self.position_embeddings(position_ids)

        left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
        upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
        right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
        lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])

        embeddings = (
            position_embeddings
            + left_position_embeddings
            + upper_position_embeddings
            + right_position_embeddings
            + lower_position_embeddings
            + h_position_embeddings
            + w_position_embeddings
        )

        words_embeddings = self.word_embeddings(input_ids)
        embeddings = embeddings + words_embeddings

        if self.token_type_embeddings is not None:
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class LayoutReaderSelfAttention(nn.Module):
    def __init__(self, config):
        super(LayoutReaderSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # TODO remove this
        # if hasattr(config, "num_qkv") and (config.num_qkv > 1):
        #     self.num_qkv = config.num_qkv
        # else:
        #
        self.num_qkv = 1

        self.query = nn.Linear(config.hidden_size, self.all_head_size * self.num_qkv)
        self.key = nn.Linear(config.hidden_size, self.all_head_size * self.num_qkv)
        self.value = nn.Linear(config.hidden_size, self.all_head_size * self.num_qkv)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # TODO remove this
        # if hasattr(config, "seg_emb") and config.seg_emb:
        #     self.b_q_s = nn.Parameter(torch.zeros(1, self.num_attention_heads, 1, self.attention_head_size))
        #     self.seg_emb = nn.Embedding(config.type_vocab_size, self.all_head_size)
        # else:
        self.b_q_s = None
        self.seg_emb = None

    def transpose_for_scores(self, x, mask_qkv=None):
        if self.num_qkv > 1:
            sz = x.size()[:-1] + (self.num_qkv, self.num_attention_heads, self.all_head_size)
            # (batch, pos, num_qkv, head, head_hid)
            x = x.view(*sz)
            if mask_qkv is None:
                x = x[:, :, 0, :, :]
            elif isinstance(mask_qkv, int):
                x = x[:, :, mask_qkv, :, :]
            else:
                # mask_qkv: (batch, pos)
                if mask_qkv.size(1) > sz[1]:
                    mask_qkv = mask_qkv[:, : sz[1]]
                # -> x: (batch, pos, head, head_hid)
                x = x.gather(2, mask_qkv.view(sz[0], sz[1], 1, 1, 1).expand(sz[0], sz[1], 1, sz[3], sz[4])).squeeze(2)
        else:
            sz = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            # (batch, pos, head, head_hid)
            x = x.view(*sz)
        # (batch, head, pos, head_hid)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask,
        history_states=None,
        mask_qkv=None,
        seg_ids=None,
        key_history=None,
        value_history=None,
        key_cache=None,
        value_cache=None,
    ):
        if history_states is None:
            mixed_query_layer = self.query(hidden_states)
            # possible issue: https://github.com/NVIDIA/apex/issues/131
            mixed_key_layer = nn.functional.linear(hidden_states, self.key.weight)
            mixed_value_layer = self.value(hidden_states)
        else:
            x_states = torch.cat((history_states, hidden_states), dim=1)
            mixed_query_layer = self.query(hidden_states)
            # possible issue: https://github.com/NVIDIA/apex/issues/131
            mixed_key_layer = nn.functional.linear(x_states, self.key.weight)
            mixed_value_layer = self.value(x_states)

        if key_cache is not None and isinstance(key_cache, list):
            key_cache.append(mixed_key_layer)
            mixed_key_layer = torch.cat(key_cache, dim=1)

        if value_cache is not None and isinstance(value_cache, list):
            value_cache.append(mixed_value_layer)
            mixed_value_layer = torch.cat(value_cache, dim=1)

        query_layer = self.transpose_for_scores(mixed_query_layer, mask_qkv)
        key_layer = self.transpose_for_scores(mixed_key_layer, mask_qkv)
        value_layer = self.transpose_for_scores(mixed_value_layer, mask_qkv)

        if key_history is not None and not isinstance(key_history, list):
            key_layer = torch.cat((key_history, key_layer), dim=-2)
            value_layer = torch.cat((value_history, value_layer), dim=-2)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch, head, pos, pos)
        attention_scores = torch.matmul(query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))

        # TODO remove this
        # if self.seg_emb is not None:
        #     seg_rep = self.seg_emb(seg_ids)
        #     # (batch, pos, head, head_hid)
        #     seg_rep = seg_rep.view(
        #         seg_rep.size(0), seg_rep.size(1), self.num_attention_heads, self.attention_head_size
        #     )
        #     qs = torch.einsum("bnih,bjnh->bnij", query_layer + self.b_q_s, seg_rep)
        #     attention_scores = attention_scores + qs

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if isinstance(key_history, list):
            key_history.append(key_layer)
        if isinstance(value_history, list):
            value_history.append(value_layer)

        return context_layer


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->LayoutReader
class LayoutReaderSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LayoutReaderAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = LayoutReaderSelfAttention(config)
        self.output = LayoutReaderSelfOutput(config)
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
        input_tensor,
        attention_mask,
        history_states=None,
        mask_qkv=None,
        seg_ids=None,
        key_history=None,
        value_history=None,
    ):
        self_output = self.self(
            input_tensor,
            attention_mask,
            history_states=history_states,
            mask_qkv=mask_qkv,
            seg_ids=seg_ids,
            key_history=key_history,
            value_history=value_history,
        )
        attention_output = self.output(self_output, input_tensor)
        return attention_output


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class LayoutReaderIntermediate(nn.Module):
    def __init__(self, config):
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


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->LayoutReader
class LayoutReaderOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LayoutReaderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = LayoutReaderAttention(config)
        self.intermediate = LayoutReaderIntermediate(config)
        self.output = LayoutReaderOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask,
        history_states=None,
        mask_qkv=None,
        seg_ids=None,
        key_history=None,
        value_history=None,
    ):
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            history_states=history_states,
            mask_qkv=mask_qkv,
            seg_ids=seg_ids,
            key_history=key_history,
            value_history=value_history,
        )

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class LayoutReaderEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([LayoutReaderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_all_encoded_layers=True,
        prev_embedding=None,
        prev_encoded_layers=None,
        mask_qkv=None,
        seg_ids=None,
        key_history=None,
        value_history=None,
    ):
        # history embedding and encoded layer must be simultanously given
        assert (prev_embedding is None) == (prev_encoded_layers is None)

        all_encoder_layers = []
        if (prev_embedding is not None) and (prev_encoded_layers is not None):
            history_states = prev_embedding
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(
                    hidden_states, attention_mask, history_states=history_states, mask_qkv=mask_qkv, seg_ids=seg_ids
                )
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
                if prev_encoded_layers is not None:
                    history_states = prev_encoded_layers[i]
        else:
            for i, layer_module in enumerate(self.layer):
                set_key = None
                if isinstance(key_history, list):
                    set_key = key_history if len(key_history) < len(self.layer) else key_history[i]
                set_value = None
                if isinstance(value_history, list):
                    set_value = value_history if len(key_history) < len(self.layer) else value_history[i]
                hidden_states = layer_module(
                    hidden_states,
                    attention_mask,
                    mask_qkv=mask_qkv,
                    seg_ids=seg_ids,
                    key_history=set_key,
                    value_history=set_value,
                )
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


# Copied from transformers.models.layoutlm.modeling_layoutlm.LayoutLMPreTrainedModel with LAYOUTLM->LAYOUTREADER,LayoutLM->LayoutReader,layoutlm->layoutreader
class LayoutReaderPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = LayoutReaderConfig
    pretrained_model_archive_map = LAYOUTREADER_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "layoutreader"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayoutReaderLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LayoutReaderEncoder):
            module.gradient_checkpointing = value


LAYOUTREADER_START_DOCSTRING = r"""
    The LayoutReader model was proposed in [LayoutReader: Pre-training of Text and Layout for Document Image
    Understanding](https://arxiv.org/abs/1912.13318) by Yiheng Xu, Minghao Li, Lei Cui, Shaohan Huang, Furu Wei and
    Ming Zhou.

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`LayoutReaderConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

LAYOUTREADER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        bbox (`torch.LongTensor` of shape `({0}, 4)`, *optional*):
            Bounding boxes of each input sequence tokens. Selected in the range `[0,
            config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
            format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
            y1) represents the position of the lower right corner. See [Overview](#Overview) for normalization.
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`: `1` for
            tokens that are NOT MASKED, `0` for MASKED tokens.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`: `0` corresponds to a *sentence A* token, `1` corresponds to a *sentence B* token

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`: `1`
            indicates the head is **not masked**, `0` indicates the head is **masked**.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            If set to `True`, the attentions tensors of all attention layers are returned. See `attentions` under
            returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            If set to `True`, the hidden states of all layers are returned. See `hidden_states` under returned tensors
            for more detail.
        return_dict (`bool`, *optional*):
            If set to `True`, the model will return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LayoutReader Model transformer outputting raw hidden-states without any specific head on top.",
    LAYOUTREADER_START_DOCSTRING,
)
class LayoutReaderModel(LayoutReaderPreTrainedModel):
    def __init__(self, config):
        super(LayoutReaderModel, self).__init__(config)
        self.config = config

        self.embeddings = LayoutReaderEmbeddings(config)
        self.encoder = LayoutReaderEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # def rescale_some_parameters(self):
    #     for layer_id, layer in enumerate(self.encoder.layer):
    #         layer.attention.output.dense.weight.data.div_(
    #             math.sqrt(2.0 * (layer_id + 1)))
    #         layer.output.dense.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def get_extended_attention_mask(self, input_ids, token_type_ids, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    @add_start_docstrings_to_model_forward(LAYOUTREADER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids,
        token_type_ids,
        position_ids,
        attention_mask,
        output_all_encoded_layers=True,
        prev_embedding=None,
        prev_encoded_layers=None,
        mask_qkv=None,
        task_idx=None,
    ):
        """
        Returns:
        """
        extended_attention_mask = self.get_extended_attention_mask(input_ids[:, :, 0], token_type_ids, attention_mask)

        embedding_output = self.embeddings(
            input_ids[:, :, 0], input_ids[:, :, 1:], token_type_ids, position_ids, task_idx=task_idx
        )
        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            prev_embedding=prev_embedding,
            prev_encoded_layers=prev_encoded_layers,
            mask_qkv=mask_qkv,
            seg_ids=token_type_ids,
        )
        encoded_layers[-1]
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        return embedding_output, encoded_layers


# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->LayoutReader
class LayoutReaderPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class LayoutReaderLMPredictionHead(nn.Module):
    def __init__(self, config, src_len):
        super(LayoutReaderLMPredictionHead, self).__init__()
        self.transform = LayoutReaderPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.

        self.bias = nn.Parameter(torch.zeros(src_len))

        if hasattr(config, "relax_projection") and (config.relax_projection > 1):
            self.relax_projection = config.relax_projection
        else:
            self.relax_projection = 0
        self.fp32_embedding = False  # TODO add to config?

        def convert_to_type(tensor):
            if self.fp32_embedding:
                return tensor.half()
            else:
                return tensor

        self.type_converter = convert_to_type
        self.converted = False

    def forward(self, hidden_states, src_emb, task_idx=None):
        if not self.converted:
            self.converted = True
            if self.fp32_embedding:
                self.transform.half()
        hidden_states = self.transform(self.type_converter(hidden_states))
        if self.relax_projection > 1:
            num_batch = hidden_states.size(0)
            num_pos = hidden_states.size(1)
            # (batch, num_pos, relax_projection*hid) -> (batch, num_pos, relax_projection, hid) -> (batch, num_pos, hid)
            hidden_states = hidden_states.view(num_batch, num_pos, self.relax_projection, -1)[
                torch.arange(0, num_batch).long(), :, task_idx, :
            ]
        if self.fp32_embedding:
            hidden_states = torch.einsum(
                "btf,bsf->bts", self.type_converter(hidden_states), self.type_converter(src_emb)
            ) + self.type_converter(self.bias)
        else:
            hidden_states = torch.einsum("btf,bsf->bts", hidden_states, src_emb) + self.bias
        return hidden_states


@add_start_docstrings(
    """LayoutReader model for reading order prediction.

    ForSeq2SeqDecoding is a bad name, should actually become ForSeq2SeqDecoder.

See: https://github.com/microsoft/unilm/blob/master/layoutreader/s2s_ft/modeling_decoding.py#L1129.

""",
    LAYOUTREADER_START_DOCSTRING,
)
class LayoutReaderForSeq2SeqDecoding(LayoutReaderPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.layoutreader = LayoutReaderModel(config)
        self.cls = LayoutReaderLMPredictionHead(config, src_len=config.max_source_length)

        self.layout_flag = True  # TODO remove this attribute
        self.mask_word_id = config.mask_word_id
        self.beam_size = config.beam_size
        self.length_penalty = config.length_penalty
        self.eos_id = config.eos_id
        self.sos_id = config.sos_id
        self.forbid_duplicate_ngrams = config.forbid_duplicate_ngrams
        self.forbid_ignore_set = config.forbid_ignore_set
        self.ngram_size = config.ngram_size
        self.pos_shift = config.pos_shift

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.layoutreader.embeddings.word_embeddings

    def generate(self, input_ids, token_type_ids, position_ids, attention_mask, task_idx=None, mask_qkv=None):
        if self.config.beam_size > 1:
            return self.beam_search(
                input_ids, token_type_ids, position_ids, attention_mask, task_idx=task_idx, mask_qkv=mask_qkv
            )

        input_shape = list(input_ids.size())
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size())
        output_length = output_shape[1]

        output_ids = []
        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = input_ids

        if not self.layout_flag:
            mask_ids = input_ids.new(batch_size, 1).fill_(self.mask_word_id)
        else:
            mask_ids = input_ids.new_zeros(batch_size, 1, 5)
            mask_ids[:, :, 0] = self.mask_word_id

        next_pos = input_length
        if self.pos_shift:
            if not self.layout_flag:
                sos_ids = input_ids.new(batch_size, 1).fill_(self.sos_id)
            else:
                sos_ids = input_ids.new_zeros(batch_size, 1, 5)
                sos_ids[:, :, 0] = self.sos_id

        src_embedding = None

        while next_pos < output_length:
            curr_length = list(curr_ids.size())[1]

            if self.pos_shift:
                if next_pos == input_length:
                    x_input_ids = torch.cat((curr_ids, sos_ids), dim=1)
                    start_pos = 0
                else:
                    x_input_ids = curr_ids
                    start_pos = next_pos
            else:
                start_pos = next_pos - curr_length
                if next_pos < 520:
                    print("Current ids:", curr_ids[0].tolist())
                x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)

            curr_token_type_ids = token_type_ids[:, start_pos : next_pos + 1]
            curr_attention_mask = attention_mask[:, start_pos : next_pos + 1, : next_pos + 1]
            curr_position_ids = position_ids[:, start_pos : next_pos + 1]

            if next_pos < 520:
                print("Input ids:", x_input_ids[0].tolist())

            new_embedding, new_encoded_layers = self.layoutreader(
                x_input_ids,
                curr_token_type_ids,
                curr_position_ids,
                curr_attention_mask,
                output_all_encoded_layers=True,
                prev_embedding=prev_embedding,
                prev_encoded_layers=prev_encoded_layers,
                mask_qkv=mask_qkv,
            )

            if src_embedding is None:
                # note: cut three embedding: CLS (1st), ..., SEP (-2nd), next to pred (-1st)
                # note: (NEW) the sep is kept for ignore index in loss func (for padding's index)
                # NOTE: only remove the next to pred token
                print("Shape of new_embedding:", new_embedding.shape)
                src_embedding = new_embedding[:, :-1, :]
                print("Src embedding:", src_embedding.shape)

            last_hidden = new_encoded_layers[-1][:, -1:, :]
            if next_pos == 514:
                print("Shape of new_encoded_layers:", new_encoded_layers[-1].shape)
                print("Shape of last_hidden:", last_hidden.shape)
            prediction_scores = self.cls(last_hidden, src_embedding, task_idx=task_idx)

            if next_pos < 520:
                print("Time step:", next_pos)
                print("Shape of prediction_scores:", prediction_scores.shape)
                print("--------------------")

            # print("First values of prediction_scores:", prediction_scores[0,:3,:3])
            # print_flag = False
            _, max_ids = torch.max(prediction_scores, dim=-1)
            output_ids.append(max_ids)

            if self.pos_shift:
                if prev_embedding is None:
                    prev_embedding = new_embedding
                else:
                    prev_embedding = torch.cat((prev_embedding, new_embedding), dim=1)
                if prev_encoded_layers is None:
                    prev_encoded_layers = [x for x in new_encoded_layers]
                else:
                    prev_encoded_layers = [
                        torch.cat((x[0], x[1]), dim=1) for x in zip(prev_encoded_layers, new_encoded_layers)
                    ]
            else:
                if prev_embedding is None:
                    prev_embedding = new_embedding[:, :-1, :]
                else:
                    prev_embedding = torch.cat((prev_embedding, new_embedding[:, :-1, :]), dim=1)
                if prev_encoded_layers is None:
                    prev_encoded_layers = [x[:, :-1, :] for x in new_encoded_layers]
                else:
                    prev_encoded_layers = [
                        torch.cat((x[0], x[1][:, :-1, :]), dim=1) for x in zip(prev_encoded_layers, new_encoded_layers)
                    ]

            if not self.layout_flag:
                index = max_ids
                curr_ids = torch.gather(input_ids, 1, index)
            else:
                _, _, dim = input_ids.shape
                index = max_ids.unsqueeze(-1)
                index = index.expand(index.shape[0], index.shape[1], dim)
                # index = index.repeat(1, 1, dim)
                curr_ids = torch.gather(input_ids, 1, index)

            next_pos += 1

        return torch.cat(output_ids, dim=1)

    # TODO: do the same with beam search as forward()
    def beam_search(self, input_ids, token_type_ids, position_ids, attention_mask, task_idx=None, mask_qkv=None):
        input_shape = list(input_ids.size())
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size())
        output_length = output_shape[1]

        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = input_ids
        # mask_ids = input_ids.new(batch_size, 1).fill_(self.mask_word_id)
        if not self.layout_flag:
            mask_ids = input_ids.new(batch_size, 1).fill_(self.mask_word_id)
        else:
            mask_ids = input_ids.new_zeros(batch_size, 1, 5)
            mask_ids[:, :, 0] = self.mask_word_id

        next_pos = input_length
        if self.pos_shift:
            if not self.layout_flag:
                sos_ids = input_ids.new(batch_size, 1).fill_(self.sos_id)
            else:
                sos_ids = input_ids.new_zeros(batch_size, 1, 5)
                sos_ids[:, :, 0] = self.sos_id

        K = self.config.beam_size

        total_scores = []
        beam_masks = []
        step_ids = []
        step_back_ptrs = []
        partial_seqs = []
        forbid_word_mask = None
        buf_matrix = None

        src_embedding = None

        while next_pos < output_length:
            curr_length = list(curr_ids.size())[1]

            if self.pos_shift:
                if next_pos == input_length:
                    x_input_ids = torch.cat((curr_ids, sos_ids), dim=1)
                    start_pos = 0
                else:
                    x_input_ids = curr_ids
                    start_pos = next_pos
            else:
                start_pos = next_pos - curr_length
                x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)

            curr_token_type_ids = token_type_ids[:, start_pos : next_pos + 1]
            curr_attention_mask = attention_mask[:, start_pos : next_pos + 1, : next_pos + 1]
            curr_position_ids = position_ids[:, start_pos : next_pos + 1]
            new_embedding, new_encoded_layers = self.layoutreader(
                x_input_ids,
                curr_token_type_ids,
                curr_position_ids,
                curr_attention_mask,
                output_all_encoded_layers=True,
                prev_embedding=prev_embedding,
                prev_encoded_layers=prev_encoded_layers,
                mask_qkv=mask_qkv,
            )

            def first_expand(x):
                input_shape = list(x.size())
                expanded_shape = input_shape[:1] + [1] + input_shape[1:]
                x = torch.reshape(x, expanded_shape)
                repeat_count = [1, K] + [1] * (len(input_shape) - 1)
                x = x.repeat(*repeat_count)
                x = torch.reshape(x, [input_shape[0] * K] + input_shape[1:])
                return x

            if src_embedding is None:
                src_embedding = new_embedding[:, :-1, :]

            if src_embedding.shape[0] != new_embedding.shape[0]:
                src_embedding = first_expand(src_embedding)

            last_hidden = new_encoded_layers[-1][:, -1:, :]
            prediction_scores = self.cls(last_hidden, src_embedding, task_idx=task_idx)
            log_scores = torch.nn.functional.log_softmax(prediction_scores, dim=-1)
            # if forbid_word_mask is not None:
            #     log_scores += (forbid_word_mask * -10000.0)
            # if self.min_len and (next_pos - input_length + 1 <= self.min_len):
            #     log_scores[:, :, self.eos_id].fill_(-10000.0)
            kk_scores, kk_ids = torch.topk(log_scores, k=K)
            if len(total_scores) == 0:
                k_ids = torch.reshape(kk_ids, [batch_size, K])
                back_ptrs = torch.zeros(batch_size, K, dtype=torch.long)
                k_scores = torch.reshape(kk_scores, [batch_size, K])
            else:
                last_eos = torch.reshape(beam_masks[-1], [batch_size * K, 1, 1])
                last_seq_scores = torch.reshape(total_scores[-1], [batch_size * K, 1, 1])
                kk_scores += last_eos * (-10000.0) + last_seq_scores
                kk_scores = torch.reshape(kk_scores, [batch_size, K * K])
                k_scores, k_ids = torch.topk(kk_scores, k=K)
                back_ptrs = torch.floor_divide(k_ids, K)
                kk_ids = torch.reshape(kk_ids, [batch_size, K * K])
                k_ids = torch.gather(kk_ids, 1, k_ids)
            step_back_ptrs.append(back_ptrs)
            step_ids.append(k_ids)
            beam_masks.append(torch.eq(k_ids, self.eos_id).type_as(kk_scores))
            total_scores.append(k_scores)

            def select_beam_items(x, ids):
                id_shape = list(ids.size())
                id_rank = len(id_shape)
                assert len(id_shape) == 2
                x_shape = list(x.size())
                x = torch.reshape(x, [batch_size, K] + x_shape[1:])
                x_rank = len(x_shape) + 1
                assert x_rank >= 2
                if id_rank < x_rank:
                    ids = torch.reshape(ids, id_shape + [1] * (x_rank - id_rank))
                    ids = ids.expand(id_shape + x_shape[1:])
                y = torch.gather(x, 1, ids)
                y = torch.reshape(y, x_shape)
                return y

            is_first = prev_embedding is None

            if self.pos_shift:
                if prev_embedding is None:
                    prev_embedding = first_expand(new_embedding)
                else:
                    prev_embedding = torch.cat((prev_embedding, new_embedding), dim=1)
                    prev_embedding = select_beam_items(prev_embedding, back_ptrs)
                if prev_encoded_layers is None:
                    prev_encoded_layers = [first_expand(x) for x in new_encoded_layers]
                else:
                    prev_encoded_layers = [
                        torch.cat((x[0], x[1]), dim=1) for x in zip(prev_encoded_layers, new_encoded_layers)
                    ]
                    prev_encoded_layers = [select_beam_items(x, back_ptrs) for x in prev_encoded_layers]
            else:
                if prev_embedding is None:
                    prev_embedding = first_expand(new_embedding[:, :-1, :])
                else:
                    prev_embedding = torch.cat((prev_embedding, new_embedding[:, :-1, :]), dim=1)
                    prev_embedding = select_beam_items(prev_embedding, back_ptrs)
                if prev_encoded_layers is None:
                    prev_encoded_layers = [first_expand(x[:, :-1, :]) for x in new_encoded_layers]
                else:
                    prev_encoded_layers = [
                        torch.cat((x[0], x[1][:, :-1, :]), dim=1) for x in zip(prev_encoded_layers, new_encoded_layers)
                    ]
                    prev_encoded_layers = [select_beam_items(x, back_ptrs) for x in prev_encoded_layers]

            max_ids = torch.reshape(k_ids, [batch_size * K, 1])

            if len(input_ids.shape) == 2:
                expand_input_ids = first_expand(input_ids)
                index = max_ids
                curr_ids = torch.gather(expand_input_ids, 1, index)
            else:
                expand_input_ids = first_expand(input_ids)

                _, _, dim = expand_input_ids.shape
                index = max_ids.unsqueeze(-1)
                index = index.expand(index.shape[0], index.shape[1], dim)

                curr_ids = torch.gather(expand_input_ids, 1, index)

            if is_first:
                token_type_ids = first_expand(token_type_ids)
                position_ids = first_expand(position_ids)
                attention_mask = first_expand(attention_mask)
                mask_ids = first_expand(mask_ids)
                if mask_qkv is not None:
                    mask_qkv = first_expand(mask_qkv)

            if self.forbid_duplicate_ngrams:
                wids = step_ids[-1].tolist()
                ptrs = step_back_ptrs[-1].tolist()
                if is_first:
                    partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            partial_seqs.append([wids[b][k]])
                else:
                    new_partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            new_partial_seqs.append(partial_seqs[ptrs[b][k] + b * K] + [wids[b][k]])
                    partial_seqs = new_partial_seqs

                def get_dup_ngram_candidates(seq, n):
                    cands = set()
                    if len(seq) < n:
                        return []
                    tail = seq[-(n - 1) :]
                    if self.forbid_ignore_set and any(tk in self.forbid_ignore_set for tk in tail):
                        return []
                    for i in range(len(seq) - (n - 1)):
                        mismatch = False
                        for j in range(n - 1):
                            if tail[j] != seq[i + j]:
                                mismatch = True
                                break
                        if (not mismatch) and not (
                            self.forbid_ignore_set and (seq[i + n - 1] in self.forbid_ignore_set)
                        ):
                            cands.add(seq[i + n - 1])
                    return list(sorted(cands))

                if len(partial_seqs[0]) >= self.ngram_size:
                    dup_cands = []
                    for seq in partial_seqs:
                        dup_cands.append(get_dup_ngram_candidates(seq, self.ngram_size))
                    if max(len(x) for x in dup_cands) > 0:
                        if buf_matrix is None:
                            vocab_size = list(log_scores.size())[-1]
                            buf_matrix = np.zeros((batch_size * K, vocab_size), dtype=float)
                        else:
                            buf_matrix.fill(0)
                        for bk, cands in enumerate(dup_cands):
                            for i, wid in enumerate(cands):
                                buf_matrix[bk, wid] = 1.0
                        forbid_word_mask = torch.tensor(buf_matrix, dtype=log_scores.dtype)
                        forbid_word_mask = torch.reshape(forbid_word_mask, [batch_size * K, 1, vocab_size]).to(
                            input_ids.device
                        )
                    else:
                        forbid_word_mask = None

            next_pos += 1

        # [(batch, beam)]
        total_scores = [x.tolist() for x in total_scores]
        step_ids = [x.tolist() for x in step_ids]
        step_back_ptrs = [x.tolist() for x in step_back_ptrs]
        # back tracking
        traces = {"pred_seq": [], "scores": [], "wids": [], "ptrs": []}
        for b in range(batch_size):
            # [(beam,)]
            scores = [x[b] for x in total_scores]
            wids_list = [x[b] for x in step_ids]
            ptrs = [x[b] for x in step_back_ptrs]
            traces["scores"].append(scores)
            traces["wids"].append(wids_list)
            traces["ptrs"].append(ptrs)
            # first we need to find the eos frame where all symbols are eos
            # any frames after the eos frame are invalid
            last_frame_id = len(scores) - 1
            for i, wids in enumerate(wids_list):
                if all(wid == self.eos_id for wid in wids):
                    last_frame_id = i
                    break
            max_score = -math.inf
            frame_id = -1
            pos_in_frame = -1

            for fid in range(last_frame_id + 1):
                for i, wid in enumerate(wids_list[fid]):
                    if wid == self.eos_id or fid == last_frame_id:
                        s = scores[fid][i]
                        if self.length_penalty > 0:
                            s /= math.pow((5 + fid + 1) / 6.0, self.length_penalty)
                        if s > max_score:
                            max_score = s
                            frame_id = fid
                            pos_in_frame = i
            if frame_id == -1:
                traces["pred_seq"].append([0])
            else:
                seq = [wids_list[frame_id][pos_in_frame]]
                for fid in range(frame_id, 0, -1):
                    pos_in_frame = ptrs[fid][pos_in_frame]
                    seq.append(wids_list[fid - 1][pos_in_frame])
                seq.reverse()
                traces["pred_seq"].append(seq)

        def _pad_sequence(sequences, max_len, padding_value=0):
            trailing_dims = sequences[0].size()[1:]
            out_dims = (len(sequences), max_len) + trailing_dims

            out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                out_tensor[i, :length, ...] = tensor
            return out_tensor

        # convert to tensors for DataParallel
        for k in ("pred_seq", "scores", "wids", "ptrs"):
            ts_list = traces[k]
            if not isinstance(ts_list[0], torch.Tensor):
                dt = torch.float if k == "scores" else torch.long
                ts_list = [torch.tensor(it, dtype=dt) for it in ts_list]
            traces[k] = _pad_sequence(ts_list, output_length, padding_value=0).to(input_ids.device)

        return traces
