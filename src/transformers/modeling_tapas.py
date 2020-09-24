# coding=utf-8
# Copyright (...) and The HuggingFace Inc. team.
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
"""PyTorch TAPAS model. """


import logging
import warnings
import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

#from .modeling_tapas_utilities import *
#import .modeling_tapas_utilities as utils
from transformers import modeling_tapas_utilities as utils

from .configuration_tapas import TapasConfig
from .modeling_bert import BertLayerNorm, BertPreTrainedModel, BertEncoder, BertPooler, BertOnlyMLMHead
from .modeling_outputs import (
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    QuestionAnsweringModelOutput, # to be used
    SequenceClassifierOutput, # to be used
)

logger = logging.getLogger(__name__)

TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "tapas-base",
    "tapas-large",
    # See all TAPAS models at https://huggingface.co/models?filter=tapas
]

def load_tf_weights_in_tapas(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a PyTorch model. 3 changes compared to "load_tf_weights_in_bert":
        - change start of all variable names to "tapas" rather than "bert" (except for "cls" layer)
        - skip seq_relationship variables (as the model is expected to be TapasModel)
        - take into account additional token type embedding layers
    """
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculate m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step", "seq_relationship"]
            #"column_output_bias", "column_output_weights", "output_bias", "output_weights"]
            for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        # if first scope name starts with "bert", change it to "tapas"
        if name[0] == "bert":
            name[0] = "tapas"
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            # cell selection heads
            elif scope_names[0] == "output_bias":
                pointer = getattr(pointer, "output_bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "output_weights")
            elif scope_names[0] == "column_output_bias":
                pointer = getattr(pointer, "column_output_bias")
            elif scope_names[0] == "column_output_weights":
                pointer = getattr(pointer, "column_output_weights")
            # aggregation head
            elif scope_names[0] == "output_bias_agg":
                pointer = getattr(pointer, "output_bias_agg")
            elif scope_names[0] == "output_weights_agg":
                pointer = getattr(pointer, "output_weights_agg")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name[-13:] in ["_embeddings_0", "_embeddings_1", "_embeddings_2", "_embeddings_3", "_embeddings_4", "_embeddings_5", "_embeddings_6"]:
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        # added a check whether the array is a scalar (because bias terms are scalar => should first be converted to numpy arrays)
        if np.isscalar(array):
            array = np.array(array)
        pointer.data = torch.from_numpy(array)
    return model



class TapasEmbeddings(nn.Module):
    """
    Same as BertEmbeddings but with a number of additional token type embeddings to encode tabular structure.
    """

    def __init__(self, config):
        super().__init__()
        # we do not include config.disabled_features and config.disable_position_embeddings
        # word embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # position embeddings
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # token type embeddings
        token_type_embedding_name = "token_type_embeddings"
        
        for i, type_vocab_size in enumerate(config.type_vocab_size):
            name="%s_%d" % (token_type_embedding_name, i)
            setattr(self, name, nn.Embedding(type_vocab_size, config.hidden_size)) 

        self.number_of_token_type_embeddings = len(config.type_vocab_size) 

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros((*input_shape, self.number_of_token_type_embeddings), dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # currently, only absolute position embeddings are implemented
        # to do: should be updated to account for when config.reset_position_index_per_cell is set to True
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings
        
        token_type_embedding_name = "token_type_embeddings"
        
        for i in range(self.number_of_token_type_embeddings):
            name="%s_%d" % (token_type_embedding_name, i)
            embeddings += getattr(self, name)(token_type_ids[:,:,i])

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TapasModel(BertPreTrainedModel):
    """
    This class is a small adaption from :class:`~transformers.BertModel`. Please check this
    class for the appropriate documentation alongside usage examples.
    """

    config_class = TapasConfig
    base_model_prefix = "tapas"

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = TapasEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros((*input_shape, len(self.config.type_vocab_size)), dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class TapasForMaskedLM(BertPreTrainedModel):
    config_class = TapasConfig
    base_model_prefix = "tapas"

    def __init__(self, config):
        super().__init__(config)

        assert (
            not config.is_decoder
        ), "If you want to use `TapasForMaskedLM` make sure `config.is_decoder=False` for bi-directional self-attention."

        self.tapas = TapasModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.tapas(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class TapasForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):   
        super().__init__(config)

        # base model
        self.tapas = TapasModel(config)
        
        # cell selection heads
        """init_cell_selection_weights_to_zero: Whether the initial weights should be
        set to 0. This ensures that all tokens have the same prior probability."""
        if config.init_cell_selection_weights_to_zero: 
            self.output_weights = nn.Parameter(torch.zeros(config.hidden_size)) 
            self.column_output_weights = nn.Parameter(torch.zeros(config.hidden_size))
        else:
            self.output_weights = nn.Parameter(torch.empty(config.hidden_size))
            nn.init.normal_(self.output_weights, std=0.02) # here, a truncated normal is used in the original implementation
            self.column_output_weights = nn.Parameter(torch.empty(config.hidden_size))
            nn.init.normal_(self.column_output_weights, std=0.02) # here, a truncated normal is used in the original implementation
        self.output_bias = nn.Parameter(torch.zeros([]))
        self.column_output_bias = nn.Parameter(torch.zeros([]))

        # aggregation head
        if config.num_aggregation_labels > 0:
            self.output_weights_agg = nn.Parameter(torch.empty([config.num_aggregation_labels, config.hidden_size]))
            nn.init.normal_(self.output_weights_agg, std=0.02) # here, a truncated normal is used in the original implementation
            self.output_bias_agg = nn.Parameter(torch.zeros([config.num_aggregation_labels]))

        # classification head
        if config.num_classification_labels > 0:
            self.output_weights_cls = nn.Parameter(torch.empty([config.num_classification_labels, config.hidden_size]))
            nn.init.normal_(self.output_weights_cls, std=0.02) # here, a truncated normal is used in the original implementation
            self.output_bias_cls = nn.Parameter(torch.zeros([config.num_classification_labels]))

        self.init_weights()

    def compute_token_logits(self, sequence_output, temperature):
        """Computes logits per token.
        Args:
            sequence_output: <float>[batch_size, seq_length, hidden_dim] Output of the
            encoder layer.
            temperature: float Temperature for the Bernoulli distribution.
        Returns:
            logits: <float>[batch_size, seq_length] Logits per token.
        """
        logits = (torch.einsum("bsj,j->bs", sequence_output, self.output_weights) +
                self.output_bias) / temperature

        return logits

    def compute_classification_logits(self, pooled_output):
        """Computes logits for each classification of the sequence.
        Args:
            pooled_output: <float>[batch_size, hidden_dim] Output of the pooler (BertPooler) on top of the encoder layer.
        Returns:
            logits_cls: <float>[batch_size, config.num_classification_labels] Logits per class.
        """
        logits_cls = torch.matmul(pooled_output, self.output_weights_cls.T)
        logits_cls += self.output_bias_cls
        
        return logits_cls

    def _calculate_aggregation_logits(self, pooled_output):
        """Calculates the aggregation logits.
        Args:
            pooled_output: torch.FloatTensor[batch_size, hidden_dim] Output of the pooler (BertPooler) on top of the encoder layer.
        Returns:
            logits_aggregation: torch.FloatTensor[batch_size, config.num_aggregation_labels] Logits per aggregation operation.
        """
        logits_aggregation = torch.matmul(pooled_output, self.output_weights_agg.T)
        logits_aggregation += self.output_bias_agg
        
        return logits_aggregation

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        table_mask=None, 
        label_ids=None,
        aggregation_function_id=None,
        answer=None,
        numeric_values=None,
        numeric_values_scale=None,
        classification_class_index=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        table_mask (:obj: `torch.LongTensor` of shape :obj:`(batch_size, seq_length)`, `optional`): 
            Mask for the table.   
        label_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, seq_length)`, `optional`):
            Labels per token.
        aggregation_function_id (:obj:`torch.LongTensor` of shape :obj:`(batch_size, )`, `optional`):
            Aggregation function id for every example in the batch. 
        answer (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, )`, `optional`):
            Answer for every example in the batch. 
        numeric_values (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_length)`, `optional`):
            Numeric values of every token. 
        numeric_values_scale (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, seq_length)`, `optional`):
            Scale of the numeric values of every token. 
        classification_class_index (:obj:`torch.LongTensor` of shape :obj:`(batch_size, )`, `optional`):
            Classification class index for every example in the batch. 
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.tapas(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]

        #################### aggregation and classification logits ###############################
        
        logits_aggregation = None
        if self.config.num_aggregation_labels > 0:
            logits_aggregation = self._calculate_aggregation_logits(pooled_output)

        logits_cls = None
        if self.config.num_classification_labels > 0:
            logits_cls = self.compute_classification_logits(pooled_output)

        ############################### cell selection logits ####################################
        
        if token_type_ids is None:
            raise ValueError("You have to specify token type ids")
        
        token_types = ["segment_ids", "column_ids", "row_ids", "prev_label_ids", "column_ranks",
                            "inv_column_ranks", "numeric_relations"]
        
        row_ids = token_type_ids[:,:,token_types.index("row_ids")]
        column_ids = token_type_ids[:,:,token_types.index("column_ids")]
        
        # Construct indices for the table.
        row_index = utils.IndexMap(
            indices=torch.min(row_ids, torch.as_tensor(self.config.max_num_rows - 1, device=row_ids.device)),
            num_segments=self.config.max_num_rows,
            batch_dims=1)
        col_index = utils.IndexMap(
            indices=torch.min(column_ids, torch.as_tensor(self.config.max_num_columns - 1, device=column_ids.device)),
            num_segments=self.config.max_num_columns,
            batch_dims=1)
        cell_index = utils.ProductIndexMap(row_index, col_index)
        
        # Masks.
        input_shape = input_ids.size() if input_ids is not None else inputs_embeds.size()[:-1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        # Table cells only, without question tokens and table headers.
        if table_mask is None:
            table_mask = torch.where(row_ids > 0, torch.ones_like(row_ids),
                                    torch.zeros_like(row_ids))
        # torch.FloatTensor[batch_size, seq_length] there's probably a more elegant way to do the 4 lines below
        input_mask_float = attention_mask.type(torch.FloatTensor).to(device)
        table_mask_float = table_mask.type(torch.FloatTensor).to(device)
        # Mask for cells that exist in the table (i.e. that are not padding).
        cell_mask, _ = utils.reduce_mean(input_mask_float, cell_index)

        token_logits = self.compute_token_logits(sequence_output, self.config.temperature)

        # Compute logits per column. These are used to select a column.
        column_logits = None
        selection_loss_per_example = None
        if self.config.select_one_column:
            column_logits = utils.compute_column_logits(
                sequence_output,
                self.column_output_weights,
                self.column_output_bias,
                cell_index,
                cell_mask,
                self.config.allow_empty_column_selection
            )
            # Compute cell selection loss per example.
            if label_ids is not None:
                selection_loss_per_example, logits = utils._single_column_cell_selection_loss(token_logits, column_logits, label_ids,
                                                                                            cell_index, col_index, cell_mask)

        # Total loss calculation
        total_loss = 0.0
        is_supervised = not self.config.num_aggregation_labels > 0 or not self.config.use_answer_as_supervision

        ### Semi-supervised cell selection in case of no aggregation
        #############################################################

        # If the answer (the denotation) appears directly in the table we might
        # select the answer without applying any aggregation function. There are
        # some ambiguous cases, see _calculate_aggregate_mask for more info.
        # `aggregate_mask` is 1 for examples where we chose to aggregate and 0
        #  for examples where we chose to select the answer directly.
        # `label_ids` encodes the positions of the answer appearing in the table.
        if is_supervised:
            aggregate_mask = None
        else:
            # <float32>[batch_size]
            print("we are here")
            aggregate_mask = utils._calculate_aggregate_mask(
                answer,
                pooled_output,
                self.config.cell_select_pref,
                label_ids
            )
            
        print(aggregate_mask)

        return logits_aggregation, logits_cls, token_logits, column_logits, selection_loss_per_example