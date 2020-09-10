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
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        # currently I also skip the classification heads on top since I first convert only the base model 
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step", "seq_relationship",
            "column_output_bias", "column_output_weights", "output_bias", "output_weights"]
            for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        # if the variable is not a classification head, change first scope name to "tapas"
        if name[0] != "cls":
            name[0] = "tapas"
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
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
        
        # cell selection head
        """init_cell_selection_weights_to_zero: Whether the initial weights should be
        set to 0. This ensures that all tokens have the same prior probability."""
        if config.init_cell_selection_weights_to_zero: 
            self.output_weights = nn.Parameter(torch.zeros(config.hidden_size)) 
        else:
            self.output_weights = nn.Parameter(torch.empty(config.hidden_size))
            nn.init.normal_(self.output_weights, std=0.02) # here, a truncated normal is used in the original implementation
        self.output_bias = nn.Parameter(torch.zeros([]))

        # classification head
        self.output_weights_cls = nn.Parameter(torch.empty([config.num_classification_labels, config.hidden_size]))
        nn.init.normal_(self.output_weights_cls, std=0.02) # here, a truncated normal is used in the original implementation
        self.output_bias_cls = nn.Parameter(torch.zeros([config.num_classification_labels]))

        # aggregation head
        self.output_weights_agg = nn.Parameter(torch.empty([config.num_aggregation_labels, config.hidden_size]))
        nn.init.normal_(self.output_weights_agg, std=0.02) # here, a truncated normal is used in the original implementation
        self.output_bias_agg = nn.Parameter(torch.zeros([config.num_aggregation_labels]))

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
            pooled_output: <float>[batch_size, hidden_dim] Output of the pooler (BertPooler) on top of the encoder layer.
        Returns:
            logits_aggregation: <float32>[batch_size, config.num_aggregation_labels] Logits per aggregation operation.
        """
        logits_aggregation = torch.matmul(pooled_output, self.output_weights_agg.T)
        logits_aggregation += self.output_bias_agg
        
        return logits_aggregation

    def _calculate_aggregate_mask(self, answer, pooled_output, cell_select_pref, label_ids):
        """Finds examples where the model should select cells with no aggregation.
        Returns a mask that determines for which examples should the model select
        answers directly from the table, without any aggregation function. If the
        answer is a piece of text the case is unambiguous as aggregation functions
        only apply to numbers. If the answer is a number but does not appear in the
        table then we must use some aggregation case. The ambiguous case is when the
        answer is a number that also appears in the table. In this case we use the
        aggregation function probabilities predicted by the model to decide whether
        to select or aggregate. The threshold for this is a hyperparameter
        `cell_select_pref`.
        Args:
            answer: <float32>[batch_size]
            pooled_output: <float32>[batch_size, hidden_size]
            cell_select_pref: Preference for cell selection in ambiguous cases.
            label_ids: torch.LongTensor[batch_size, seq_length]
        Returns:
            aggregate_mask: <float32>[batch_size] A mask set to 1 for examples that
            should use aggregation functions.
        """
        # <float32>[batch_size]
        aggregate_mask_init = torch.logical_not(torch.isnan(answer)).type(torch.FloatTensor)
        logits_aggregation = self._calculate_aggregation_logits(pooled_output)
        dist_aggregation = torch.distributions.categorical.Categorical(logits=logits_aggregation)
        # Index 0 correponds to "no aggregation".
        aggregation_ops_total_mass = torch.sum(
                                        dist_aggregation.probs[:, 1:], axis=1)

        # Cell selection examples according to current model.
        is_pred_cell_selection = aggregation_ops_total_mass <= cell_select_pref

        # Examples with non-empty cell selection supervision.
        is_cell_supervision_available = torch.sum(label_ids, axis=1) > 0

        aggregate_mask = torch.where(
                            torch.logical_and(is_pred_cell_selection, is_cell_supervision_available),
                            torch.zeros_like(aggregate_mask_init, dtype=torch.float32), 
                            aggregate_mask_init)
        
        aggregate_mask = aggregate_mask.detach()
        
        return aggregate_mask

    def _calculate_aggregation_loss_known(self, logits_aggregation, aggregate_mask,
                                      aggregation_function_id):
        """Calculates aggregation loss when its type is known during training.
        In the weakly supervised setting, the only known information is that for
        cell selection examples, "no aggregation" should be predicted. For other
        examples (those that require aggregation), no loss is accumulated.
        In the setting where aggregation type is always known, standard cross entropy
        loss is accumulated for all examples.
        Args:
            logits_aggregation: <float32>[batch_size, num_aggregation_labels]
            aggregate_mask: <float32>[batch_size]
            aggregation_function_id: torch.LongTensor[batch_size]
        Returns:
            aggregation_loss_known: <float32>[batch_size, num_aggregation_labels]
        """
        if self.config.use_answer_as_supervision:
            # Prepare "no aggregation" targets for cell selection examples.
            target_aggregation = torch.zeros_like(aggregate_mask, dtype=torch.long)
        else:
            # Use aggregation supervision as the target.
            target_aggregation = aggregation_function_id

        batch_size = aggregate_mask.size()[0]

        one_hot_labels = torch.zeros(batch_size, config.num_aggregation_labels, dtype=torch.float32)
        one_hot_labels[torch.arange(batch_size), target_aggregation] = 1.0
        
        log_probs = torch.nn.functional.log_softmax(logits_aggregation, dim=-1)
        
        # <float32>[batch_size]
        per_example_aggregation_intermediate = -torch.sum(
            one_hot_labels * log_probs, axis=-1)
        if self.config.use_answer_as_supervision:
            # Accumulate loss only for examples requiring cell selection
            # (no aggregation).
            return per_example_aggregation_intermediate * (1 - aggregate_mask)
        else:
            return per_example_aggregation_intermediate

    def _calculate_aggregation_loss_unknown(self, logits_aggregation, aggregate_mask):
        """Calculates aggregation loss in the case of answer supervision."""
        
        dist_aggregation = torch.distributions.categorical.Categorical(logits=logits_aggregation)
        # Index 0 correponds to "no aggregation".
        aggregation_ops_total_mass = torch.sum(
                                        dist_aggregation.probs[:, 1:], dim=1)
        # Predict some aggregation in case of an answer that needs aggregation.
        # This increases the probability of all aggregation functions, in a way
        # similar to MML, but without considering whether the function gives the
        # correct answer.
        return -torch.log(aggregation_ops_total_mass) * aggregate_mask


    def _calculate_aggregation_loss(self, logits_aggregation, aggregate_mask,
                                    aggregation_function_id):
        """Calculates the aggregation loss per example."""
        per_example_aggregation_loss = self._calculate_aggregation_loss_known(
            logits_aggregation, aggregate_mask, aggregation_function_id)

        if self.config.use_answer_as_supervision:
            # Add aggregation loss for numeric answers that need aggregation.
            per_example_aggregation_loss += self._calculate_aggregation_loss_unknown(
                logits_aggregation, aggregate_mask)
        return self.config.aggregation_loss_importance * per_example_aggregation_loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
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

        agg_logits = self._calculate_aggregation_logits(pooled_output)

        return agg_logits
