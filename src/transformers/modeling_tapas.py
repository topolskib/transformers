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

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from .configuration_tapas import TapasConfig
from .modeling_bert import BertLayerNorm, BertModel, BertPreTrainedModel, gelu


logger = logging.getLogger(__name__)

TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "tapas-base",
    "tapas-large",
    # See all TAPAS models at https://huggingface.co/models?filter=tapas
]

def load_tf_weights_in_tapas(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model. 2 lines of code added compared to load_tf_weights_in_tapas (embeddings)
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
        # we currently ignore pooler and classification heads (MLM + next sentence prediction) 
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step", "pooler", "cls"]
            for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            continue
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

        embeddings = input_embeds + position_embeddings
        
        token_type_embedding_name = "token_type_embeddings"
        
        for i in range(self.number_of_token_type_embeddings):
            name="%s_%d" % (token_type_embedding_name, i)
            embeddings += getattr(self, name)(token_type_ids[:,:,i])

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TapasModel(BertModel):
    """
    This class overrides :class:`~transformers.BertModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = TapasConfig
    base_model_prefix = "tapas"

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = TapasEmbeddings(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


class TapasForMaskedLM(BertPreTrainedModel):
    config_class = TapasConfig
    base_model_prefix = "tapas"

    def __init__(self, config):
        super().__init__(config)

        self.tapas = TapasModel(config)
        self.lm_head = TapasLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                DeprecationWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        outputs = self.tapas(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)



class TapasLMHead(nn.Module):
    """Tapas Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


