# coding=utf-8
# Copyright 2020 The Google AI Team, Stanford University and The HuggingFace Inc. team.
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

from typing import Any, Dict, Iterable, List, Mapping, Optional, overload, Text, Tuple, Union
import pandas as pd
import dataclasses

from .tokenization_bert import BertTokenizer, BertTokenizerFast
from .tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING,
    INIT_TOKENIZER_DOCSTRING,
    AddedToken,
    BatchEncoding,
    EncodedInput,
    EncodedInputPair,
    PaddingStrategy,
    PreTokenizedInput,
    PreTokenizedInputPair,
    PreTrainedTokenizerBase,
    TensorType,
    TextInput,
    TextInputPair,
    TruncationStrategy,
)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        # to be added
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    # to be added
}


PRETRAINED_INIT_CONFIGURATION = {
    # to be added
}

# @dataclasses.dataclass(frozen=True)
# class Token:
#   original_text: Text
#   piece: Text

# def _get_pieces(tokens):
#   return (token.piece for token in tokens)


@dataclasses.dataclass(frozen=True)
class TokenCoordinates:
  column_index: int
  row_index: int
  token_index: int


@dataclasses.dataclass
class TokenizedTable:
  rows: List[List[List[Text]]]
  selected_tokens: List[TokenCoordinates]


@dataclasses.dataclass(frozen=True)
class SerializedExample:
  tokens: List[Text]
  column_ids: List[int]
  row_ids: List[int]
  segment_ids: List[int]
  

def _is_inner_wordpiece(token):
    return token.startswith('##')


class TapasTokenizer(BertTokenizer):
    r"""
    Construct an TAPAS tokenizer.

    :class:`~transformers.TapasTokenizer` is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting and wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION

    def __init__(self, 
                cell_trim_length: int = -1,
                max_column_id: int = None,
                max_row_id: int = None,
                strip_column_names: bool = False,
                add_aggregation_candidates: bool = False,
                expand_entity_descriptions: bool = False,
                entity_descriptions_sentence_limit: int = 5,
                #use_document_title: bool = False, suggestion: remove this?
                update_answer_coordinates: bool = False, # Re-compute answer coordinates from the answer text.
                drop_rows_to_fit: bool = False, # Drop last rows if table doesn't fit within max sequence length.
                **kwargs):
        super().__init__(**kwargs)

        # Added tokens - We store this for both slow and fast tokenizers
        # until the serialization of Fast tokenizers is updated
        self.added_tokens_encoder: Dict[str, int] = {}
        self.added_tokens_decoder: Dict[int, str] = {}
        self.unique_no_split_tokens: List[str] = []

        # Additional properties 
        self.cell_trim_length = cell_trim_length
        self.max_column_id = max_column_id if max_column_id is not None else self.model_max_length
        self.max_row_id = max_row_id if max_row_id is not None else self.model_max_length
        self.strip_column_names = strip_column_names
        self.add_aggregation_candidates = add_aggregation_candidates
        self.expand_entity_descriptions = expand_entity_descriptions
        self.entity_descriptions_sentence_limit = entity_descriptions_sentence_limit
        #self.use_document_title = use_document_title
        self.update_answer_coordinates = update_answer_coordinates
        self.drop_rows_to_fit = drop_rows_to_fit
    
    def _tokenize_table(
        self,
        table: pd.DataFrame = None,
        ):
        """Runs tokenizer over columns and table cell texts."""
        tokenized_rows = []
        tokenized_row = []
        # tokenize column headers
        for column in table:
            if self.strip_column_names:
                tokenized_row.append(self.tokenize(''))
            else:
                tokenized_row.append(self.tokenize(column))
        tokenized_rows.append(tokenized_row)

        # tokenize cell values
        for idx, row in table.iterrows():
            tokenized_row = []
            for cell in row:
                tokenized_row.append(self.tokenize(cell))
            tokenized_rows.append(tokenized_row)

        token_coordinates = []
        for row_index, row in enumerate(tokenized_rows):
            for column_index, cell in enumerate(row):
                for token_index, _ in enumerate(cell):
                    token_coordinates.append(
                        TokenCoordinates(
                            row_index=row_index,
                            column_index=column_index,
                            token_index=token_index,
                        ))

        return TokenizedTable(
            rows=tokenized_rows,
            selected_tokens=token_coordinates,
        )
    
    def question_encoding_cost(self, question_tokens):
        # Two extra spots of SEP and CLS.
        return len(question_tokens) + 2
    
    def _get_token_budget(self, question_tokens):
        return self.model_max_length - self.question_encoding_cost(question_tokens)
    
    def _get_table_values(self, table, num_columns,
                        num_rows,
                        num_tokens):
        """Iterates over partial table and returns token, col. and row indexes."""
        for tc in table.selected_tokens:
            # First row is header row.
            if tc.row_index >= num_rows + 1:
                continue
            if tc.column_index >= num_columns:
                continue
            cell = table.rows[tc.row_index][tc.column_index]
            token = cell[tc.token_index]
            word_begin_index = tc.token_index
            # Don't add partial words. Find the starting word piece and check if it
            # fits in the token budget.
            while word_begin_index >= 0 \
                and _is_inner_wordpiece(cell[word_begin_index]):
                word_begin_index -= 1
            if word_begin_index >= num_tokens:
                continue
            yield token, tc.column_index + 1, tc.row_index
    
    def _get_table_boundaries(self,
                            table):
        """Return maximal number of rows, columns and tokens."""
        max_num_tokens = 0
        max_num_columns = 0
        max_num_rows = 0
        for tc in table.selected_tokens:
            max_num_columns = max(max_num_columns, tc.column_index + 1)
            max_num_rows = max(max_num_rows, tc.row_index + 1)
            max_num_tokens = max(max_num_tokens, tc.token_index + 1)
            max_num_columns = min(self.max_column_id, max_num_columns)
            max_num_rows = min(self.max_row_id, max_num_rows)
        return max_num_rows, max_num_columns, max_num_tokens

    def _get_table_cost(self, table, num_columns,
                        num_rows, num_tokens):
        return sum(1 for _ in self._get_table_values(table, num_columns, num_rows,
                                                    num_tokens))
    
    def _get_max_num_tokens(
        self,
        question_tokens,
        tokenized_table,
        num_columns,
        num_rows,
    ):
        """Computes max number of tokens that can be squeezed into the budget."""
        token_budget = self._get_token_budget(question_tokens)
        _, _, max_num_tokens = self._get_table_boundaries(tokenized_table)
        if self.cell_trim_length >= 0 and max_num_tokens > self.cell_trim_length:
            max_num_tokens = self.cell_trim_length
        num_tokens = 0
        for num_tokens in range(max_num_tokens + 1):
            cost = self._get_table_cost(tokenized_table, num_columns, num_rows,
                                        num_tokens + 1)
            if cost > token_budget:
                break
        if num_tokens < max_num_tokens:
            if self._cell_trim_length >= 0:
                # We don't allow dynamic trimming if a cell_trim_length is set.
                return None
            if num_tokens == 0:
                return None
        return num_tokens
    
    def _get_num_columns(self, table):
        num_columns = table.shape[1]
        if num_columns >= self.max_column_id:
            raise ValueError('Too many columns')
        return num_columns

    def _get_num_rows(self, table, drop_rows_to_fit):
        num_rows = table.shape[0]
        if num_rows >= self.max_row_id:
            if drop_rows_to_fit:
                num_rows = self.max_row_id - 1
            else:
                raise ValueError('Too many rows')
        return num_rows
    
    def _serialize_text(self, question_tokens):
        """Serialzes texts in index arrays."""
        tokens = []
        segment_ids = []
        column_ids = []
        row_ids = []

        # add [CLS] token at the beginning
        tokens.append(self.cls_token)
        segment_ids.append(0)
        column_ids.append(0)
        row_ids.append(0)

        for token in question_tokens:
            tokens.append(token)
            segment_ids.append(0)
            column_ids.append(0)
            row_ids.append(0)

        return tokens, segment_ids, column_ids, row_ids

    def _serialize(
        self,
        question_tokens,
        table,
        num_columns,
        num_rows,
        num_tokens,
    ):
        """Serializes table and text."""
        tokens, segment_ids, column_ids, row_ids = self._serialize_text(
            question_tokens)

        # add [SEP] token between question and table tokens
        tokens.append(self.sep_token)
        segment_ids.append(0)
        column_ids.append(0)
        row_ids.append(0)

        for token, column_id, row_id in self._get_table_values(
            table, num_columns, num_rows, num_tokens):
            tokens.append(token)
            segment_ids.append(1)
            column_ids.append(column_id)
            row_ids.append(row_id)

        return SerializedExample(
            tokens=tokens,
            segment_ids=segment_ids,
            column_ids=column_ids,
            row_ids=row_ids,
        )
        
    def _to_features(self, tokens, token_ids_dict, table, question):
        """Produces a dict of features."""
        tokens = list(tokens)
        token_ids_dict = {
            key: list(values) for key, values in token_ids_dict.items()
        }

        length = len(tokens)
        for values in token_ids_dict.values():
            if len(values) != length:
                raise ValueError('Inconsistent length')

        input_ids = self.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        return NotImplementedError

        # self._pad_to_seq_length(input_ids)
        # self._pad_to_seq_length(input_mask)
        # for values in token_ids_dict.values():
        #     self._pad_to_seq_length(values)

        # assert len(input_ids) == self._max_seq_length
        # assert len(input_mask) == self._max_seq_length
        # for values in token_ids_dict.values():
        #     assert len(values) == self._max_seq_length

        # features = collections.OrderedDict()
        # features['input_ids'] = create_int_feature(input_ids)
        # features['input_mask'] = create_int_feature(input_mask)
        # for key, values in sorted(token_ids_dict.items()):
        #     features[key] = create_int_feature(values)

        # return features
    
    def _to_trimmed_features(
            self,
            question,
            table,
            question_tokens,
            tokenized_table,
            num_columns,
            num_rows,
            drop_rows_to_fit = False,
        ):
            """Finds optiomal number of table tokens to include and serializes."""
            init_num_rows = num_rows
            while True:
                num_tokens = self._get_max_num_tokens(
                    question_tokens,
                    tokenized_table,
                    num_rows=num_rows,
                    num_columns=num_columns,
                )
                if num_tokens is not None:
                    # We could fit the table.
                    break
                if not drop_rows_to_fit or num_rows == 0:
                    raise ValueError('Sequence too long')
                # Try to drop a row to fit the table.
                num_rows -= 1
            
            serialized_example = self._serialize(question_tokens, tokenized_table,
                                                num_columns, num_rows, num_tokens)

            assert len(serialized_example.tokens) <= self.model_max_length

            feature_dict = {
                'column_ids': serialized_example.column_ids,
                'row_ids': serialized_example.row_ids,
                'segment_ids': serialized_example.segment_ids,
            }

            return feature_dict
        # features = self._to_features(
        #     serialized_example.tokens, feature_dict, table=table, question=question)
        # return serialized_example, features
    
    # def _batch_encode_plus(
    #     self,
    #     table: pd.DataFrame = None,
    #     queries: Union[
    #         List[TextInput],
    #         List[TextInputPair],
    #         List[PreTokenizedInput],
    #         List[PreTokenizedInputPair],
    #         List[EncodedInput],
    #         List[EncodedInputPair],
    #     ],
    #     add_special_tokens: bool = True,
    #     padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    #     truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    #     max_length: Optional[int] = None,
    #     stride: int = 0,
    #     is_split_into_words: bool = False,
    #     pad_to_multiple_of: Optional[int] = None,
    #     return_tensors: Optional[Union[str, TensorType]] = None,
    #     return_token_type_ids: Optional[bool] = None,
    #     return_attention_mask: Optional[bool] = None,
    #     return_overflowing_tokens: bool = False,
    #     return_special_tokens_mask: bool = False,
    #     return_offsets_mapping: bool = False,
    #     return_length: bool = False,
    #     verbose: bool = True,
    #     **kwargs
    # ) -> BatchEncoding:
    #     def get_input_ids(text):
    #         if isinstance(text, str):
    #             tokens = self.tokenize(text, **kwargs)
    #             return self.convert_tokens_to_ids(tokens)
    #         elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], str):
    #             if is_split_into_words:
    #                 tokens = list(
    #                     itertools.chain(*(self.tokenize(t, is_split_into_words=True, **kwargs) for t in text))
    #                 )
    #                 return self.convert_tokens_to_ids(tokens)
    #             else:
    #                 return self.convert_tokens_to_ids(text)
    #         elif isinstance(text, (list, tuple)) and len(text) > 0 and isinstance(text[0], int):
    #             return text
    #         else:
    #             raise ValueError(
    #                 "Input is not valid. Should be a string, a list/tuple of strings or a list/tuple of integers."
    #             )

    #     if return_offsets_mapping:
    #         raise NotImplementedError(
    #             "return_offset_mapping is not available when using Python tokenizers."
    #             "To use this feature, change your tokenizer to one deriving from "
    #             "transformers.PreTrainedTokenizerFast."
    #         )

    #     if "is_pretokenized" in kwargs:
    #         warnings.warn(
    #             "`is_pretokenized` is deprecated and will be removed in a future version, use `is_split_into_words` instead.",
    #             FutureWarning,
    #         )
    #         is_split_into_words = kwargs.pop("is_pretokenized")

    #     input_ids = []
    #     for ids_or_pair_ids in queries:
    #         if not isinstance(ids_or_pair_ids, (list, tuple)):
    #             ids, pair_ids = ids_or_pair_ids, None
    #         elif is_split_into_words and not isinstance(ids_or_pair_ids[0], (list, tuple)):
    #             ids, pair_ids = ids_or_pair_ids, None
    #         else:
    #             ids, pair_ids = ids_or_pair_ids

    #         first_ids = get_input_ids(ids)
    #         second_ids = get_input_ids(pair_ids) if pair_ids is not None else None
    #         input_ids.append((first_ids, second_ids))

    #     batch_outputs = self._batch_prepare_for_model(
    #         input_ids,
    #         add_special_tokens=add_special_tokens,
    #         padding_strategy=padding_strategy,
    #         truncation_strategy=truncation_strategy,
    #         max_length=max_length,
    #         stride=stride,
    #         pad_to_multiple_of=pad_to_multiple_of,
    #         return_attention_mask=return_attention_mask,
    #         return_token_type_ids=return_token_type_ids,
    #         return_overflowing_tokens=return_overflowing_tokens,
    #         return_special_tokens_mask=return_special_tokens_mask,
    #         return_length=return_length,
    #         return_tensors=return_tensors,
    #         verbose=verbose,
    #     )

    #     return BatchEncoding(batch_outputs)
    
    
    # @add_end_docstrings(ENCODE_KWARGS_DOCSTRING, ENCODE_PLUS_ADDITIONAL_KWARGS_DOCSTRING)
    # def _batch_prepare_for_model(
    #     self,
    #     batch_ids_pairs: List[Union[PreTokenizedInputPair, Tuple[List[int], None]]],
    #     add_special_tokens: bool = True,
    #     padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
    #     truncation_strategy: TruncationStrategy = TruncationStrategy.DO_NOT_TRUNCATE,
    #     max_length: Optional[int] = None,
    #     stride: int = 0,
    #     pad_to_multiple_of: Optional[int] = None,
    #     return_tensors: Optional[str] = None,
    #     return_token_type_ids: Optional[bool] = None,
    #     return_attention_mask: Optional[bool] = None,
    #     return_overflowing_tokens: bool = False,
    #     return_special_tokens_mask: bool = False,
    #     return_length: bool = False,
    #     verbose: bool = True,
    # ) -> BatchEncoding:
    #     """
    #     Prepares a sequence of input id, or a pair of sequences of inputs ids so that it can be used by the model.
    #     It adds special tokens, truncates sequences if overflowing while taking into account the special tokens and
    #     manages a moving window (with user defined stride) for overflowing tokens
    #     Args:
    #         batch_ids_pairs: list of tokenized input ids or input ids pairs
    #     """

    #     batch_outputs = {}
    #     for first_ids, second_ids in batch_ids_pairs:
    #         outputs = self.prepare_for_model(
    #             first_ids,
    #             second_ids,
    #             add_special_tokens=add_special_tokens,
    #             padding=PaddingStrategy.DO_NOT_PAD.value,  # we pad in batch afterward
    #             truncation=truncation_strategy.value,
    #             max_length=max_length,
    #             stride=stride,
    #             pad_to_multiple_of=None,  # we pad in batch afterward
    #             return_attention_mask=False,  # we pad in batch afterward
    #             return_token_type_ids=return_token_type_ids,
    #             return_overflowing_tokens=return_overflowing_tokens,
    #             return_special_tokens_mask=return_special_tokens_mask,
    #             return_length=return_length,
    #             return_tensors=None,  # We convert the whole batch to tensors at the end
    #             prepend_batch_axis=False,
    #             verbose=verbose,
    #         )

    #         for key, value in outputs.items():
    #             if key not in batch_outputs:
    #                 batch_outputs[key] = []
    #             batch_outputs[key].append(value)

    #     batch_outputs = self.pad(
    #         batch_outputs,
    #         padding=padding_strategy.value,
    #         max_length=max_length,
    #         pad_to_multiple_of=pad_to_multiple_of,
    #         return_attention_mask=return_attention_mask,
    #     )

    #     batch_outputs = BatchEncoding(batch_outputs, tensor_type=return_tensors)

    #     return batch_outputs