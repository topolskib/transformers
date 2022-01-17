# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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


import json
import os
import unittest
from typing import List

import pandas as pd

from transformers import TapexTokenizer
from transformers.models.tapex.tokenization_tapex import VOCAB_FILES_NAMES
from transformers.testing_utils import require_pandas, slow

from .test_tokenization_common import TokenizerTesterMixin


@require_pandas
class TapexTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = TapexTokenizer
    test_rust_tokenizer = False
    from_pretrained_kwargs = {"cls_token": "<s>"}
    test_seq2seq = False
    maxDiff = None

    def setUp(self):
        super().setUp()

        # Adapted from Sennrich et al. 2015 and https://github.com/rsennrich/subword-nmt
        vocab = [
            "l",
            "o",
            "w",
            "e",
            "r",
            "s",
            "t",
            "i",
            "d",
            "n",
            "\u0120",
            "\u0120l",
            "\u0120n",
            "\u0120lo",
            "\u0120low",
            "er",
            "\u0120lowest",
            "\u0120newer",
            "\u0120wider",
            "<unk>",
        ]
        vocab_tokens = dict(zip(vocab, range(len(vocab))))
        merges = ["#version: 0.2", "\u0120 l", "\u0120l o", "\u0120lo w", "e r", ""]
        self.special_tokens_map = {"unk_token": "<unk>"}

        self.vocab_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["vocab_file"])
        self.merges_file = os.path.join(self.tmpdirname, VOCAB_FILES_NAMES["merges_file"])
        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab_tokens) + "\n")
        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_table(
        self,
        tokenizer: TapexTokenizer,
        length=5,
    ):
        toks = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in range(len(tokenizer))]

        if length == 0:
            data = {}
        else:
            data = {toks[0]: [toks[tok] for tok in range(1, length)]}

        table = pd.DataFrame.from_dict(data)

        return table

    def get_table_and_query(
        self,
        tokenizer: TapexTokenizer,
        length=5,
    ):
        toks = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in range(len(tokenizer))]
        table = self.get_table(tokenizer, length=length - 3)
        query = " ".join(toks[:3])

        return table, query

    def get_clean_sequence(
        self,
        tokenizer: TapexTokenizer,
        with_prefix_space=False,
        max_length=20,
        min_length=5,
        empty_table: bool = False,
        add_special_tokens: bool = True,
        return_table_and_query: bool = False,
    ):

        toks = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in range(len(tokenizer))]

        if empty_table:
            table = pd.DataFrame.from_dict({})
            query = " ".join(toks[:min_length])
        else:
            data = {toks[0]: [toks[tok] for tok in range(1, min_length - 3)]}
            table = pd.DataFrame.from_dict(data)
            query = " ".join(toks[:3])

        output_ids = tokenizer.encode(table, query, add_special_tokens=add_special_tokens)
        output_txt = tokenizer.decode(output_ids)

        assert len(output_ids) >= min_length, "Update the code to generate the sequences so that they are larger"
        assert len(output_ids) <= max_length, "Update the code to generate the sequences so that they are smaller"

        if return_table_and_query:
            return output_txt, output_ids, table, query

        return output_txt, output_ids

    def get_tokenizer(self, **kwargs):
        kwargs.update(self.special_tokens_map)
        return self.tokenizer_class.from_pretrained(self.tmpdirname, **kwargs)

    def get_input_output_texts(self, tokenizer):
        input_text = "lower newer"
        output_text = "lower newer"
        return input_text, output_text

    def test_full_tokenizer_roberta(self):
        tokenizer = self.tokenizer_class(self.vocab_file, self.merges_file, **self.special_tokens_map)
        text = "lower newer"
        bpe_tokens = ["l", "o", "w", "er", "\u0120", "n", "e", "w", "er"]
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.unk_token]
        input_bpe_tokens = [0, 1, 2, 15, 10, 9, 3, 2, 15, 19]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def roberta_dict_integration_testing(self):
        tokenizer = self.get_tokenizer()

        self.assertListEqual(tokenizer.encode("Hello world!", add_special_tokens=False), [0, 31414, 232, 328, 2])
        self.assertListEqual(
            tokenizer.encode("Hello world! cécé herlolip 418", add_special_tokens=False),
            [0, 31414, 232, 328, 740, 1140, 12695, 69, 46078, 1588, 2],
        )

    def test_add_special_tokens(self):
        tokenizers: List[TapexTokenizer] = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                input_table = self.get_table(tokenizer, length=0)

                special_token = "[SPECIAL_TOKEN]"

                tokenizer.add_special_tokens({"cls_token": special_token})
                print("Input table:", input_table)
                print("Special token:", special_token)
                encoded_special_token = tokenizer.encode(input_table, special_token, add_special_tokens=False)
                self.assertEqual(len(encoded_special_token), 1)

                decoded = tokenizer.decode(encoded_special_token, skip_special_tokens=True)
                self.assertTrue(special_token not in decoded)

    def test_add_tokens_tokenizer(self):
        tokenizers: List[TapexTokenizer] = self.get_tokenizers(do_lower_case=False)
        for tokenizer in tokenizers:
            with self.subTest(f"{tokenizer.__class__.__name__}"):
                table = self.get_table(tokenizer, length=0)
                vocab_size = tokenizer.vocab_size
                all_size = len(tokenizer)

                self.assertNotEqual(vocab_size, 0)

                # We usually have added tokens from the start in tests because our vocab fixtures are
                # smaller than the original vocabs - let's not assert this
                # self.assertEqual(vocab_size, all_size)

                new_toks = ["aaaaa bbbbbb", "cccccccccdddddddd"]
                added_toks = tokenizer.add_tokens(new_toks)
                vocab_size_2 = tokenizer.vocab_size
                all_size_2 = len(tokenizer)

                self.assertNotEqual(vocab_size_2, 0)
                self.assertEqual(vocab_size, vocab_size_2)
                self.assertEqual(added_toks, len(new_toks))
                self.assertEqual(all_size_2, all_size + len(new_toks))

                tokens = tokenizer.encode(table, "aaaaa bbbbbb low cccccccccdddddddd l", add_special_tokens=False)

                print(tokens)
                for token in tokens:
                    print(token, tokenizer.decode([token]))

                self.assertGreaterEqual(len(tokens), 4)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)

                new_toks_2 = {"eos_token": ">>>>|||<||<<|<<", "pad_token": "<<<<<|||>|>>>>|>"}
                added_toks_2 = tokenizer.add_special_tokens(new_toks_2)
                vocab_size_3 = tokenizer.vocab_size
                all_size_3 = len(tokenizer)

                self.assertNotEqual(vocab_size_3, 0)
                self.assertEqual(vocab_size, vocab_size_3)
                self.assertEqual(added_toks_2, len(new_toks_2))
                self.assertEqual(all_size_3, all_size_2 + len(new_toks_2))

                tokens = tokenizer.encode(
                    table,
                    ">>>>|||<||<<|<< aaaaabbbbbb low cccccccccdddddddd <<<<<|||>|>>>>|> l",
                    add_special_tokens=False,
                )

                self.assertGreaterEqual(len(tokens), 6)
                self.assertGreater(tokens[0], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[0], tokens[1])
                self.assertGreater(tokens[-2], tokenizer.vocab_size - 1)
                self.assertGreater(tokens[-2], tokens[-3])
                self.assertEqual(tokens[0], tokenizer.eos_token_id)
                self.assertEqual(tokens[-2], tokenizer.pad_token_id)

    @slow
    def test_full_tokenizer(self):
        question = "Greece held its last Summer Olympics in 2004"
        table_dict = {
            "header": ["Year", "City", "Country", "Nations"],
            "rows": [
                [1896, "Athens", "Greece", 14],
                [1900, "Paris", "France", 24],
                [1904, "St. Louis", "USA", 12],
                [2004, "Athens", "Greece", 201],
                [2008, "Beijing", "China", 204],
                [2012, "London", "UK", 204],
            ],
        }
        table = pd.DataFrame.from_dict(table_dict["rows"])
        table.columns = table_dict["header"]

        # TODO replace nielsr by microsoft
        tokenizer = TapexTokenizer.from_pretrained("nielsr/tapex-large-finetuned-wtq")
        encoding = tokenizer(table, question)

        print(encoding.input_ids)
        
        # fmt: off
        expected_results = {'input_ids': [0, 571, 5314, 1755, 547, 63, 94, 1035, 1021, 31434, 2857, 11, 4482, 11311, 4832, 76, 1721, 343, 1721, 247, 1721, 3949, 3236, 112, 4832, 42773, 1721, 23, 27859, 1721, 821, 5314, 1755, 1721, 501, 3236, 132, 4832, 23137, 1721, 2242, 354, 1721, 6664, 2389, 1721, 706, 3236, 155, 4832, 42224, 1721, 1690, 4, 26120, 354, 1721, 201, 102, 1721, 316, 3236, 204, 4832, 4482, 1721, 23, 27859, 1721, 821, 5314, 1755, 1721, 21458, 3236, 195, 4832, 2266, 1721, 28, 40049, 1721, 1855, 1243, 1721, 28325, 3236, 231, 4832, 1125, 1721, 784, 24639, 1721, 1717, 330, 1721, 28325, 2]}
        # fmt: on

        self.assertListEqual(encoding.input_ids, expected_results["input_ids"])
