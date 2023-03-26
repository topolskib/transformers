# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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


import inspect
import unittest

from transformers import UdopConfig, is_torch_available
from transformers.testing_utils import (
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    slow,
    torch_device,
)
from transformers.utils import cached_property

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import T5Tokenizer, UdopForConditionalGeneration, UdopModel
    from transformers.models.udop.modeling_udop import UDOP_PRETRAINED_MODEL_ARCHIVE_LIST


class UdopModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        encoder_seq_length=7,
        decoder_seq_length=9,
        # For common tests
        is_training=True,
        use_attention_mask=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        d_ff=37,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        initializer_factor=0.002,
        eos_token_id=1,
        pad_token_id=0,
        scope=None,
        decoder_layers=None,
        range_bbox=1000,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        # For common tests
        self.seq_length = self.decoder_seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.d_ff = d_ff
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.initializer_factor = initializer_factor
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.scope = None
        self.decoder_layers = decoder_layers
        self.range_bbox = range_bbox

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size)
        seg_data = ids_tensor([self.batch_size, self.encoder_seq_length, 4], self.range_bbox).float()
        # Ensure that seg_data is legal
        for i in range(seg_data.shape[0]):
            for j in range(seg_data.shape[1]):
                if seg_data[i, j, 3] < seg_data[i, j, 1]:
                    t = seg_data[i, j, 3]
                    seg_data[i, j, 3] = seg_data[i, j, 1]
                    seg_data[i, j, 1] = t
                if seg_data[i, j, 2] < seg_data[i, j, 0]:
                    t = seg_data[i, j, 2]
                    seg_data[i, j, 2] = seg_data[i, j, 0]
                    seg_data[i, j, 0] = t
        decoder_input_ids = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)

        attention_mask = None
        decoder_attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)
            decoder_attention_mask = ids_tensor([self.batch_size, self.decoder_seq_length], vocab_size=2)

        lm_labels = None
        if self.use_labels:
            lm_labels = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)

        config = self.get_config()

        return (
            config,
            input_ids,
            seg_data,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        )

    def get_config(self):
        return UdopConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_decoder_layers=self.decoder_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        seg_data,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = UdopModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids=input_ids,
            seg_data=seg_data,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        result = model(input_ids=input_ids, seg_data=seg_data, decoder_input_ids=decoder_input_ids)
        decoder_output = result.last_hidden_state
        decoder_past = result.past_key_values
        encoder_output = result.encoder_last_hidden_state

        self.parent.assertEqual(encoder_output.size(), (self.batch_size, self.encoder_seq_length, self.hidden_size))
        self.parent.assertEqual(decoder_output.size(), (self.batch_size, self.decoder_seq_length, self.hidden_size))
        # There should be `num_layers` key value embeddings stored in decoder_past
        self.parent.assertEqual(len(decoder_past), config.num_layers)
        # There should be a self attn key, a self attn value, a cross attn key and a cross attn value stored in each decoder_past tuple
        self.parent.assertEqual(len(decoder_past[0]), 4)

    def create_and_check_with_lm_head(
        self,
        config,
        input_ids,
        seg_data,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = UdopForConditionalGeneration(config=config).to(torch_device).eval()
        outputs = model(
            input_ids=input_ids,
            seg_data=seg_data,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )
        self.parent.assertEqual(len(outputs), 4)
        self.parent.assertEqual(outputs["logits"].size(), (self.batch_size, self.decoder_seq_length, self.vocab_size))
        self.parent.assertEqual(outputs["loss"].size(), ())

    def create_and_check_generate_with_past_key_values(
        self,
        config,
        input_ids,
        seg_data,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = UdopForConditionalGeneration(config=config).to(torch_device).eval()
        torch.manual_seed(0)
        output_without_past_cache = model.generate(
            input_ids[:1], seg_data=seg_data[:1, :, :], num_beams=2, max_length=5, do_sample=True, use_cache=False
        )
        torch.manual_seed(0)
        output_with_past_cache = model.generate(
            input_ids[:1], seg_data=seg_data[:1, :, :], num_beams=2, max_length=5, do_sample=True
        )
        self.parent.assertTrue(torch.all(output_with_past_cache == output_without_past_cache))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            seg_data,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "seg_data": seg_data,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "use_cache": False,
        }
        return config, inputs_dict


@require_torch
class UdopModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            UdopModel,
            UdopForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    all_generative_model_classes = (UdopForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = {"feature-extraction": UdopModel} if is_torch_available() else {}
    fx_compatible = False
    test_pruning = False
    test_torchscript = False
    test_head_masking = False
    test_resize_embeddings = True
    test_model_parallel = False
    is_encoder_decoder = True
    # The small UDOP model needs higher percentages for CPU/MP tests
    model_split_percents = [0.8, 0.9]

    def setUp(self):
        self.model_tester = UdopModelTester(self)
        self.config_tester = ConfigTester(self, config_class=UdopConfig, d_model=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_with_lm_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_with_lm_head(*config_and_inputs)

    def test_generate_with_past_key_values(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_generate_with_past_key_values(*config_and_inputs)

    @unittest.skipIf(torch_device == "cpu", "Cant do half precision")
    def test_model_fp16_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_fp16_forward(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = sorted([*signature.parameters.keys()])

            expected_arg_names = [
                "attention_mask",
                "char_ids",
                "char_seg_data",
                "cross_attn_head_mask",
                "decoder_attention_mask",
                "decoder_head_mask",
                "decoder_input_ids",
                "decoder_inputs_embeds",
                "encoder_outputs",
                "head_mask",
                "ids_keep",
                "ids_restore",
                "image_mask_label",
                "input_ids",
                "inputs_embeds",
                "kwargs",
                "mask_ratio",
            ]
            if model_class in self.all_generative_model_classes:
                expected_arg_names.append(
                    "labels",
                )
                expected_arg_names = sorted(expected_arg_names)
            self.assertListEqual(sorted(arg_names[: len(expected_arg_names)]), expected_arg_names)

    @slow
    def test_model_from_pretrained(self):
        for model_name in UDOP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = UdopForConditionalGeneration.from_pretrained(model_name)
            self.assertIsNotNone(model)


def use_task_specific_params(model, task):
    model.config.update(model.config.task_specific_params[task])


@require_torch
@require_sentencepiece
@require_tokenizers
class UdopModelIntegrationTests(unittest.TestCase):
    @cached_property
    def model(self):
        return UdopForConditionalGeneration.from_pretrained("udop-base").to(torch_device)

    @cached_property
    def tokenizer(self):
        return T5Tokenizer.from_pretrained("udop-base")

    @slow
    def test_small_generation(self):
        model = UdopForConditionalGeneration.from_pretrained("microsoft/udop-large").to(torch_device)
        model.config.max_length = 8
        model.config.num_beams = 1
        model.config.do_sample = False
        tokenizer = T5Tokenizer.from_pretrained("microsoft/udop-large")

        input_ids = tokenizer("summarize: Hello there", return_tensors="pt").input_ids.to(torch_device)

        sequences = model.generate(input_ids)

        output_str = tokenizer.batch_decode(sequences, skip_special_tokens=True)[0]
        self.assertTrue(output_str == "Hello there!")
