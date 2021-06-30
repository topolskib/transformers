# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch SegFormer model. """


import unittest
import inspect

from transformers import is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, slow, torch_device, require_vision

from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch

    from transformers import (
        SegFormerConfig,
        SegFormerForImageSegmentation,
        SegFormerModel,
    )
    from transformers.models.segformer.modeling_segformer import SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image

    from transformers import SegFormerFeatureExtractor


class SegFormerConfigTester(ConfigTester):
    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        self.parent.assertTrue(hasattr(config, "hidden_sizes"))
        self.parent.assertTrue(hasattr(config, "num_attention_heads"))
        self.parent.assertTrue(hasattr(config, "num_encoder_blocks"))

class SegFormerModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        image_size=30,
        num_channels=3,
        num_encoder_blocks=4,
        depths=[2, 2, 2, 2],
        hidden_sizes=[16, 32, 64, 128],
        num_attention_heads=[1, 2, 4, 8],
        is_training=True,
        use_labels=True,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        num_labels=3,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_encoder_blocks = num_encoder_blocks
        self.depths = depths
        self.hidden_sizes = hidden_sizes
        self.num_attention_heads = num_attention_heads
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.scope = scope

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size, self.image_size, self.image_size], self.num_labels)

        config = SegFormerConfig(
            image_size=self.image_size,
            num_channels=self.num_channels,
            num_encoder_blocks=self.num_encoder_blocks,
            depths=self.depths,
            hidden_sizes=self.hidden_sizes,
            num_attention_heads=self.num_attention_heads,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
        )

        return config, pixel_values, labels

    def create_and_check_model(
        self, config, pixel_values, labels
    ):
        model = SegFormerModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.hidden_sizes[-1], self.image_size, self.image_size))

    def create_and_check_for_image_segmentation(
        self, config, pixel_values, labels
    ):
        config.num_labels = self.num_labels
        model = SegFormerForImageSegmentation(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels, ))
        result = model(pixel_values, labels=labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.num_labels, ))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values,
            labels,
        ) = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class SegFormerModelTest(ModelTesterMixin, unittest.TestCase):

    all_model_classes = (
        (
            SegFormerModel,
            SegFormerForImageSegmentation,
        )
        if is_torch_available()
        else ()
    )

    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = SegFormerModelTester(self)
        self.config_tester = SegFormerConfigTester(self, config_class=SegFormerConfig)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_image_segmentation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_segmentation(*config_and_inputs)
    
    @unittest.skip("SegFormer does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip("SegFormer does not have get_input_embeddings method and get_output_embeddings methods")
    def test_model_common_attributes(self):
        pass
    
    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = False
            config.return_dict = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions
            
            expected_num_attentions = sum(self.model_tester.depths)       
            self.assertEqual(len(attentions), expected_num_attentions)

            # check that output_attentions also work using config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))
            attentions = outputs.attentions

            self.assertEqual(len(attentions), expected_num_attentions)

            # verify the first attentions (first block, first layer)
            self.assertListEqual(
                    list(attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads[0], None, None],
            )
            
            # verify the last attentions (last block, last layer)
            self.assertListEqual(
                    list(attentions[-1].shape[-3:]),
                    [self.model_tester.num_attention_heads[-1], None, None],
            )
            
            out_len = len(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            inputs_dict["output_hidden_states"] = True
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            if hasattr(self.model_tester, "num_hidden_states_types"):
                added_hidden_states = self.model_tester.num_hidden_states_types
            else:
                added_hidden_states = 1
            self.assertEqual(out_len + added_hidden_states, len(outputs))

            self_attentions = outputs.attentions

            self.assertEqual(len(self_attentions), expected_num_attentions)
            self.assertListEqual(
                    list(self_attentions[0].shape[-3:]),
                    [self.model_tester.num_attention_heads[0], None, None],
            )
    
    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = self.model_tester.num_encoder_blocks
            self.assertEqual(len(hidden_states), expected_num_layers)

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [None, self.model_tester.hidden_sizes[0]],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)
    
    @slow
    def test_model_from_pretrained(self):
        for model_name in SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = SegFormerModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


@require_torch
class SegFormerModelIntegrationTest(unittest.TestCase):
    @slow
    def test_inference_image_segmentation(self):
        # TODO replace nielsr by nvidia
        model = SegFormerForImageSegmentation.from_pretrained("nielsr/segformer-b0-finetuned-ade-512-512")
        pixel_values = torch.randn((2, 3, 224, 224))
        outputs = model(pixel_values)

        expected_shape = torch.Size((2, model.config.num_labels, 224, 224))
        self.assertEqual(outputs.logits.shape, expected_shape)

        # TODO Replace values below with what was printed above.
        expected_slice = torch.tensor(
            [[[-0.0483, 0.1188, -0.0313], [-0.0606, 0.1435, 0.0199], [-0.0235, 0.1519, 0.0175]]]
        )

        self.assertTrue(torch.allclose(outputs.last_hidden_state[:, :3, :3], expected_slice, atol=1e-4))
