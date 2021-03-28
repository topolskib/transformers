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



import unittest
import tempfile

from transformers import is_tf_available, ViTConfig
from transformers.file_utils import cached_property, is_torch_available
# TODO uncomment once available
#, is_vision_available
from transformers.testing_utils import require_tf, slow, torch_device
# require_vision, 

from .test_configuration_common import ConfigTester
from .test_modeling_tf_common import TFModelTesterMixin, ids_tensor, floats_tensor


if is_tf_available():
    import tensorflow as tf

    from transformers import ViTConfig, TFViTForImageClassification, TFViTModel
    from transformers.models.vit.modeling_tf_vit import to_2tuple


# TODO: uncomment once available
#if is_vision_available():
from PIL import Image

#from transformers import ViTFeatureExtractor

# TODO: remove any "input_ids" from this file


class TFViTModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        image_size=30, 
        patch_size=2,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=5,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_sequence_label_size=10,
        initializer_range=0.02,
        num_labels=3,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.scope = scope


    def prepare_config_and_inputs(self):
        # we provide channels as last dimension
        pixel_values = floats_tensor([self.batch_size, self.image_size, self.image_size, self.num_channels])

        labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)

        config = ViTConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
        )

        return config, pixel_values, labels
        

    def create_and_check_model(self, config, pixel_values, labels):
        model = TFViTModel(config=config)
        inputs = {"input_ids": None, "pixel_values": pixel_values}
        result = model(inputs)

        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = to_2tuple(self.image_size)
        patch_size = to_2tuple(self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))

    def create_and_check_for_image_classification(self, config, pixel_values, labels):
        config.num_labels = self.type_sequence_label_size
        model = TFViTForImageClassification(config)
        inputs = {"input_ids": None, "pixel_values": pixel_values, "labels": labels}  
        result = model(inputs)
        
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.type_sequence_label_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            pixel_values,
            labels,
        ) = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_tf
class TFViTModelTest(TFModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_tf_common.py, as ViT does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """


    all_model_classes = (
        (
            TFViTModel,
            TFViTForImageClassification,
        )
        if is_tf_available()
        else ()
    )

    test_head_masking = False
    test_onnx = False
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = TFViTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=ViTConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_for_image_classification(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_image_classification(*config_and_inputs)

    def test_inputs_embeds(self):
        # ViT does not use inputs_embeds
        pass
    
    def test_lm_head_model_random_no_beam_search_generate(self):
        # ViT does not have input_ids and is not a language model
        pass

    def test_lm_head_model_random_beam_search_generate(self):
        # ViT does not have input_ids and is not a language model
        pass

    def test_attention_outputs(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = True
        
        # in ViT, the seq_len equals the number of patches + 1 (we add 1 for the [CLS] token)
        image_size = to_2tuple(self.model_tester.image_size)
        patch_size = to_2tuple(self.model_tester.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        seq_len = num_patches + 1
        encoder_seq_length = getattr(self.model_tester, "encoder_seq_length", seq_len)
        encoder_key_length = getattr(self.model_tester, "key_length", encoder_seq_length)

        def check_encoder_attentions_output(outputs):
            attentions = [
                t.numpy() for t in (outputs.encoder_attentions if config.is_encoder_decoder else outputs.attentions)
            ]
            self.assertEqual(len(attentions), self.model_tester.num_hidden_layers)
            self.assertListEqual(
                list(attentions[0].shape[-3:]),
                [self.model_tester.num_attention_heads, encoder_seq_length, encoder_key_length],
            )

        for model_class in self.all_model_classes:
            inputs_dict["output_attentions"] = True
            inputs_dict["use_cache"] = False
            config.output_hidden_states = False
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            out_len = len(outputs)
            self.assertEqual(config.output_hidden_states, False)
            check_encoder_attentions_output(outputs)

            if self.is_encoder_decoder:
                model = model_class(config)
                outputs = model(self._prepare_for_class(inputs_dict, model_class))
                self.assertEqual(config.output_hidden_states, False)
                check_decoder_attentions_output(outputs)

            # Check that output attentions can also be changed via the config
            del inputs_dict["output_attentions"]
            config.output_attentions = True
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            self.assertEqual(config.output_hidden_states, False)
            check_encoder_attentions_output(outputs)

            # Check attention is always last and order is fine
            inputs_dict["output_attentions"] = True
            config.output_hidden_states = True
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))

            self.assertEqual(out_len + (2 if self.is_encoder_decoder else 1), len(outputs))
            self.assertEqual(model.config.output_hidden_states, True)
            check_encoder_attentions_output(outputs)

    def test_hidden_states_output(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        def check_hidden_states_output(config, inputs_dict, model_class):
            model = model_class(config)
            outputs = model(self._prepare_for_class(inputs_dict, model_class))
            expected_num_layers = getattr(
                self.model_tester, "expected_num_hidden_layers", self.model_tester.num_hidden_layers + 1
            )

            # ViT has a different seq_length
            image_size = to_2tuple(self.model_tester.image_size)
            patch_size = to_2tuple(self.model_tester.patch_size)
            num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
            seq_length = num_patches + 1

            hidden_states = outputs.hidden_states
            self.assertEqual(config.output_attentions, False)
            self.assertEqual(len(hidden_states), expected_num_layers)
            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.hidden_size],
            )

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(config, inputs_dict, model_class)

            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True
            check_hidden_states_output(config, inputs_dict, model_class)
    
    def test_compile_tf_model(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")

        for model_class in self.all_model_classes:
            # ViT uses pixel_values instead of input_ids
            input_ids = tf.keras.Input(batch_shape=(self.model_tester.batch_size, 512), name="input_ids", dtype="int32")
            pixel_values = tf.keras.Input(batch_shape=(self.model_tester.batch_size, self.model_tester.image_size, 
                                                         self.model_tester.image_size, self.model_tester.num_channels), 
                                         name="pixel_values", 
                                         dtype="float32")

            # Prepare our model
            model = model_class(config)
            model(self._prepare_for_class(inputs_dict, model_class))  # Model must be called before saving.
            # Let's load it from the disk to be sure we can use pretrained weights
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname, saved_model=False)
                model = model_class.from_pretrained(tmpdirname)

            outputs_dict = model([input_ids, pixel_values])
            hidden_states = outputs_dict[0]

            # Add a dense layer on top to test integration with other keras modules
            outputs = tf.keras.layers.Dense(2, activation="softmax", name="outputs")(hidden_states)

            # Compile extended model
            extended_model = tf.keras.Model(inputs=[input_ids, pixel_values], outputs=[outputs])
            extended_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    
    @slow
    def test_model_from_pretrained(self):
        # TODO add once available
        # model = TFViTModel.from_pretrained("google/vit-base-patch16-224")
        # self.assertIsNotNone(model)
        pass

# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img = Image.open(requests.get(url, stream=True).raw)
    return img


# @require_tf
# # TODO uncomment once available
# #@require_vision
# class TFViTModelIntegrationTest(unittest.TestCase):
#     @cached_property
#     def default_feature_extractor(self):
#         return ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224") if is_vision_available() else None
    
#     @slow
#     def test_inference_image_classification_head(self):
#         model = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to(torch_device)
        
#         feature_extractor = self.default_feature_extractor
#         image = prepare_img()
#         inputs = feature_extractor(images=image, return_tensors="tf").to(torch_device)

#         # forward pass
#         outputs = model(input_ids=None, pixel_values=pixel_values)
        
#         # verify the logits
#         expected_shape = torch.Size((1, 1000))
#         self.assertEqual(outputs.logits.shape, expected_shape)

#         expected_slice = tf.constant([-0.2744, 0.8215, -0.0836])
#         tf.debugging.assert_near(output[0, :3], expected_slice, atol=1e-4)