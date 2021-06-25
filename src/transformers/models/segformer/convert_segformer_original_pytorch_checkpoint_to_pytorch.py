# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Convert SegFormer checkpoints."""


import argparse
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image

import requests
from transformers import SegFormerConfig, SegFormerFeatureExtractor, SegFormerForImageSegmentation
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def rename_keys(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("backbone"):
            key = key.replace("backbone", "segformer.encoder")
        if "patch_embed" in key:
            # replace for example patch_embed1 by patch_embeddings.0
            idx = key[key.find("patch_embed") + len("patch_embed")]
            key = key.replace(f"patch_embed{idx}", f"patch_embeddings.{int(idx)-1}")
        if "norm" in key:
            key = key.replace("norm", "layer_norm")
        if "segformer.encoder.layer_norm" in key:
            # replace for example layer_norm1 by layer_norm.0
            idx = key[key.find("segformer.encoder.layer_norm") + len("segformer.encoder.layer_norm")]
            key = key.replace(f"layer_norm{idx}", f"layer_norm.{int(idx)-1}")
        if "layer_norm1" in key:
            key = key.replace("layer_norm1", "layer_norm_1")
        if "layer_norm2" in key:
            key = key.replace("layer_norm2", "layer_norm_2")
        if "block" in key:
            # replace for example block1 by block.0
            idx = key[key.find("block") + len("block")]
            key = key.replace(f"block{idx}", f"block.{int(idx)-1}")
        if "attn.q" in key:
            key = key.replace("attn.q", "attention.self.query")
        if "attn.proj" in key:
            key = key.replace("attn.proj", "attention.output.dense")
        if "attn" in key:
            key = key.replace("attn", "attention.self")
        if "fc1" in key:
            key = key.replace("fc1", "dense1")
        if "fc2" in key:
            key = key.replace("fc2", "dense2")
        if key.startswith("decode_head."):
            key = key[len("decode_head.") :]
        if "linear_pred" in key:
            key = key.replace("linear_pred", "classifier")
        if "linear_fuse" in key:
            key = key.replace("linear_fuse.conv", "linear_fuse")
            key = key.replace("linear_fuse.bn", "batch_norm")
        if "linear_c" in key:
            # replace for example linear_c4 by linear_c.3
            idx = key[key.find("linear_c") + len("linear_c")]
            key = key.replace(f"linear_c{idx}", f"linear_c.{int(idx)-1}")
        new_state_dict[key] = value

    return new_state_dict


def read_in_k_v(state_dict, config):
    # for each of the encoder blocks:
    for i in range(config.num_encoder_blocks):
        for j in range(config.depths[i]):
            # read in weights + bias of keys and values (which is a single matrix in the original implementation)
            kv_weight = state_dict.pop(f"segformer.encoder.block.{i}.{j}.attention.self.kv.weight")
            kv_bias = state_dict.pop(f"segformer.encoder.block.{i}.{j}.attention.self.kv.bias")
            # next, add keys and values (in that order) to the state dict
            state_dict[f"segformer.encoder.block.{i}.{j}.attention.self.key.weight"] = kv_weight[
                : config.hidden_sizes[i], :
            ]
            state_dict[f"segformer.encoder.block.{i}.{j}.attention.self.key.bias"] = kv_bias[: config.hidden_sizes[i]]
            state_dict[f"segformer.encoder.block.{i}.{j}.attention.self.value.weight"] = kv_weight[
                config.hidden_sizes[i] :, :
            ]
            state_dict[f"segformer.encoder.block.{i}.{j}.attention.self.value.bias"] = kv_bias[
                config.hidden_sizes[i] :
            ]


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    return im


@torch.no_grad()
def convert_segformer_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our SegFormer structure.
    """

    # load default config
    config = SegFormerConfig()
    # set config attributes based on name of the model
    if "ade" in model_name:
        config.num_labels = 150
        # TODO id2label
        # config.id2label = id2label
        # config.label2id = {v: k for k, v in id2label.items()}
    elif "city" in model_name:
        config.num_labels = 19
        # TODO id2label
        # config.id2label = id2label
        # config.label2id = {v: k for k, v in id2label.items()}

    # load feature extractor
    feature_extractor = SegFormerFeatureExtractor()

    # prepare image
    img = prepare_img()
    encoding = feature_extractor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    logger.info(f"Converting model {model_name}...")

    # load original state dict
    state_dict = torch.load(checkpoint_path)["state_dict"]

    # rename keys
    state_dict = rename_keys(state_dict)
    del state_dict['conv_seg.weight']
    del state_dict['conv_seg.bias']

    # key and value matrices need special treatment
    read_in_k_v(state_dict, config)

    # create HuggingFace model and load state dict
    model = SegFormerForImageSegmentation(config)
    model.load_state_dict(state_dict)
    model.eval()

    # finally, save model and feature extractor
    logger.info(f"Saving PyTorch model and feature extractor to {pytorch_dump_folder_path}...")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="segformer.b0.512x512.ade.160k",
        type=str,
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Path to the original PyTorch checkpoint (.pth file)."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    args = parser.parse_args()
    convert_segformer_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path)
