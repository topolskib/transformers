# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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

# here we list all keys to be renamed (original name on the left, our name on the right)
rename_keys = []


def rename_key(state_dict, old, new):
    val = state_dict.pop(old)
    state_dict[new] = val


def rename_backbone_keys(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if "backbone.0.body" in key:
            new_key = key.replace("backbone.0.body", "backbone.conv_encoder.model")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict


def read_in_q_k_v(state_dict):
    for i in range(6):
        # read in weights + bias of input projection layer (in PyTorch's MultiHeadAttention, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]


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
        #config.id2label = id2label
        #config.label2id = {v: k for k, v in id2label.items()}
    elif "city" in model_name:
        config.num_labels = 19
        # TODO id2label
        #config.id2label = id2label
        #config.label2id = {v: k for k, v in id2label.items()}

    # load feature extractor
    feature_extractor = SegFormerFeatureExtractor()

    # prepare image
    img = prepare_img()
    encoding = feature_extractor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    logger.info(f"Converting model {model_name}...")

    # load original state dict
    state_dict = torch.load(checkpoint_path)['state_dict']
    
    # rename keys
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    state_dict = rename_backbone_keys(state_dict)
    
    # query, key and value matrices need special treatment
    read_in_q_k_v(state_dict, is_panoptic=is_panoptic)
    
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
        "--model_name", default=None, type=str, help="Name of the model you'd like to convert."
    )
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Path to the folder containing an original checkpoint."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    args = parser.parse_args()
    convert_detr_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path)