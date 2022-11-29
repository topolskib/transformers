# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert ResNetv2 checkpoints from the timm library."""


import argparse
from pathlib import Path

import torch
from PIL import Image

import requests
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import ResNetv2Config, ResNetv2ForImageClassification
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)



def rename_key(name):
    if "stem.conv" in name:
        name = name.replace("stem.conv", "resnetv2.embedder.convolution")
    if "blocks" in name:
        name = name.replace("blocks", "layers")
    if "head.fc" in name:
        name = name.replace("head.fc", "classifier.1")
    if name.startswith("norm"):
        name = "resnetv2." + name
    if "resnetv2" not in name and "classifier" not in name:
        name = "resnetv2.encoder." + name

    return name


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_resnetv2_checkpoint(model_name, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our ResNetv2 structure.
    """

    # define default ResNetv2 configuration
    config = ResNetv2Config(num_labels=1000)

    # load original model from timm
    timm_model = create_model(model_name, pretrained=True)
    timm_model.eval()

    # load state_dict of original model
    state_dict = timm_model.state_dict()
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val.squeeze() if "head" in key else val

    # load HuggingFace model
    model = ResNetv2ForImageClassification(config)
    model.eval()
    model.load_state_dict(state_dict)

    # TODO verify logits
    transform = create_transform(**resolve_data_config({}, model=model))

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # weird bug: we don't get the same pixel values as in Colab
    # load pixel values from the hub for the moment
    # pixel_values = transform(image).unsqueeze(0)

    from huggingface_hub import hf_hub_download

    pixel_values = torch.load(
        hf_hub_download("nielsr/dummy-pixel-values", repo_type="dataset", filename="pixel_values.pt")
    )

    print("Shape of pixel values:", pixel_values.shape)
    print("First values of pixel values:", pixel_values[0, 0, :3, :3])

    outputs = model(pixel_values)
    logits = outputs.logits

    print("Predicted class:", logits.argmax(-1))

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        # print(f"Saving feature extractor to {pytorch_dump_folder_path}")
        # feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="resnetv2_50",
        type=str,
        help="Name of the ViT timm model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_resnetv2_checkpoint(args.model_name, args.pytorch_dump_folder_path)
