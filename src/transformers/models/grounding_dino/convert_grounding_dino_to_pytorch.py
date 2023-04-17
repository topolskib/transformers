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
"""Convert Grounding DINO checkpoints."""


import argparse
import json
from pathlib import Path

import requests
import torch
from huggingface_hub import cached_download, hf_hub_url
from PIL import Image

from transformers import GroundingDINOConfig, DeformableDetrImageProcessor, GroundingDINOForObjectDetection
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def rename_key(orig_key):
    if "backbone.0.body" in orig_key:
        orig_key = orig_key.replace("backbone.0.body", "backbone.conv_encoder.model")
    if "transformer" in orig_key:
        orig_key = orig_key.replace("transformer.", "")
    if "norm1" in orig_key:
        if "encoder" in orig_key:
            orig_key = orig_key.replace("norm1", "self_attn_layer_norm")
        else:
            orig_key = orig_key.replace("norm1", "encoder_attn_layer_norm")
    if "norm2" in orig_key:
        if "encoder" in orig_key:
            orig_key = orig_key.replace("norm2", "final_layer_norm")
        else:
            orig_key = orig_key.replace("norm2", "self_attn_layer_norm")
    if "norm3" in orig_key:
        orig_key = orig_key.replace("norm3", "final_layer_norm")
    if "linear1" in orig_key:
        orig_key = orig_key.replace("linear1", "fc1")
    if "linear2" in orig_key:
        orig_key = orig_key.replace("linear2", "fc2")
    if "query_embed" in orig_key:
        orig_key = orig_key.replace("query_embed", "query_position_embeddings")
    if "cross_attn" in orig_key:
        orig_key = orig_key.replace("cross_attn", "encoder_attn")

    return orig_key


def read_in_q_k_v(state_dict):
    # transformer decoder self-attention layers
    for i in range(6):
        # read in weights + bias of input projection layer of self-attention
        in_proj_weight = state_dict.pop(f"decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"decoder.layers.{i}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    return im


@torch.no_grad()
def convert_grounding_dino_checkpoint(checkpoint_path, model_name, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our Grounding DINO structure.
    """

    # load default config
    config = GroundingDINOConfig()

    # # load image processor
    # processor = DeformableDetrImageProcessor(format="coco_detection")

    # # prepare image
    # img = prepare_img()
    # encoding = processor(images=img, return_tensors="pt")
    # pixel_values = encoding["pixel_values"]

    # logger.info("Converting model...")

    # load original state dict
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

    for name, param in state_dict.items():
        print(name, param.shape)

    # # rename keys
    # for key in state_dict.copy().keys():
    #     val = state_dict.pop(key)
    #     state_dict[rename_key(key)] = val
    # # query, key and value matrices need special treatment
    # read_in_q_k_v(state_dict)
    # # important: we need to prepend a prefix to each of the base model keys as the head models use different attributes for them
    # prefix = "model."
    # for key in state_dict.copy().keys():
    #     if not key.startswith("class_embed") and not key.startswith("bbox_embed"):
    #         val = state_dict.pop(key)
    #         state_dict[prefix + key] = val
    # # finally, create HuggingFace model and load state dict
    # model = GroundingDINOForObjectDetection(config)
    # model.load_state_dict(state_dict)
    # model.eval()

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model.to(device)
    # # verify our conversion
    # outputs = model(pixel_values.to(device))

    # expected_logits = torch.tensor(
    #     [[-9.6645, -4.3449, -5.8705], [-9.7035, -3.8504, -5.0724], [-10.5634, -5.3379, -7.5116]]
    # )
    # expected_boxes = torch.tensor([[0.8693, 0.2289, 0.2492], [0.3150, 0.5489, 0.5845], [0.5563, 0.7580, 0.8518]])

    # print("Logits:", outputs.logits[0, :3, :3])

    # assert torch.allclose(outputs.logits[0, :3, :3], expected_logits.to(device), atol=1e-4)
    # assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes.to(device), atol=1e-4)

    # print("Everything ok!")

    # # Save model and image processor
    # if pytorch_dump_folder_path is not None:
    #     logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
    #     Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    #     model.save_pretrained(pytorch_dump_folder_path)
    #     processor.save_pretrained(pytorch_dump_folder_path)

    # # Push to hub
    # if push_to_hub:
    #     print("Pushing model to hub...")
    #     model.push_to_hub(f"nielsr/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/Users/nielsrogge/Documents/GroundingDINO/groundingdino_swint_ogc.pth",
        help="Path to Pytorch checkpoint (.pth file) you'd like to convert.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="grounding-dino-swin-tiny",
        help="Name of the model on the hub.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=False,
        help="Path to the folder to output PyTorch model.",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    args = parser.parse_args()
    convert_grounding_dino_checkpoint(
        args.checkpoint_path,
        args.model_name,
        args.pytorch_dump_folder_path,
        args.push_to_hub,
    )
