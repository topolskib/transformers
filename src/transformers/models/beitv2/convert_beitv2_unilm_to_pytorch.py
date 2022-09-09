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
"""Convert BEiTv2 checkpoints from the unilm repository."""


import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from PIL import Image

import requests
from huggingface_hub import hf_hub_download
from transformers import (
    BeitFeatureExtractor,
    Beitv2Config,
    Beitv2ForImageClassification,
    Beitv2ForMaskedImageModeling,
    Beitv2ForSemanticSegmentation,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config, has_lm_head=False, is_semantic=False):
    prefix = "backbone." if is_semantic else ""

    rename_keys = []
    for i in range(config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append((f"{prefix}blocks.{i}.norm1.weight", f"beitv2.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.norm1.bias", f"beitv2.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append(
            (f"{prefix}blocks.{i}.attn.proj.weight", f"beitv2.encoder.layer.{i}.attention.output.dense.weight")
        )
        rename_keys.append(
            (f"{prefix}blocks.{i}.attn.proj.bias", f"beitv2.encoder.layer.{i}.attention.output.dense.bias")
        )
        rename_keys.append((f"{prefix}blocks.{i}.norm2.weight", f"beitv2.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.norm2.bias", f"beitv2.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append(
            (f"{prefix}blocks.{i}.mlp.fc1.weight", f"beitv2.encoder.layer.{i}.intermediate.dense.weight")
        )
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc1.bias", f"beitv2.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc2.weight", f"beitv2.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc2.bias", f"beitv2.encoder.layer.{i}.output.dense.bias"))

    # projection layer + position embeddings
    rename_keys.extend(
        [
            (f"{prefix}cls_token", "beitv2.embeddings.cls_token"),
            (f"{prefix}patch_embed.proj.weight", "beitv2.embeddings.patch_embeddings.projection.weight"),
            (f"{prefix}patch_embed.proj.bias", "beitv2.embeddings.patch_embeddings.projection.bias"),
        ]
    )

    if has_lm_head:
        # mask token + shared relative position bias + layernorm
        rename_keys.extend(
            [
                ("mask_token", "beitv2.embeddings.mask_token"),
                (
                    "rel_pos_bias.relative_position_bias_table",
                    "beitv2.encoder.relative_position_bias.relative_position_bias_table",
                ),
                (
                    "rel_pos_bias.relative_position_index",
                    "beitv2.encoder.relative_position_bias.relative_position_index",
                ),
                ("norm.weight", "layernorm.weight"),
                ("norm.bias", "layernorm.bias"),
            ]
        )
    elif is_semantic:
        # semantic segmentation classification heads
        rename_keys.extend(
            [
                ("decode_head.conv_seg.weight", "decode_head.classifier.weight"),
                ("decode_head.conv_seg.bias", "decode_head.classifier.bias"),
                ("auxiliary_head.conv_seg.weight", "auxiliary_head.classifier.weight"),
                ("auxiliary_head.conv_seg.bias", "auxiliary_head.classifier.bias"),
            ]
        )
    else:
        # layernorm + classification head
        rename_keys.extend(
            [
                ("fc_norm.weight", "beitv2.pooler.layernorm.weight"),
                ("fc_norm.bias", "beitv2.pooler.layernorm.bias"),
                ("head.weight", "classifier.weight"),
                ("head.bias", "classifier.bias"),
            ]
        )

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config, has_lm_head=False, is_semantic=False):
    for i in range(config.num_hidden_layers):
        prefix = "backbone." if is_semantic else ""
        # queries, keys and values
        in_proj_weight = state_dict.pop(f"{prefix}blocks.{i}.attn.qkv.weight")
        q_bias = state_dict.pop(f"{prefix}blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"{prefix}blocks.{i}.attn.v_bias")

        state_dict[f"beitv2.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"beitv2.encoder.layer.{i}.attention.attention.query.bias"] = q_bias
        state_dict[f"beitv2.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"beitv2.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"beitv2.encoder.layer.{i}.attention.attention.value.bias"] = v_bias

        # gamma_1 and gamma_2
        # we call them lambda because otherwise they are renamed when using .from_pretrained
        gamma_1 = state_dict.pop(f"{prefix}blocks.{i}.gamma_1")
        gamma_2 = state_dict.pop(f"{prefix}blocks.{i}.gamma_2")

        state_dict[f"beitv2.encoder.layer.{i}.lambda_1"] = gamma_1
        state_dict[f"beitv2.encoder.layer.{i}.lambda_2"] = gamma_2

        # relative_position bias table + index
        if not has_lm_head:
            # each layer has its own relative position bias
            table = state_dict.pop(f"{prefix}blocks.{i}.attn.relative_position_bias_table")
            index = state_dict.pop(f"{prefix}blocks.{i}.attn.relative_position_index")

            state_dict[
                f"beitv2.encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_bias_table"
            ] = table
            state_dict[
                f"beitv2.encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_index"
            ] = index


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_beitv2_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our BEiTv2 structure.
    """

    model_to_url = {
        "beitv2-base-patch16-224": (
            "https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k.pth"
        ),
        "beitv2-large-patch16-224": (
            "https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k.pth"
        ),
        "beitv2-base-patch16-224-pt1k_ft21k": "https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_base_patch16_224_pt1k_ft21k.pth",
        "beitv2-large-patch16-224-pt1k_ft21k": "https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/beitv2_large_patch16_224_pt1k_ft21k.pth",
    }

    checkpoint_url = model_to_url[model_name]

    # define default BEiTv2 configuration
    config = Beitv2Config()
    has_lm_head = False
    is_semantic = False
    repo_id = "datasets/huggingface/label-files"
    # set config parameters based on URL
    if checkpoint_url[-9:-4] == "pt22k":
        # masked image modeling
        config.use_shared_relative_position_bias = True
        config.use_mask_token = True
        has_lm_head = True
    elif checkpoint_url[-9:-4] == "ft22k":
        # intermediate fine-tuning on ImageNet-22k
        config.use_relative_position_bias = True
        config.num_labels = 21841
        filename = "imagenet-22k-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        # this dataset contains 21843 labels but the model only has 21841
        # we delete the classes as mentioned in https://github.com/google-research/big_transfer/issues/18
        del id2label[9205]
        del id2label[15027]
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
    elif model_name in ["beitv2-base-patch16-224", "beitv2-large-patch16-224"]:
        # fine-tuning on ImageNet-1k
        config.use_relative_position_bias = True
        config.num_labels = 1000
        filename = "imagenet-1k-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
        if "384" in checkpoint_url:
            config.image_size = 384
        if "512" in checkpoint_url:
            config.image_size = 512
    elif "ade20k" in checkpoint_url:
        # fine-tuning
        config.use_relative_position_bias = True
        config.num_labels = 150
        filename = "ade20k-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
        config.image_size = 640
        is_semantic = True
    else:
        raise ValueError("Checkpoint not supported, URL should either end with 'pt22k', 'ft22k', 'to1k' or 'ade20k'")

    # size of the architecture
    if "base" in model_name:
        pass
    elif "large" in model_name:
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        if "ade20k" in model_name:
            config.image_size = 640
            config.out_indices = [7, 11, 15, 23]
    else:
        raise ValueError("Should either find 'base' or 'large' in model name")

    # load state_dict of original model, remove and rename some keys
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu", check_hash=True)
    state_dict = state_dict["model"] if "ade20k" not in checkpoint_url else state_dict["state_dict"]

    rename_keys = create_rename_keys(config, has_lm_head=has_lm_head, is_semantic=is_semantic)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config, has_lm_head=has_lm_head, is_semantic=is_semantic)
    if is_semantic:
        # add prefix to decoder keys
        for key, val in state_dict.copy().items():
            val = state_dict.pop(key)
            if key.startswith("backbone.fpn"):
                key = key.replace("backbone.fpn", "fpn")
            state_dict[key] = val

    # load HuggingFace model
    if checkpoint_url[-9:-4] == "pt22k":
        model = Beitv2ForMaskedImageModeling(config)
    elif "ade20k" in checkpoint_url:
        model = Beitv2ForSemanticSegmentation(config)
    else:
        model = Beitv2ForImageClassification(config)
    model.eval()
    model.load_state_dict(state_dict)

    # Check outputs on an image
    if is_semantic:
        feature_extractor = BeitFeatureExtractor(size=config.image_size, do_center_crop=False)
        ds = load_dataset("hf-internal-testing/fixtures_ade20k", split="test")
        image = Image.open(ds[0]["file"])
    else:
        feature_extractor = BeitFeatureExtractor(size=config.image_size, resample=Image.BILINEAR, do_center_crop=False)
        image = prepare_img()

    encoding = feature_extractor(images=image, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    outputs = model(pixel_values)
    logits = outputs.logits

    # verify logits
    expected_shape = torch.Size([1, 1000])
    if model_name == "":
        expected_shape = torch.Size([1, 196, 8192])
    else:
        raise ValueError("Can't verify logits as model is not supported")

    assert logits.shape == expected_shape, "Shape of logits not as expected"
    if not has_lm_head:
        raise NotImplementedError("To do")
        # if is_semantic:
        #     assert torch.allclose(
        #         logits[0, :3, :3, :3], expected_logits, atol=1e-3
        #     ), "First elements of logits not as expected"
        # else:
        #     print("Predicted class idx:", logits.argmax(-1).item())
        #     assert torch.allclose(
        #         logits[0, :3], expected_logits, atol=1e-3
        #     ), "First elements of logits not as expected"
        #     assert logits.argmax(-1).item() == expected_class_idx, "Predicted class index not as expected"

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving feature extractor to {pytorch_dump_folder_path}")
        feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model.push_to_hub("nielsr/test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="beitv2-base-patch16-224",
        type=str,
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_beitv2_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
