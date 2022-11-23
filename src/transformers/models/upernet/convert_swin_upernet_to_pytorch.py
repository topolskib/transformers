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
"""Convert Swin Transformer + UperNet checkpoints from mmsegmentation."""

import argparse

import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

import requests
from transformers import SwinConfig, UperNetConfig, UperNetForSemanticSegmentation
from transformers.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def get_upernet_config(model_name):
    backbone_config = SwinConfig(out_features=["stage1", "stage2", "stage3", "stage4"])
    config = UperNetConfig(backbone_config=backbone_config, num_labels=150)

    return config


# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    rename_keys = []

    # fmt: off
    # stem
    rename_keys.append(("backbone.downsample_layers.0.0.weight", "backbone.swin.embeddings.patch_embeddings.weight"))
    rename_keys.append(("backbone.downsample_layers.0.0.bias", "backbone.swin.embeddings.patch_embeddings.bias"))
    rename_keys.append(("backbone.downsample_layers.0.1.weight", "backbone.swin.embeddings.layernorm.weight"))
    rename_keys.append(("backbone.downsample_layers.0.1.bias", "backbone.swin.embeddings.layernorm.bias"))
    # stages
    for i in range(len(config.backbone_config.depths)):
        for j in range(config.backbone_config.depths[i]):
            rename_keys.append((f"backbone.stages.{i}.{j}.gamma", f"backbone.swin.encoder.stages.{i}.layers.{j}.layer_scale_parameter"))
            rename_keys.append((f"backbone.stages.{i}.{j}.depthwise_conv.weight", f"backbone.swin.encoder.stages.{i}.layers.{j}.dwconv.weight"))
            rename_keys.append((f"backbone.stages.{i}.{j}.depthwise_conv.bias", f"backbone.swin.encoder.stages.{i}.layers.{j}.dwconv.bias"))
            rename_keys.append((f"backbone.stages.{i}.{j}.norm.weight", f"backbone.swin.encoder.stages.{i}.layers.{j}.layernorm.weight"))
            rename_keys.append((f"backbone.stages.{i}.{j}.norm.bias", f"backbone.swin.encoder.stages.{i}.layers.{j}.layernorm.bias"))
            rename_keys.append((f"backbone.stages.{i}.{j}.pointwise_conv1.weight", f"backbone.swin.encoder.stages.{i}.layers.{j}.pwconv1.weight"))
            rename_keys.append((f"backbone.stages.{i}.{j}.pointwise_conv1.bias", f"backbone.swin.encoder.stages.{i}.layers.{j}.pwconv1.bias"))
            rename_keys.append((f"backbone.stages.{i}.{j}.pointwise_conv2.weight", f"backbone.swin.encoder.stages.{i}.layers.{j}.pwconv2.weight"))
            rename_keys.append((f"backbone.stages.{i}.{j}.pointwise_conv2.bias", f"backbone.swin.encoder.stages.{i}.layers.{j}.pwconv2.bias"))
        if i > 0:
            rename_keys.append((f"backbone.downsample_layers.{i}.0.weight", f"backbone.swin.encoder.stages.{i}.downsampling_layer.0.weight"))
            rename_keys.append((f"backbone.downsample_layers.{i}.0.bias", f"backbone.swin.encoder.stages.{i}.downsampling_layer.0.bias"))
            rename_keys.append((f"backbone.downsample_layers.{i}.1.weight", f"backbone.swin.encoder.stages.{i}.downsampling_layer.1.weight"))
            rename_keys.append((f"backbone.downsample_layers.{i}.1.bias", f"backbone.swin.encoder.stages.{i}.downsampling_layer.1.bias"))

        rename_keys.append((f"backbone.norm{i}.weight", f"backbone.hidden_states_norms.{i}.weight"))
        rename_keys.append((f"backbone.norm{i}.bias", f"backbone.hidden_states_norms.{i}.bias"))

    # decode head
    rename_keys.extend(
        [
            ("decode_head.conv_seg.weight", "decode_head.classifier.weight"),
            ("decode_head.conv_seg.bias", "decode_head.classifier.bias"),
            ("auxiliary_head.conv_seg.weight", "auxiliary_head.classifier.weight"),
            ("auxiliary_head.conv_seg.bias", "auxiliary_head.classifier.bias"),
        ]
    )
    # fmt: on

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


image_transforms = Compose(
    [Resize((512, 512)), ToTensor(), Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)]
)


def convert_upernet_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    model_name_to_url = {
        "upernet-swin-tiny": "https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth",
    }
    checkpoint_url = model_name_to_url[model_name]
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["state_dict"]

    config = get_upernet_config(model_name)
    model = UperNetForSemanticSegmentation(config)
    model.eval()

    # replace "bn" => "batch_norm"
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        if "bn" in key:
            key = key.replace("bn", "batch_norm")
        state_dict[key] = val

    # rename keys
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)

    model.load_state_dict(state_dict)

    # verify on image
    url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ade20k/resolve/main/ADE_val_00000001.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    pixel_values = image_transforms(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(pixel_values)

    expected_slice = torch.tensor(
        [[-8.8110, -7.5399, -7.5429], [-8.5200, -7.0736, -7.2054], [-8.5220, -7.2897, -7.3901]]
    )
    print("Logits:", outputs.logits[0, 0, :3, :3])
    assert torch.allclose(outputs.logits[0, 0, :3, :3], expected_slice, atol=1e-4)
    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        # print(f"Saving feature extractor to {pytorch_dump_folder_path}")
        # feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print(f"Pushing model and feature extractor for {model_name} to hub")
        model.push_to_hub(f"nielsr/{model_name}")
        # feature_extractor.push_to_hub(f"nielsr/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="upernet-swin-tiny",
        type=str,
        choices=["upernet-swin-tiny", "upernet-swin-small"],
        help="Name of the Swin + UperNet model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_upernet_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
