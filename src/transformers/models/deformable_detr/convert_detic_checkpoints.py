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
"""Convert Deformable DETR Detic checkpoints.

URL: https://github.com/facebookresearch/Detic
"""


import argparse
import json
from pathlib import Path

import requests
import torch
from huggingface_hub import cached_download, hf_hub_url
from PIL import Image

from transformers import (
    DeformableDetrConfig,
    DeformableDetrForObjectDetection,
    DeformableDetrImageProcessor,
    ResNetConfig,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_config():
    backbone_config = ResNetConfig()
    config = DeformableDetrConfig(
        use_timm_backbone=False, backbone_config=backbone_config, with_box_refine=True, two_stage=True
    )

    # set labels
    config.num_labels = 91
    repo_id = "huggingface/label-files"
    filename = "coco-detection-id2label.json"
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config


def create_rename_keys(config):
    # here we list all keys to be renamed (original name on the left, our name on the right)
    rename_keys = []

    # stem
    # fmt: off
    rename_keys.append(("detr.backbone.0.backbone.stem.conv1.weight", "model.backbone.conv_encoder.model.embedder.embedder.convolution.weight"))
    rename_keys.append(("detr.backbone.0.backbone.stem.conv1.norm.weight", "model.backbone.conv_encoder.model.embedder.embedder.normalization.weight"))
    rename_keys.append(("detr.backbone.0.backbone.stem.conv1.norm.bias", "model.backbone.conv_encoder.model.embedder.embedder.normalization.bias"))
    rename_keys.append(("detr.backbone.0.backbone.stem.conv1.norm.running_mean", "model.backbone.conv_encoder.model.embedder.embedder.normalization.running_mean"))
    rename_keys.append(("detr.backbone.0.backbone.stem.conv1.norm.running_var", "model.backbone.conv_encoder.model.embedder.embedder.normalization.running_var"))

    # stages
    for stage_idx in range(len(config.backbone_config.depths)):
        for layer_idx in range(config.backbone_config.depths[stage_idx]):
            # shortcut
            if layer_idx == 0:
                rename_keys.append(
                    (
                        f"detr.backbone.0.backbone.res{stage_idx+2}.{layer_idx}.shortcut.weight",
                        f"model.backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.convolution.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"detr.backbone.0.backbone.res{stage_idx+2}.{layer_idx}.shortcut.norm.weight",
                        f"model.backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"detr.backbone.0.backbone.res{stage_idx+2}.{layer_idx}.shortcut.norm.bias",
                        f"model.backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.bias",
                    )
                )
                rename_keys.append(
                    (
                        f"detr.backbone.0.backbone.res{stage_idx+2}.{layer_idx}.shortcut.norm.running_mean",
                        f"model.backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.running_mean",
                    )
                )
                rename_keys.append(
                    (
                        f"detr.backbone.0.backbone.res{stage_idx+2}.{layer_idx}.shortcut.norm.running_var",
                        f"model.backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.running_var",
                    )
                )
            # 3 convs
            for i in range(3):
                rename_keys.append(
                    (
                        f"detr.backbone.0.backbone.res{stage_idx+2}.{layer_idx}.conv{i+1}.weight",
                        f"model.backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.convolution.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"detr.backbone.0.backbone.res{stage_idx+2}.{layer_idx}.conv{i+1}.norm.weight",
                        f"model.backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.normalization.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"detr.backbone.0.backbone.res{stage_idx+2}.{layer_idx}.conv{i+1}.norm.bias",
                        f"model.backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.normalization.bias",
                    )
                )
                rename_keys.append(
                    (
                        f"detr.backbone.0.backbone.res{stage_idx+2}.{layer_idx}.conv{i+1}.norm.running_mean",
                        f"model.backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.normalization.running_mean",
                    )
                )
                rename_keys.append(
                    (
                        f"detr.backbone.0.backbone.res{stage_idx+2}.{layer_idx}.conv{i+1}.norm.running_var",
                        f"model.backbone.conv_encoder.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.normalization.running_var",
                    )
                )
    
    # input projection layers
    rename_keys.append(("detr.input_proj.0.0.weight", "model.input_proj.0.0.weight"))
    rename_keys.append(("detr.input_proj.0.0.bias", "model.input_proj.0.0.bias"))
    rename_keys.append(("detr.input_proj.0.1.weight", "model.input_proj.0.1.weight"))
    rename_keys.append(("detr.input_proj.0.1.bias", "model.input_proj.0.1.bias"))
    rename_keys.append(("detr.input_proj.1.0.weight", "model.input_proj.1.0.weight"))
    rename_keys.append(("detr.input_proj.1.0.bias", "model.input_proj.1.0.bias"))
    rename_keys.append(("detr.input_proj.1.1.weight", "model.input_proj.1.1.weight"))
    rename_keys.append(("detr.input_proj.1.1.bias", "model.input_proj.1.1.bias"))
    rename_keys.append(("detr.input_proj.2.0.weight", "model.input_proj.2.0.weight"))
    rename_keys.append(("detr.input_proj.2.0.bias", "model.input_proj.2.0.bias"))
    rename_keys.append(("detr.input_proj.2.1.weight", "model.input_proj.2.1.weight"))
    rename_keys.append(("detr.input_proj.2.1.bias", "model.input_proj.2.1.bias"))
    rename_keys.append(("detr.input_proj.3.0.weight", "model.input_proj.3.0.weight"))
    rename_keys.append(("detr.input_proj.3.0.bias", "model.input_proj.3.0.bias"))
    rename_keys.append(("detr.input_proj.3.1.weight", "model.input_proj.3.1.weight"))
    rename_keys.append(("detr.input_proj.3.1.bias", "model.input_proj.3.1.bias"))

    # Transformer encoder
    for layer_idx in range(config.encoder_layers):
        rename_keys.append((f"detr.transformer.encoder.layers.{layer_idx}.self_attn.sampling_offsets.weight", f"model.encoder.layers.{layer_idx}.self_attn.sampling_offsets.weight"))
        rename_keys.append((f"detr.transformer.encoder.layers.{layer_idx}.self_attn.sampling_offsets.bias", f"model.encoder.layers.{layer_idx}.self_attn.sampling_offsets.bias"))
        rename_keys.append((f"detr.transformer.encoder.layers.{layer_idx}.self_attn.attention_weights.weight", f"model.encoder.layers.{layer_idx}.self_attn.attention_weights.weight"))
        rename_keys.append((f"detr.transformer.encoder.layers.{layer_idx}.self_attn.attention_weights.bias", f"model.encoder.layers.{layer_idx}.self_attn.attention_weights.bias"))
        rename_keys.append((f"detr.transformer.encoder.layers.{layer_idx}.self_attn.value_proj.weight", f"model.encoder.layers.{layer_idx}.self_attn.value_proj.weight"))
        rename_keys.append((f"detr.transformer.encoder.layers.{layer_idx}.self_attn.value_proj.bias", f"model.encoder.layers.{layer_idx}.self_attn.value_proj.bias"))
        rename_keys.append((f"detr.transformer.encoder.layers.{layer_idx}.self_attn.output_proj.weight", f"model.encoder.layers.{layer_idx}.self_attn.output_proj.weight"))
        rename_keys.append((f"detr.transformer.encoder.layers.{layer_idx}.self_attn.output_proj.bias", f"model.encoder.layers.{layer_idx}.self_attn.output_proj.bias"))
        # TODO check whether norm1 is self attention layer norm
        rename_keys.append((f"detr.transformer.encoder.layers.{layer_idx}.norm1.weight", f"model.encoder.layers.{layer_idx}.self_attn_layer_norm.weight"))
        rename_keys.append((f"detr.transformer.encoder.layers.{layer_idx}.norm1.bias", f"model.encoder.layers.{layer_idx}.self_attn_layer_norm.bias"))
        rename_keys.append((f"detr.transformer.encoder.layers.{layer_idx}.linear1.weight", f"model.encoder.layers.{layer_idx}.fc1.weight"))
        rename_keys.append((f"detr.transformer.encoder.layers.{layer_idx}.linear1.bias", f"model.encoder.layers.{layer_idx}.fc1.bias"))
        rename_keys.append((f"detr.transformer.encoder.layers.{layer_idx}.linear2.weight", f"model.encoder.layers.{layer_idx}.fc2.weight"))
        rename_keys.append((f"detr.transformer.encoder.layers.{layer_idx}.linear2.bias", f"model.encoder.layers.{layer_idx}.fc2.bias"))
        rename_keys.append((f"detr.transformer.encoder.layers.{layer_idx}.norm2.weight", f"model.encoder.layers.{layer_idx}.final_layer_norm.weight"))
        rename_keys.append((f"detr.transformer.encoder.layers.{layer_idx}.norm2.bias", f"model.encoder.layers.{layer_idx}.final_layer_norm.bias"))

    # Transformer decoder
    for layer_idx in range(config.decoder_layers):
        rename_keys.append((f"detr.transformer.decoder.layers.{layer_idx}.cross_attn.sampling_offsets.weight", f"model.decoder.layers.{layer_idx}.encoder_attn.sampling_offsets.weight"))
        rename_keys.append((f"detr.transformer.decoder.layers.{layer_idx}.cross_attn.sampling_offsets.bias", f"model.decoder.layers.{layer_idx}.encoder_attn.sampling_offsets.bias"))
        rename_keys.append((f"detr.transformer.decoder.layers.{layer_idx}.cross_attn.attention_weights.weight", f"model.decoder.layers.{layer_idx}.encoder_attn.attention_weights.weight"))
        rename_keys.append((f"detr.transformer.decoder.layers.{layer_idx}.cross_attn.attention_weights.bias", f"model.decoder.layers.{layer_idx}.encoder_attn.attention_weights.bias"))
        rename_keys.append((f"detr.transformer.decoder.layers.{layer_idx}.cross_attn.value_proj.weight", f"model.decoder.layers.{layer_idx}.encoder_attn.value_proj.weight"))
        rename_keys.append((f"detr.transformer.decoder.layers.{layer_idx}.cross_attn.value_proj.bias", f"model.decoder.layers.{layer_idx}.encoder_attn.value_proj.bias"))
        rename_keys.append((f"detr.transformer.decoder.layers.{layer_idx}.cross_attn.output_proj.weight", f"model.decoder.layers.{layer_idx}.encoder_attn.output_proj.weight"))
        rename_keys.append((f"detr.transformer.decoder.layers.{layer_idx}.cross_attn.output_proj.bias", f"model.decoder.layers.{layer_idx}.encoder_attn.output_proj.bias"))
        # TODO check layernorms
        rename_keys.append((f"detr.transformer.decoder.layers.{layer_idx}.norm1.weight", f"model.decoder.layers.{layer_idx}.encoder_attn_layer_norm.weight"))
        rename_keys.append((f"detr.transformer.decoder.layers.{layer_idx}.norm1.bias", f"model.decoder.layers.{layer_idx}.encoder_attn_layer_norm.bias"))
        rename_keys.append((f"detr.transformer.decoder.layers.{layer_idx}.self_attn.out_proj.weight", f"model.decoder.layers.{layer_idx}.self_attn.out_proj.weight"))
        rename_keys.append((f"detr.transformer.decoder.layers.{layer_idx}.self_attn.out_proj.bias", f"model.decoder.layers.{layer_idx}.self_attn.out_proj.bias"))
        rename_keys.append((f"detr.transformer.decoder.layers.{layer_idx}.norm2.weight", f"model.decoder.layers.{layer_idx}.self_attn_layer_norm.weight"))
        rename_keys.append((f"detr.transformer.decoder.layers.{layer_idx}.norm2.bias", f"model.decoder.layers.{layer_idx}.self_attn_layer_norm.bias"))
        rename_keys.append((f"detr.transformer.decoder.layers.{layer_idx}.linear1.weight", f"model.decoder.layers.{layer_idx}.fc1.weight"))
        rename_keys.append((f"detr.transformer.decoder.layers.{layer_idx}.linear1.bias", f"model.decoder.layers.{layer_idx}.fc1.bias"))
        rename_keys.append((f"detr.transformer.decoder.layers.{layer_idx}.linear2.weight", f"model.decoder.layers.{layer_idx}.fc2.weight"))
        rename_keys.append((f"detr.transformer.decoder.layers.{layer_idx}.linear2.bias", f"model.decoder.layers.{layer_idx}.fc2.bias"))
        rename_keys.append((f"detr.transformer.decoder.layers.{layer_idx}.norm3.weight", f"model.decoder.layers.{layer_idx}.final_layer_norm.weight"))
        rename_keys.append((f"detr.transformer.decoder.layers.{layer_idx}.norm3.bias", f"model.decoder.layers.{layer_idx}.final_layer_norm.bias"))
    
    # fmt: on
    
    return rename_keys


def rename_key(state_dict, old, new):
    val = state_dict.pop(old)
    state_dict[new] = val


def read_in_q_k_v(state_dict, config):
    # transformer decoder self-attention layers
    for layer_idx in range(config.decoder_layers):
        # read in weights + bias of input projection layer of self-attention
        in_proj_weight = state_dict.pop(f"detr.transformer.decoder.layers.{layer_idx}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"detr.transformer.decoder.layers.{layer_idx}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.decoder.layers.{layer_idx}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"model.decoder.layers.{layer_idx}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"model.decoder.layers.{layer_idx}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"model.decoder.layers.{layer_idx}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"model.decoder.layers.{layer_idx}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"model.decoder.layers.{layer_idx}.self_attn.v_proj.bias"] = in_proj_bias[-256:]


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    return im


@torch.no_grad()
def convert_deformable_detr_checkpoint(
    model_name,
    single_scale,
    dilation,
    with_box_refine,
    two_stage,
    pytorch_dump_folder_path,
    push_to_hub,
):
    """
    Copy/paste/tweak model's weights to our Deformable DETR structure.
    """

    # load config
    config = get_config()

    # load image processor
    processor = DeformableDetrImageProcessor(format="coco_detection")

    # prepare image
    img = prepare_img()
    encoding = processor(images=img, return_tensors="pt")
    encoding["pixel_values"]

    logger.info("Converting model...")

    # load original state dict
    url = "https://dl.fbaipublicfiles.com/detic/Detic_DeformDETR_LI_R50_4x_ft4x.pth"
    state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")["model"]

    # rename keys
    for src, dest in create_rename_keys(config):
        rename_key(state_dict, src, dest)
    
    # query, key and value matrices of decoder need special treatment
    read_in_q_k_v(state_dict, config)
    
    # important: we need to prepend a prefix to each of the base model keys as the head models use different attributes for them
    # prefix = "model."
    # for key in state_dict.copy().keys():
    #     if not key.startswith("class_embed") and not key.startswith("bbox_embed"):
    #         val = state_dict.pop(key)
    #         state_dict[prefix + key] = val
    # finally, create HuggingFace model and load state dict
    model = DeformableDetrForObjectDetection(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # TODO verify our conversion
    # outputs = model(pixel_values.to(device))

    # expected_logits = torch.tensor(
    #     [[-9.6645, -4.3449, -5.8705], [-9.7035, -3.8504, -5.0724], [-10.5634, -5.3379, -7.5116]]
    # )
    # expected_boxes = torch.tensor([[0.8693, 0.2289, 0.2492], [0.3150, 0.5489, 0.5845], [0.5563, 0.7580, 0.8518]])

    # if with_box_refine and two_stage:
    #     expected_logits = torch.tensor(
    #         [[-6.7108, -4.3213, -6.3777], [-8.9014, -6.1799, -6.7240], [-6.9315, -4.4735, -6.2298]]
    #     )
    #     expected_boxes = torch.tensor([[0.2583, 0.5499, 0.4683], [0.7652, 0.9068, 0.4882], [0.5490, 0.2763, 0.0564]])

    # print("Logits:", outputs.logits[0, :3, :3])

    # assert torch.allclose(outputs.logits[0, :3, :3], expected_logits.to(device), atol=1e-4)
    # assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes.to(device), atol=1e-4)

    # print("Everything ok!")

    if pytorch_dump_folder_path is not None:
        # Save model and image processor
        logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        # Push to hub
        model_name = "deformable-detr"
        model_name += "-single-scale" if single_scale else ""
        model_name += "-dc5" if dilation else ""
        model_name += "-with-box-refine" if with_box_refine else ""
        model_name += "-two-stage" if two_stage else ""
        print("Pushing model to hub...")
        model.push_to_hub(repo_path_or_name=model_name, organization="nielsr", commit_message="Add model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/niels/checkpoints/deformable_detr/r50_deformable_detr-checkpoint.pth",
        help="Path to Pytorch checkpoint (.pth file) you'd like to convert.",
    )
    parser.add_argument("--single_scale", action="store_true", help="Whether to set config.num_features_levels = 1.")
    parser.add_argument("--dilation", action="store_true", help="Whether to set config.dilation=True.")
    parser.add_argument("--with_box_refine", action="store_false", help="Whether to set config.with_box_refine=True.")
    parser.add_argument("--two_stage", action="store_false", help="Whether to set config.two_stage=True.")
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the folder to output PyTorch model.",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    args = parser.parse_args()
    convert_deformable_detr_checkpoint(
        args.checkpoint_path,
        args.single_scale,
        args.dilation,
        args.with_box_refine,
        args.two_stage,
        args.pytorch_dump_folder_path,
        args.push_to_hub,
    )
