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
from huggingface_hub import hf_hub_download
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
    backbone_config = ResNetConfig(out_features=["stage2", "stage3", "stage4"])
    config = DeformableDetrConfig(
        use_timm_backbone=False,
        backbone_config=backbone_config,
        with_box_refine=True,
        two_stage=True,
    )

    # add labels
    repo_id = "huggingface/label-files"
    filename = "lvis-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
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

    # intermediate class and bbox embeddings
    for layer_idx in range(config.decoder_layers + 1):
        rename_keys.append((f"detr.transformer.decoder.class_embed.{layer_idx}.weight", f"model.decoder.class_embed.{layer_idx}.weight"))
        rename_keys.append((f"detr.transformer.decoder.class_embed.{layer_idx}.bias", f"model.decoder.class_embed.{layer_idx}.bias"))
        for i in range(3):
            rename_keys.append((f"detr.transformer.decoder.bbox_embed.{layer_idx}.layers.{i}.weight", f"model.decoder.bbox_embed.{layer_idx}.layers.{i}.weight"))
            rename_keys.append((f"detr.transformer.decoder.bbox_embed.{layer_idx}.layers.{i}.bias", f"model.decoder.bbox_embed.{layer_idx}.layers.{i}.bias"))

    # embeddings on top
    for layer_idx in range(config.decoder_layers + 1):
        rename_keys.append((f"detr.class_embed.{layer_idx}.weight", f"class_embed.{layer_idx}.weight"))
        rename_keys.append((f"detr.class_embed.{layer_idx}.bias", f"class_embed.{layer_idx}.bias"))
        for i in range(3):
            rename_keys.append((f"detr.bbox_embed.{layer_idx}.layers.{i}.weight", f"bbox_embed.{layer_idx}.layers.{i}.weight"))
            rename_keys.append((f"detr.bbox_embed.{layer_idx}.layers.{i}.bias", f"bbox_embed.{layer_idx}.layers.{i}.bias"))

    # remaining things
    rename_keys.append(("detr.transformer.level_embed", "model.level_embed"))
    rename_keys.append(("detr.transformer.enc_output.weight", "model.enc_output.weight"))
    rename_keys.append(("detr.transformer.enc_output.bias", "model.enc_output.bias"))
    rename_keys.append(("detr.transformer.enc_output_norm.weight", "model.enc_output_norm.weight"))
    rename_keys.append(("detr.transformer.enc_output_norm.bias", "model.enc_output_norm.bias"))
    rename_keys.append(("detr.transformer.pos_trans.weight", "model.pos_trans.weight"))
    rename_keys.append(("detr.transformer.pos_trans.bias", "model.pos_trans.bias"))
    rename_keys.append(("detr.transformer.pos_trans_norm.weight", "model.pos_trans_norm.weight"))
    rename_keys.append(("detr.transformer.pos_trans_norm.bias", "model.pos_trans_norm.bias"))

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
    image = Image.open(requests.get(url, stream=True).raw)

    return image


@torch.no_grad()
def convert_deformable_detr_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    """
    Copy/paste/tweak model's weights to our Deformable DETR structure.
    """

    # load config
    config = get_config()

    # load image processor
    processor = DeformableDetrImageProcessor(format="coco_detection")

    # prepare image
    image = prepare_img()
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # load original state dict
    name_to_url = {
        "deformable-detr-detic": "https://dl.fbaipublicfiles.com/detic/Detic_DeformDETR_LI_R50_4x_ft4x.pth",
        "deformable-detr-box-supervised": "https://dl.fbaipublicfiles.com/detic/BoxSup-DeformDETR_L_R50_4x.pth",
    }
    checkpoint_url = name_to_url[model_name]
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["model"]

    # rename keys
    for src, dest in create_rename_keys(config):
        rename_key(state_dict, src, dest)

    # query, key and value matrices of decoder need special treatment
    read_in_q_k_v(state_dict, config)

    # finally, create HuggingFace model and load state dict
    model = DeformableDetrForObjectDetection(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(missing_keys) == 0
    assert unexpected_keys == ["criterion.fed_loss_weight"]

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # verify our conversion
    outputs = model(pixel_values.to(device))

    if model_name == "deformable-detr-detic":
        expected_logits = torch.tensor(
            [[-8.4442, -6.9143, -4.8993], [-9.2303, -7.5461, -5.7119], [-6.3788, -6.6090, -6.2570]]
        )
        expected_boxes = torch.tensor([[0.5486, 0.2752, 0.0554], [0.1688, 0.1989, 0.2109], [0.1688, 0.1991, 0.2110]])
    elif model_name == "deformable-detr-box-supervised":
        expected_logits = torch.tensor(
            [[-5.2632, -6.1146, -3.0053], [-6.2580, -5.6745, -5.2027], [-7.1642, -6.2351, -5.7986]]
        )
        expected_boxes = torch.tensor([[0.5488, 0.2747, 0.0559], [0.1691, 0.1986, 0.2122], [0.7656, 0.4035, 0.4640]])

    assert torch.allclose(outputs.logits[0, :3, :3], expected_logits.to(device), atol=1e-4)
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes.to(device), atol=1e-4)
    print("Everything ok!")

    if pytorch_dump_folder_path is not None:
        # Save model and image processor
        print(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing model and image processor to the hub...")
        model.push_to_hub(f"nielsr/{model_name}")
        processor.push_to_hub(f"nielsr/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="deformable-detr-detic",
        type=str,
        choices=["deformable-detr-detic", "deformable-detr-box-supervised"],
        help="Name of the model you'd like to convert.",
    )
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
    convert_deformable_detr_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
