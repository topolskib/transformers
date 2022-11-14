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
"""Convert MaskFormer checkpoints with a Swin backbone from the original repository. URL:
https://github.com/facebookresearch/MaskFormer"""


import argparse
import pickle
from pathlib import Path

import torch
from PIL import Image

import requests
from transformers import MaskFormerConfig, MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation, SwinConfig
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_maskformer_config(model_name: str):
    if "tiny" in model_name:
        backbone_config = SwinConfig.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    else:
        raise ValueError("Not supported")
    config = MaskFormerConfig(backbone_config=backbone_config)

    # TODO id2label mappings
    config.num_labels = 150

    return config


def create_rename_keys(config):
    rename_keys = []
    # stem
    # fmt: off
    rename_keys.append(("backbone.patch_embed.proj.weight", "model.pixel_level_module.encoder.model.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("backbone.patch_embed.proj.bias", "model.pixel_level_module.encoder.model.embeddings.patch_embeddings.projection.bias"))
    rename_keys.append(("backbone.patch_embed.norm.weight", "model.pixel_level_module.encoder.model.embeddings.norm.weight"))
    rename_keys.append(("backbone.patch_embed.norm.bias", "model.pixel_level_module.encoder.model.embeddings.norm.bias"))
    # fmt: on
    # stages
    # fmt: off
    for stage_idx in range(len(config.backbone_config.depths)):
        for layer_idx in range(config.backbone_config.depths[stage_idx]):
            rename_keys.append((f"backbone.layers.{stage_idx}.blocks.{layer_idx}.norm1.weight", f"model.pixel_level_module.encoder.model.encoder.layers.{stage_idx}.blocks.{layer_idx}.layernorm_before.weight"))
            rename_keys.append((f"backbone.layers.{stage_idx}.blocks.{layer_idx}.norm1.bias", f"model.pixel_level_module.encoder.model.encoder.layers.{stage_idx}.blocks.{layer_idx}.layernorm_before.bias"))
            rename_keys.append((f"backbone.layers.{stage_idx}.blocks.{layer_idx}.norm2.weight", f"model.pixel_level_module.encoder.model.encoder.layers.{stage_idx}.blocks.{layer_idx}.layernorm_after.weight"))
            rename_keys.append((f"backbone.layers.{stage_idx}.blocks.{layer_idx}.norm2.bias", f"model.pixel_level_module.encoder.model.encoder.layers.{stage_idx}.blocks.{layer_idx}.layernorm_after.bias"))
            rename_keys.append((f"backbone.layers.{stage_idx}.blocks.{layer_idx}.attn.relative_position_bias_table", f"model.pixel_level_module.encoder.model.encoder.layers.{stage_idx}.blocks.{layer_idx}.attention.self.relative_position_bias_table"))
            rename_keys.append((f"backbone.layers.{stage_idx}.blocks.{layer_idx}.attn.relative_position_index", f"model.pixel_level_module.encoder.model.encoder.layers.{stage_idx}.blocks.{layer_idx}.attention.self.relative_position_index"))
            rename_keys.append((f"backbone.layers.{stage_idx}.blocks.{layer_idx}.mlp.fc1.weight", f"model.pixel_level_module.encoder.model.encoder.layers.{stage_idx}.blocks.{layer_idx}.intermediate.dense.weight"))
            rename_keys.append((f"backbone.layers.{stage_idx}.blocks.{layer_idx}.mlp.fc1.bias", f"model.pixel_level_module.encoder.model.encoder.layers.{stage_idx}.blocks.{layer_idx}.intermediate.dense.bias"))
            rename_keys.append((f"backbone.layers.{stage_idx}.blocks.{layer_idx}.mlp.fc2.weight", f"model.pixel_level_module.encoder.model.encoder.layers.{stage_idx}.blocks.{layer_idx}.output.dense.weight"))
            rename_keys.append((f"backbone.layers.{stage_idx}.blocks.{layer_idx}.mlp.fc2.bias", f"model.pixel_level_module.encoder.model.encoder.layers.{stage_idx}.blocks.{layer_idx}.output.dense.bias"))
            rename_keys.append((f"backbone.layers.{stage_idx}.blocks.{layer_idx}.attn.proj.weight", f"model.pixel_level_module.encoder.model.encoder.layers.{stage_idx}.blocks.{layer_idx}.attention.output.dense.weight"))
            rename_keys.append((f"backbone.layers.{stage_idx}.blocks.{layer_idx}.attn.proj.bias", f"model.pixel_level_module.encoder.model.encoder.layers.{stage_idx}.blocks.{layer_idx}.attention.output.dense.bias"))
        if stage_idx < 3:
            rename_keys.append((f"backbone.layers.{stage_idx}.downsample.reduction.weight", f"model.pixel_level_module.encoder.model.encoder.layers.{stage_idx}.downsample.reduction.weight"))
            rename_keys.append((f"backbone.layers.{stage_idx}.downsample.norm.weight", f"model.pixel_level_module.encoder.model.encoder.layers.{stage_idx}.downsample.norm.weight"))
            rename_keys.append((f"backbone.layers.{stage_idx}.downsample.norm.bias", f"model.pixel_level_module.encoder.model.encoder.layers.{stage_idx}.downsample.norm.bias"))
        rename_keys.append((f"backbone.norm{stage_idx}.weight", f"model.pixel_level_module.encoder.hidden_states_norms.{stage_idx}.weight"))
        rename_keys.append((f"backbone.norm{stage_idx}.bias", f"model.pixel_level_module.encoder.hidden_states_norms.{stage_idx}.bias"))
    # fmt: on
    # FPN
    # fmt: off
    rename_keys.append(("sem_seg_head.layer_4.weight", "model.pixel_level_module.decoder.fpn.stem.0.weight"))
    rename_keys.append(("sem_seg_head.layer_4.norm.weight", "model.pixel_level_module.decoder.fpn.stem.1.weight"))
    rename_keys.append(("sem_seg_head.layer_4.norm.bias", "model.pixel_level_module.decoder.fpn.stem.1.bias"))
    for source_index, target_index in zip(range(3, 0, -1), range(0, 3)):
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.0.weight"))
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.norm.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.1.weight"))
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.norm.bias", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.1.bias"))
        rename_keys.append((f"sem_seg_head.layer_{source_index}.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.0.weight"))
        rename_keys.append((f"sem_seg_head.layer_{source_index}.norm.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.1.weight"))
        rename_keys.append((f"sem_seg_head.layer_{source_index}.norm.bias", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.1.bias"))
    rename_keys.append(("sem_seg_head.mask_features.weight", "model.pixel_level_module.decoder.mask_projection.weight"))
    rename_keys.append(("sem_seg_head.mask_features.bias", "model.pixel_level_module.decoder.mask_projection.bias"))
    # fmt: on

    # Transformer decoder
    # fmt: off
    for idx in range(config.decoder_config.decoder_layers):
        # self-attention out projection
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.out_proj.weight", f"model.transformer_module.decoder.layers.{idx}.self_attn.out_proj.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.out_proj.bias", f"model.transformer_module.decoder.layers.{idx}.self_attn.out_proj.bias"))
        # cross-attention out projection
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.out_proj.weight", f"model.transformer_module.decoder.layers.{idx}.encoder_attn.out_proj.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.out_proj.bias", f"model.transformer_module.decoder.layers.{idx}.encoder_attn.out_proj.bias"))
        # MLP 1
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear1.weight", f"model.transformer_module.decoder.layers.{idx}.fc1.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear1.bias", f"model.transformer_module.decoder.layers.{idx}.fc1.bias"))
        # MLP 2
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear2.weight", f"model.transformer_module.decoder.layers.{idx}.fc2.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear2.bias", f"model.transformer_module.decoder.layers.{idx}.fc2.bias"))
        # layernorm 1 (self-attention layernorm)
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm1.weight", f"model.transformer_module.decoder.layers.{idx}.self_attn_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm1.bias", f"model.transformer_module.decoder.layers.{idx}.self_attn_layer_norm.bias"))
        # layernorm 2 (cross-attention layernorm)
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm2.weight", f"model.transformer_module.decoder.layers.{idx}.encoder_attn_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm2.bias", f"model.transformer_module.decoder.layers.{idx}.encoder_attn_layer_norm.bias"))
        # layernorm 3 (final layernorm)
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm3.weight", f"model.transformer_module.decoder.layers.{idx}.final_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm3.bias", f"model.transformer_module.decoder.layers.{idx}.final_layer_norm.bias"))

    rename_keys.append(("sem_seg_head.predictor.transformer.decoder.norm.weight", "model.transformer_module.decoder.layernorm.weight"))
    rename_keys.append(("sem_seg_head.predictor.transformer.decoder.norm.bias", "model.transformer_module.decoder.layernorm.bias"))
    # fmt: on

    # heads on top
    # fmt: off
    rename_keys.append(("sem_seg_head.predictor.query_embed.weight", "model.transformer_module.queries_embedder.weight"))

    rename_keys.append(("sem_seg_head.predictor.input_proj.weight", "model.transformer_module.input_projection.weight"))
    rename_keys.append(("sem_seg_head.predictor.input_proj.bias", "model.transformer_module.input_projection.bias"))

    rename_keys.append(("sem_seg_head.predictor.class_embed.weight", "class_predictor.weight"))
    rename_keys.append(("sem_seg_head.predictor.class_embed.bias", "class_predictor.bias"))

    for i in range(3):
        rename_keys.append((f"sem_seg_head.predictor.mask_embed.layers.{i}.weight", f"mask_embedder.{i}.0.weight"))
        rename_keys.append((f"sem_seg_head.predictor.mask_embed.layers.{i}.bias", f"mask_embedder.{i}.0.bias"))
    # fmt: on

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# we split up the matrix of each encoder layer into queries, keys and values
# fmt: off
def read_in_swin_q_k_v(state_dict, config):
    
     for stage_idx in range(len(config.backbone_config.depths)):
        for layer_idx in range(config.backbone_config.depths[stage_idx]):
            # read in weights + bias of input projection layer (in the original implementation, this is a single matrix + bias)
            in_proj_weight = state_dict.pop(f"backbone.layers.{stage_idx}.blocks.{layer_idx}.attn.qkv.weight")
            in_proj_bias = state_dict.pop(f"backbone.layers.{stage_idx}.blocks.{layer_idx}.attn.qkv.bias")
            hidden_size = int(config.backbone_config.embed_dim * 2** stage_idx)
            # next, add query, keys and values (in that order) to the state dict
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{stage_idx}.blocks.{layer_idx}.attention.self.query.weight"] = in_proj_weight[: hidden_size, :]
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{stage_idx}.blocks.{layer_idx}.attention.self.query.bias"] = in_proj_bias[:hidden_size]
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{stage_idx}.blocks.{layer_idx}.attention.self.key.weight"] = in_proj_weight[hidden_size : hidden_size * 2, :]
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{stage_idx}.blocks.{layer_idx}.attention.self.key.bias"] = in_proj_bias[hidden_size : hidden_size * 2]
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{stage_idx}.blocks.{layer_idx}.attention.self.value.weight"] = in_proj_weight[-hidden_size :, :]
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{stage_idx}.blocks.{layer_idx}.attention.self.value.bias"] = in_proj_bias[-hidden_size :] 
# fmt: on

# we split up the matrix of each encoder layer into queries, keys and values
def read_in_decoder_q_k_v(state_dict, config):
    # fmt: off
    hidden_size = config.decoder_config.hidden_size
    for idx in range(config.decoder_config.decoder_layers):
        # read in weights + bias of self-attention input projection layer (in the original implementation, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.q_proj.weight"] = in_proj_weight[: hidden_size, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.q_proj.bias"] = in_proj_bias[:config.hidden_size]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.k_proj.weight"] = in_proj_weight[hidden_size : hidden_size * 2, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.k_proj.bias"] = in_proj_bias[hidden_size : hidden_size * 2]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.v_proj.weight"] = in_proj_weight[-hidden_size :, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.v_proj.bias"] = in_proj_bias[-hidden_size :]
        # read in weights + bias of cross-attention input projection layer (in the original implementation, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.q_proj.weight"] = in_proj_weight[: hidden_size, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.q_proj.bias"] = in_proj_bias[:config.hidden_size]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.k_proj.weight"] = in_proj_weight[hidden_size : hidden_size * 2, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.k_proj.bias"] = in_proj_bias[hidden_size : hidden_size * 2]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.v_proj.weight"] = in_proj_weight[-hidden_size :, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.v_proj.bias"] = in_proj_bias[-hidden_size :]
    # fmt: on


# We will verify our results on an image of cute cats
def prepare_img() -> torch.Tensor:
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_maskformer_checkpoint(
    model_name: str, checkpoint_path: str, pytorch_dump_folder_path: str, push_to_hub: bool = False
):
    """
    Copy/paste/tweak model's weights to our MaskFormer structure.
    """
    config = get_maskformer_config(model_name)

    # load original state_dict
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    state_dict = data["model"]

    # rename keys
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_swin_q_k_v(state_dict, config)
    read_in_decoder_q_k_v(state_dict, config)

    # update to torch tensors
    for key, value in state_dict.items():
        state_dict[key] = torch.from_numpy(value)

    # load 🤗 model
    model = MaskFormerForInstanceSegmentation(config)
    model.eval()

    for name, param in model.named_parameters():
        print(name, param.shape)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    # verify results
    image = prepare_img()
    feature_extractor = MaskFormerFeatureExtractor(ignore_index=255, reduce_labels=True)

    inputs = feature_extractor(image, return_tensors="pt")

    outputs = model(**inputs)
    expected_logits = torch.tensor(
        [[6.7710, -0.1452, -3.5687], [1.9165, -1.0010, -1.8614], [3.6209, -0.2950, -1.3813]]
    )
    assert torch.allclose(outputs.class_queries_logits[0, :3, :3], expected_logits, atol=1e-4)
    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving feature extractor to {pytorch_dump_folder_path}")
        # feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing to the hub...")
        model.push_to_hub(f"nielsr/{model_name}")
        # feature_extractor.push_to_hub(model_name, organization="hustvl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="maskformer-swin-tiny-ade",
        type=str,
        help=("Name of the MaskFormer model you'd like to convert",),
    )
    parser.add_argument(
        "--checkpoint_path",
        default="/Users/nielsrogge/Documents/MaskFormer_checkpoints/MaskFormer-Swin-tiny-ADE20k/model.pkl",
        type=str,
        help="Path to the original state dict (.pth file).",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_maskformer_checkpoint(
        args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub
    )
