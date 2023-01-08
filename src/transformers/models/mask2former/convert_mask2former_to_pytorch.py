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
"""Convert Mask2Former checkpoints with Swin backbone from the original repository. URL:
https://github.com/facebookresearch/Mask2Former"""


import argparse
import json
import pickle
from pathlib import Path

import torch
from PIL import Image

import requests
from huggingface_hub import hf_hub_download
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation, MaskFormerImageProcessor, SwinConfig
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_mask2former_config(model_name: str):
    if "tiny" in model_name:
        backbone_config = SwinConfig(out_features=["stage1", "stage2", "stage3", "stage4"])
    elif "small" in model_name:
        # todo
        raise NotImplementedError("To do")
    elif "base" in model_name:
        backbone_config = SwinConfig(
            embed_dim=128,
            window_size=12,
            depths=(2, 2, 18, 2),
            num_heads=(4, 8, 16, 32),
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
    elif "large" in model_name:
        # todo
        raise NotImplementedError("To do")
    else:
        raise ValueError("Model name not supported")

    config = Mask2FormerConfig(backbone_config=backbone_config)

    repo_id = "huggingface/label-files"
    if "ade20k-full" in model_name:
        # this should be ok
        config.num_labels = 847
        filename = "mask2former-ade20k-full-id2label.json"
    elif "ade" in model_name:
        # this should be ok
        config.num_labels = 150
        filename = "ade20k-id2label.json"
    elif "coco-stuff" in model_name:
        # this should be ok
        config.num_labels = 171
        filename = "mask2former-coco-stuff-id2label.json"
    elif "coco" in model_name:
        # TODO
        config.num_labels = 133
        # filename = "coco-panoptic-id2label.json"
    elif "cityscapes" in model_name:
        # this should be ok
        config.num_labels = 19
        filename = "cityscapes-id2label.json"
    elif "vistas" in model_name:
        # this should be ok
        config.num_labels = 65
        filename = "mapillary-vistas-id2label.json"
    elif "youtubevis-2021" in model_name:
        config.num_labels = 40

    # id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    # id2label = {int(k): v for k, v in id2label.items()}

    return config


def create_rename_keys(model_name, config):
    rename_keys = []
    # stem
    # fmt: off
    rename_keys.append(("backbone.patch_embed.proj.weight", "model.pixel_level_module.encoder.model.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("backbone.patch_embed.proj.bias", "model.pixel_level_module.encoder.model.embeddings.patch_embeddings.projection.bias"))
    rename_keys.append(("backbone.patch_embed.norm.weight", "model.pixel_level_module.encoder.model.embeddings.norm.weight"))
    rename_keys.append(("backbone.patch_embed.norm.bias", "model.pixel_level_module.encoder.model.embeddings.norm.bias"))

    # stages
    for i in range(len(config.backbone_config.depths)):
        for j in range(config.backbone_config.depths[i]):
            rename_keys.append((f"backbone.layers.{i}.blocks.{j}.norm1.weight", f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.layernorm_before.weight"))
            rename_keys.append((f"backbone.layers.{i}.blocks.{j}.norm1.bias", f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.layernorm_before.bias"))
            rename_keys.append((f"backbone.layers.{i}.blocks.{j}.attn.relative_position_bias_table", f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.attention.self.relative_position_bias_table"))
            rename_keys.append((f"backbone.layers.{i}.blocks.{j}.attn.relative_position_index", f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.attention.self.relative_position_index"))
            rename_keys.append((f"backbone.layers.{i}.blocks.{j}.attn.proj.weight", f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.attention.output.dense.weight"))
            rename_keys.append((f"backbone.layers.{i}.blocks.{j}.attn.proj.bias", f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.attention.output.dense.bias"))
            rename_keys.append((f"backbone.layers.{i}.blocks.{j}.norm2.weight", f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.layernorm_after.weight"))
            rename_keys.append((f"backbone.layers.{i}.blocks.{j}.norm2.bias", f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.layernorm_after.bias"))
            rename_keys.append((f"backbone.layers.{i}.blocks.{j}.mlp.fc1.weight", f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.intermediate.dense.weight"))
            rename_keys.append((f"backbone.layers.{i}.blocks.{j}.mlp.fc1.bias", f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.intermediate.dense.bias"))
            rename_keys.append((f"backbone.layers.{i}.blocks.{j}.mlp.fc2.weight", f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.output.dense.weight"))
            rename_keys.append((f"backbone.layers.{i}.blocks.{j}.mlp.fc2.bias", f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.output.dense.bias"))

        if i < 3:
            rename_keys.append((f"backbone.layers.{i}.downsample.reduction.weight", f"model.pixel_level_module.encoder.model.encoder.layers.{i}.downsample.reduction.weight"))
            rename_keys.append((f"backbone.layers.{i}.downsample.norm.weight", f"model.pixel_level_module.encoder.model.encoder.layers.{i}.downsample.norm.weight"))
            rename_keys.append((f"backbone.layers.{i}.downsample.norm.bias", f"model.pixel_level_module.encoder.model.encoder.layers.{i}.downsample.norm.bias"))
        rename_keys.append((f"backbone.norm{i}.weight", f"model.pixel_level_module.encoder.hidden_states_norms.{i}.weight"))
        rename_keys.append((f"backbone.norm{i}.bias", f"model.pixel_level_module.encoder.hidden_states_norms.{i}.bias"))

    # input projection layers
    rename_keys.append(("sem_seg_head.pixel_decoder.input_proj.0.0.weight", "model.pixel_level_module.decoder.input_projections.0.0.weight"))
    rename_keys.append(("sem_seg_head.pixel_decoder.input_proj.0.0.bias", "model.pixel_level_module.decoder.input_projections.0.0.bias"))
    rename_keys.append(("sem_seg_head.pixel_decoder.input_proj.0.1.weight", "model.pixel_level_module.decoder.input_projections.0.1.weight"))
    rename_keys.append(("sem_seg_head.pixel_decoder.input_proj.0.1.bias", "model.pixel_level_module.decoder.input_projections.0.1.bias"))
    rename_keys.append(("sem_seg_head.pixel_decoder.input_proj.1.0.weight", "model.pixel_level_module.decoder.input_projections.1.0.weight"))
    rename_keys.append(("sem_seg_head.pixel_decoder.input_proj.1.0.bias", "model.pixel_level_module.decoder.input_projections.1.0.bias"))
    rename_keys.append(("sem_seg_head.pixel_decoder.input_proj.1.1.weight", "model.pixel_level_module.decoder.input_projections.1.1.weight"))
    rename_keys.append(("sem_seg_head.pixel_decoder.input_proj.1.1.bias", "model.pixel_level_module.decoder.input_projections.1.1.bias"))
    rename_keys.append(("sem_seg_head.pixel_decoder.input_proj.2.0.weight", "model.pixel_level_module.decoder.input_projections.2.0.weight"))
    rename_keys.append(("sem_seg_head.pixel_decoder.input_proj.2.0.bias", "model.pixel_level_module.decoder.input_projections.2.0.bias"))
    rename_keys.append(("sem_seg_head.pixel_decoder.input_proj.2.1.weight", "model.pixel_level_module.decoder.input_projections.2.1.weight"))
    rename_keys.append(("sem_seg_head.pixel_decoder.input_proj.2.1.bias", "model.pixel_level_module.decoder.input_projections.2.1.bias"))

    rename_keys.append(("sem_seg_head.pixel_decoder.transformer.level_embed", "model.pixel_level_module.decoder.level_embed"))

    # pixel decoder encoder layers
    for i in range(config.encoder_layers):
        rename_keys.append((f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.self_attn.sampling_offsets.weight", f"model.pixel_level_module.decoder.encoder.layers.{i}.self_attn.sampling_offsets.weight"))
        rename_keys.append((f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.self_attn.sampling_offsets.bias", f"model.pixel_level_module.decoder.encoder.layers.{i}.self_attn.sampling_offsets.bias"))
        rename_keys.append((f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.self_attn.attention_weights.weight", f"model.pixel_level_module.decoder.encoder.layers.{i}.self_attn.attention_weights.weight"))
        rename_keys.append((f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.self_attn.attention_weights.bias", f"model.pixel_level_module.decoder.encoder.layers.{i}.self_attn.attention_weights.bias"))
        rename_keys.append((f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.self_attn.value_proj.weight", f"model.pixel_level_module.decoder.encoder.layers.{i}.self_attn.value_proj.weight"))
        rename_keys.append((f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.self_attn.value_proj.bias", f"model.pixel_level_module.decoder.encoder.layers.{i}.self_attn.value_proj.bias"))
        rename_keys.append((f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.self_attn.output_proj.weight", f"model.pixel_level_module.decoder.encoder.layers.{i}.self_attn.output_proj.weight"))
        rename_keys.append((f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.self_attn.output_proj.bias", f"model.pixel_level_module.decoder.encoder.layers.{i}.self_attn.output_proj.bias"))
        rename_keys.append((f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.norm1.weight", f"model.pixel_level_module.decoder.encoder.layers.{i}.self_attn_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.norm1.bias", f"model.pixel_level_module.decoder.encoder.layers.{i}.self_attn_layer_norm.bias"))
        rename_keys.append((f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.linear1.weight", f"model.pixel_level_module.decoder.encoder.layers.{i}.fc1.weight"))
        rename_keys.append((f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.linear1.bias", f"model.pixel_level_module.decoder.encoder.layers.{i}.fc1.bias"))
        rename_keys.append((f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.linear2.weight", f"model.pixel_level_module.decoder.encoder.layers.{i}.fc2.weight"))
        rename_keys.append((f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.linear2.bias", f"model.pixel_level_module.decoder.encoder.layers.{i}.fc2.bias"))
        rename_keys.append((f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.norm2.weight", f"model.pixel_level_module.decoder.encoder.layers.{i}.final_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.pixel_decoder.transformer.encoder.layers.{i}.norm2.bias", f"model.pixel_level_module.decoder.encoder.layers.{i}.final_layer_norm.bias"))

    # pixel decoder extra features
    rename_keys.append(("sem_seg_head.pixel_decoder.mask_features.weight", "model.pixel_level_module.decoder.mask_projection.weight"))
    rename_keys.append(("sem_seg_head.pixel_decoder.mask_features.bias", "model.pixel_level_module.decoder.mask_projection.bias"))
    rename_keys.append(("sem_seg_head.pixel_decoder.adapter_1.weight", "model.pixel_level_module.decoder.adapter_1.0.weight"))
    rename_keys.append(("sem_seg_head.pixel_decoder.adapter_1.norm.weight", "model.pixel_level_module.decoder.adapter_1.1.weight"))
    rename_keys.append(("sem_seg_head.pixel_decoder.adapter_1.norm.bias", "model.pixel_level_module.decoder.adapter_1.1.bias"))
    rename_keys.append(("sem_seg_head.pixel_decoder.layer_1.weight", "model.pixel_level_module.decoder.layer_1.0.weight"))
    rename_keys.append(("sem_seg_head.pixel_decoder.layer_1.norm.weight", "model.pixel_level_module.decoder.layer_1.1.weight"))
    rename_keys.append(("sem_seg_head.pixel_decoder.layer_1.norm.bias", "model.pixel_level_module.decoder.layer_1.1.bias"))

    # transformer decoder
    for i in range(config.decoder_layers - 1):
        # self-attention needs special treatment
        # rename_keys.append((f"sem_seg_head.predictor.transformer_self_attention_layers.{i}.self_attn.in_proj_weight", f"model.transformer_module.decoder.layers.{i}.self_attn.in_proj_weight"))
        # rename_keys.append((f"sem_seg_head.predictor.transformer_self_attention_layers.{i}.self_attn.in_proj_bias", f"model.transformer_module.decoder.layers.{i}.self_attn.in_proj_bias"))
        rename_keys.append((f"sem_seg_head.predictor.transformer_self_attention_layers.{i}.self_attn.out_proj.weight", f"model.transformer_module.decoder.layers.{i}.self_attn.out_proj.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer_self_attention_layers.{i}.self_attn.out_proj.bias", f"model.transformer_module.decoder.layers.{i}.self_attn.out_proj.bias"))
        rename_keys.append((f"sem_seg_head.predictor.transformer_self_attention_layers.{i}.norm.weight", f"model.transformer_module.decoder.layers.{i}.self_attn_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer_self_attention_layers.{i}.norm.bias", f"model.transformer_module.decoder.layers.{i}.self_attn_layer_norm.bias"))
        rename_keys.append((f"sem_seg_head.predictor.transformer_cross_attention_layers.{i}.multihead_attn.in_proj_weight", f"model.transformer_module.decoder.layers.{i}.cross_attn.in_proj_weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer_cross_attention_layers.{i}.multihead_attn.in_proj_bias", f"model.transformer_module.decoder.layers.{i}.cross_attn.in_proj_bias"))
        rename_keys.append((f"sem_seg_head.predictor.transformer_cross_attention_layers.{i}.multihead_attn.out_proj.weight", f"model.transformer_module.decoder.layers.{i}.cross_attn.out_proj.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer_cross_attention_layers.{i}.multihead_attn.out_proj.bias", f"model.transformer_module.decoder.layers.{i}.cross_attn.out_proj.bias"))
        rename_keys.append((f"sem_seg_head.predictor.transformer_cross_attention_layers.{i}.norm.weight", f"model.transformer_module.decoder.layers.{i}.cross_attn_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer_cross_attention_layers.{i}.norm.bias", f"model.transformer_module.decoder.layers.{i}.cross_attn_layer_norm.bias"))
        rename_keys.append((f"sem_seg_head.predictor.transformer_ffn_layers.{i}.linear1.weight", f"model.transformer_module.decoder.layers.{i}.fc1.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer_ffn_layers.{i}.linear1.bias", f"model.transformer_module.decoder.layers.{i}.fc1.bias"))
        rename_keys.append((f"sem_seg_head.predictor.transformer_ffn_layers.{i}.linear2.weight", f"model.transformer_module.decoder.layers.{i}.fc2.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer_ffn_layers.{i}.linear2.bias", f"model.transformer_module.decoder.layers.{i}.fc2.bias"))
        rename_keys.append((f"sem_seg_head.predictor.transformer_ffn_layers.{i}.norm.weight", f"model.transformer_module.decoder.layers.{i}.final_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer_ffn_layers.{i}.norm.bias", f"model.transformer_module.decoder.layers.{i}.final_layer_norm.bias"))

    # final
    rename_keys.append(("sem_seg_head.predictor.decoder_norm.weight", "model.transformer_module.decoder.layernorm.weight"))
    rename_keys.append(("sem_seg_head.predictor.decoder_norm.bias", "model.transformer_module.decoder.layernorm.bias"))
    rename_keys.append(("sem_seg_head.predictor.query_embed.weight", "model.transformer_module.queries_embedder.weight"))
    if "youtube" not in model_name:
        rename_keys.append(("sem_seg_head.predictor.static_query.weight", "model.transformer_module.queries_features.weight"))
    rename_keys.append(("sem_seg_head.predictor.level_embed.weight", "model.transformer_module.level_embed.weight"))
    rename_keys.append(("sem_seg_head.predictor.class_embed.weight", "class_predictor.weight"))
    rename_keys.append(("sem_seg_head.predictor.class_embed.bias", "class_predictor.bias"))
    rename_keys.append(("sem_seg_head.predictor.mask_embed.layers.0.weight", "model.transformer_module.decoder.mask_predictor.mask_embedder.0.0.weight"))
    rename_keys.append(("sem_seg_head.predictor.mask_embed.layers.0.bias", "model.transformer_module.decoder.mask_predictor.mask_embedder.0.0.bias"))
    rename_keys.append(("sem_seg_head.predictor.mask_embed.layers.1.weight", "model.transformer_module.decoder.mask_predictor.mask_embedder.1.0.weight"))
    rename_keys.append(("sem_seg_head.predictor.mask_embed.layers.1.bias", "model.transformer_module.decoder.mask_predictor.mask_embedder.1.0.bias"))
    rename_keys.append(("sem_seg_head.predictor.mask_embed.layers.2.weight", "model.transformer_module.decoder.mask_predictor.mask_embedder.2.0.weight"))
    rename_keys.append(("sem_seg_head.predictor.mask_embed.layers.2.bias", "model.transformer_module.decoder.mask_predictor.mask_embedder.2.0.bias"))

    # fmt: on

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_swin_q_k_v(state_dict, backbone_config):
    num_features = [int(backbone_config.embed_dim * 2**i) for i in range(len(backbone_config.depths))]
    for i in range(len(backbone_config.depths)):
        dim = num_features[i]
        for j in range(backbone_config.depths[i]):
            # fmt: off
            # read in weights + bias of input projection layer (in original implementation, this is a single matrix + bias)
            in_proj_weight = state_dict.pop(f"backbone.layers.{i}.blocks.{j}.attn.qkv.weight")
            in_proj_bias = state_dict.pop(f"backbone.layers.{i}.blocks.{j}.attn.qkv.bias")
            # next, add query, keys and values (in that order) to the state dict
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.attention.self.query.weight"] = in_proj_weight[:dim, :]
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.attention.self.query.bias"] = in_proj_bias[: dim]
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.attention.self.key.weight"] = in_proj_weight[
                dim : dim * 2, :
            ]
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.attention.self.key.bias"] = in_proj_bias[
                dim : dim * 2
            ]
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.attention.self.value.weight"] = in_proj_weight[
                -dim :, :
            ]
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.attention.self.value.bias"] = in_proj_bias[-dim :]
            # fmt: on


def read_in_decoder_q_k_v(state_dict, config):
    for i in range(config.decoder_layers - 1):
        dim = config.hidden_dim
        # fmt: off
        # read in weights + bias of input projection layer (in original implementation, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"sem_seg_head.predictor.transformer_self_attention_layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"sem_seg_head.predictor.transformer_self_attention_layers.{i}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.transformer_module.decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:dim, :]
        state_dict[f"model.transformer_module.decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[: dim]
        state_dict[f"model.transformer_module.decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[
            dim : dim * 2, :
        ]
        state_dict[f"model.transformer_module.decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[
            dim : dim * 2
        ]
        state_dict[f"model.transformer_module.decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[
            -dim :, :
        ]
        state_dict[f"model.transformer_module.decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-dim :]
        # fmt: on


# We will verify our results on an image of cute cats
def prepare_img() -> torch.Tensor:
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_mask2former_checkpoint(model_name: str, pytorch_dump_folder_path: str, push_to_hub: bool = False):
    """
    Copy/paste/tweak model's weights to our Mask2Former structure.
    """
    config = get_mask2former_config(model_name)

    # load original state_dict
    model_name_to_filename = {
        "mask2former-swin-base-coco-panoptic": "mask2former_swin_base_coco_panoptic.pkl",
        "mask2former-swin-base-ade-semantic": "maskformer_swin_base_ade_panoptic.pkl",
        "mask2former-swin-base-youtubevis-2021": "mask2former_swin_base_youtube.pkl",
    }

    filename = model_name_to_filename[model_name]
    filepath = hf_hub_download(
        repo_id="nielsr/mask2former-original-checkpoints", filename=filename, repo_type="dataset"
    )
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    state_dict = data["model"]

    for name, param in state_dict.items():
        print(name, param.shape)

    # rename keys
    rename_keys = create_rename_keys(model_name, config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_swin_q_k_v(state_dict, config.backbone_config)
    read_in_decoder_q_k_v(state_dict, config)

    # update to torch tensors
    for key, value in state_dict.items():
        state_dict[key] = torch.from_numpy(value)

    # load 🤗 model
    model = Mask2FormerForUniversalSegmentation(config)
    model.eval()

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert missing_keys == [
        "model.pixel_level_module.encoder.model.layernorm.weight",
        "model.pixel_level_module.encoder.model.layernorm.bias",
    ]
    assert len(unexpected_keys) == 0

    # verify results
    image = prepare_img()
    if "vistas" in model_name:
        ignore_index = 65
    elif "cityscapes" in model_name:
        ignore_index = 65535
    else:
        ignore_index = 255
    reduce_labels = True if "ade" in model_name else False
    processor = MaskFormerImageProcessor(ignore_index=ignore_index, reduce_labels=reduce_labels)

    inputs = processor(image, return_tensors="pt")

    outputs = model(**inputs)

    print("Logits:", outputs.class_queries_logits[0, :3, :3])

    if model_name == "mask2former-swin-base-coco-panoptic":
        expected_logits = torch.tensor(
            [[1.2436, -9.0607, -4.7816], [-2.1374, -8.1785, -3.5263], [-1.3945, -6.2297, -4.8062]]
        )
    elif model_name == "mask2former-swin-base-ade-semantic":
        expected_logits = torch.tensor(
            [[4.0837, -1.1718, -1.4966], [2.5418, -3.0524, -1.1140], [2.8320, -3.1094, -3.3143]]
        )
    assert torch.allclose(outputs.class_queries_logits[0, :3, :3], expected_logits, atol=1e-4)
    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing model and processor to the hub...")
        model.push_to_hub(f"nielsr/{model_name}")
        processor.push_to_hub(f"nielsr/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="mask2former-swin-base-coco-panoptic",
        type=str,
        choices=["mask2former-swin-base-coco-panoptic", "mask2former-swin-base-ade-semantic", "mask2former-swin-base-youtubevis-2021"],
        help=("Name of the Mask2Former model you'd like to convert",),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_mask2former_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
