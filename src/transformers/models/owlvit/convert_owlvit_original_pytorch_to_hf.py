import argparse
import json

import numpy as np
import torch
from PIL import Image

import requests
from flax.training import checkpoints
from flax.traverse_util import flatten_dict
from huggingface_hub import hf_hub_download
from transformers import AutoFeatureExtractor, OwlViTConfig, OwlViTModel


def get_owlvit_config(model_name):
    config = OwlViTConfig()
    return config


def rename_key(name):
    # text encoder
    if "clip.text" in name:
        name = name.replace("clip.text", "text_model")
    if "positional_embedding.embedding" in name:
        name = name.replace("positional_embedding.embedding", "embeddings.position_embedding.weight")
    if "token_embedding.embedding" in name:
        name = name.replace("token_embedding.embedding", "embeddings.token_embedding.weight")
    if "transformer.resblocks" in name:
        name = name.replace("transformer.resblocks", "encoder.layers")
    if "ln_1":
        name = name.replace("ln_1", "layer_norm1")
    if "ln_2":
        name = name.replace("ln_2", "layer_norm2")
    if "c_fc" in name:
        name = name.replace("c_fc", "fc1")
    if "c_proj" in name:
        name = name.replace("c_proj", "fc2")
    if "kernel" in name:
        name = name.replace("kernel", "weight")
    if "scale" in name:
        name = name.replace("scale", "weight")
    if "ln_final" in name:
        name = name.replace("ln_final", "final_layer_norm")
    # visual encoder
    if "clip.visual" in name:
        name = name.replace("clip.visual", "vision_model")
    if "class_embedding" in name:
        name = name.replace("class_embedding", "embeddings.class_embedding")
    if "conv1" in name:
        name = name.replace("conv1", "embeddings.patch_embedding.weight")
    if "positional_embedding" in name:
        name = name.replace("positional_embedding", "embeddings.position_embedding.weight")
    if "ln_pre" in name:
        name = name.replace("ln_pre", "pre_layrnorm")
    if "ln_post" in name:
        name = name.replace("ln_post", "post_layernorm")

    return name


def convert_state_dict(orig_state_dict, config):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)
        key = ".".join(key)

        if "attn" in key:
            print("Key:", key)
            key_split = key.split(".")
            layer_num = int(key_split[4])
            # set prefix and dim
            if "text" in key:
                prefix = "text_model"
                dim = config.text_config.hidden_size
            else:
                prefix = "vision_model"
                dim = config.vision_config.hidden_size
            # set type
            if "query" in key:
                type = "q_proj"
            elif "key" in key:
                type = "k_proj"
            elif "value" in key:
                type = "v_proj"
            elif "out" in key:
                type = "out_proj"
            else:
                raise ValueError(".")
            # set type_bis
            if "kernel" in key:
                type_bis = "weight"
            elif "bias" in key:
                type_bis = "bias"
            val = torch.from_numpy(val)

            if "bias" in key:
                val = val.flatten()
            elif type == "out_proj":
                val = val.reshape(-1, dim)
            else:
                val = val.reshape(dim, -1)
            orig_state_dict[f"{prefix}.encoder.layers.{layer_num}.self_attn.{type}.{type_bis}"] = val

        else:
            new_name = rename_key(key)

            if new_name[-6:] == "weight" and "embedding" not in new_name:
                val = np.transpose(val)

            orig_state_dict[rename_key(key)] = torch.from_numpy(val)

    return orig_state_dict


def convert_owlvit_checkpoint(checkpoint_path, model_name, pytorch_dump_folder_path):
    config = get_owlvit_config(model_name)
    model = OwlViTModel(config)
    model.eval()

    restored = checkpoints.restore_checkpoint(checkpoint_path, target=None)
    state_dict = flatten_dict(restored["optimizer"]["target"]["backbone"])
    state_dict = convert_state_dict(state_dict, config)
    model.load_state_dict(state_dict)

    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    # feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/{}".format(model_name.replace("_", "-")))
    # image = Image.open(requests.get(url, stream=True).raw)
    # inputs = feature_extractor(images=image, return_tensors="pt")

    # timm_outs = timm_model(inputs["pixel_values"])
    # hf_outs = model(**inputs).logits

    # assert torch.allclose(timm_outs, hf_outs, atol=1e-3)

    # print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
    # model.save_pretrained(pytorch_dump_folder_path)

    # print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    # feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_path",
        default="/home/niels//checkpoints/owlvit/clip_vit_b32_b0203fc",
        type=str,
        help="Path to the original Flax checkpoint.",
    )
    parser.add_argument(
        "--model_name",
        default="owlvit_tiny_patch4_window7_224",
        type=str,
        help="Name of the Swin timm model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_owlvit_checkpoint(args.checkpoint_path, args.model_name, args.pytorch_dump_folder_path)
