import argparse
import json

import torch
from PIL import Image

import requests
from huggingface_hub import hf_hub_download
from transformers import AutoFeatureExtractor, SwinConfig, SwinForSemanticSegmentation


def get_swin_config(model_name):
    config = SwinConfig()

    image_size = 224
    window_size = 7

    if "tiny" in model_name:
        embed_dim = 96
        depths = (2, 2, 6, 2)
        num_heads = (3, 6, 12, 24)
    elif "small" in model_name:
        embed_dim = 96
        depths = (2, 2, 18, 2)
        num_heads = (3, 6, 12, 24)
    elif "base" in model_name:
        embed_dim = 128
        depths = (2, 2, 18, 2)
        num_heads = (4, 8, 16, 32)
    else:
        raise ValueError("Model name not supported, should include 'tiny', 'small' or 'base'")

    # TODO id2label
    config.num_labels = 150

    config.image_size = image_size
    config.embed_dim = embed_dim
    config.depths = depths
    config.num_heads = num_heads
    config.window_size = window_size

    return config


def rename_key(name):
    if "backbone" in name:
        name = name.replace("backbone", "")
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "embeddings.norm")
    if "layers" in name:
        name = "encoder" + name
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name:
        name = name.replace("attn", "attention.self")
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")

    # if name == "norm.weight":
    #     name = "layernorm.weight"
    # if name == "norm.bias":
    #     name = "layernorm.bias"

    if "decode_head" in name or "auxiliary_head" in name:
        if "conv_seg" in name:
            name = name.replace("conv_seg", "classifier")
    else:
        name = "swin." + name

    return name


def convert_state_dict(orig_state_dict, model):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if "mask" in key:
            continue
        elif "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[2])
            block_num = int(key_split[4])
            dim = model.swin.encoder.layers[layer_num].blocks[block_num].attention.self.all_head_size

            if "weight" in key:
                orig_state_dict[
                    f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.weight"
                ] = val[:dim, :]
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.weight"] = val[
                    dim : dim * 2, :
                ]
                orig_state_dict[
                    f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.weight"
                ] = val[-dim:, :]
            else:
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.bias"] = val[
                    :dim
                ]
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.bias"] = val[
                    dim : dim * 2
                ]
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.bias"] = val[
                    -dim:
                ]
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def convert_swin_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    model_to_url = {
        "swin-tiny-patch4-window7-finetuned-ade20k": "https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_tiny_patch4_window7_512x512.pth",
        "swin-small-patch4-window7-finetuned-ade20k": "https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_small_patch4_window7_512x512.pth",
        "swin-base-patch4-window7-finetuned-ade20k": "https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_base_patch4_window7_512x512.pth",
    }
    
    checkpoint_url = model_to_url[model_name]
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")['state_dict']

    for name, param in state_dict.items():
        print(name, param.shape)

    config = get_swin_config(model_name)
    model = SwinForSemanticSegmentation(config)
    model.eval()

    new_state_dict = convert_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/{}".format(model_name.replace("_", "-")))
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = feature_extractor(images=image, return_tensors="pt")

    outputs = model(**inputs).logits

    # TODO assert values 
    # assert torch.allclose(timm_outs, hf_outs, atol=1e-3)

    if pytorch_dump_folder_path is not None:
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)

        print(f"Saving feature extractor to {pytorch_dump_folder_path}")
        feature_extractor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print(f"Pushing model to hub")
        model.push_to_hub(f"microsoft/{model_name}")

        print(f"Pushing feature extractor to hub")
        feature_extractor.push_to_hub(f"microsoft/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="swin-tiny-patch4-window7-finetuned-ade20k",
        type=str,
        choices=["swin-tiny-patch4-window7-finetuned-ade20k", "swin-small-patch4-window7-finetuned-ade20k", "swin-base-patch4-window7-finetuned-ade20k"],
        help="Name of the Swin timm model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether to push the model to the hub after conversion."
    )

    args = parser.parse_args()
    convert_swin_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
