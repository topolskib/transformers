import argparse
import os

import numpy as np
import tensorflow as tf
import torch
from PIL import Image

import requests
from transformers import Pix2SeqConfig, Pix2SeqModel, ViTFeatureExtractor


def get_pix2seq_config(model_name):
    config = Pix2SeqConfig()

    return config


def rename_key(name, param):
    # general renamings
    if "/.ATTRIBUTES/VARIABLE_VALUE" in name:
        name = name.replace("/.ATTRIBUTES/VARIABLE_VALUE", "") 
    if "/" in name:
        name = name.replace("/", ".")
    # stem conv
    if "model.encoder.stem_conv" in name:
        name = name.replace("model.encoder.stem_conv", "embeddings.patch_embeddings.projection")
    if "model.encoder.stem_conv.bias" in name:
        name = name.replace("model.encoder.stem_conv.bias", "embeddings.patch_embeddings.projection")
    # stem layernorm
    if "model.encoder.stem_ln" in name:
        name = name.replace("model.encoder.stem_ln", "embeddings.patch_embeddings.layer_norm")
    if "model.encoder.stem_ln.beta" in name:
        name = name.replace("model.encoder.stem_ln", "embeddings.patch_embeddings.layer_norm")
    # encoder layers
    if "model.encoder.transformer_encoder.enc_layers" in name:
        name = name.replace("model.encoder.transformer_encoder.enc_layers", "encoder.layer")
    if "mha_ln" in name:
        name = name.replace("mha_ln", "layernorm_before")
    if "mha._output_dense" in name:
        name = name.replace("mha._output_dense", "attention.output.dense")
    if "mha" in name:
        name = name.replace("mha", "attention.attention")
    if "_query_dense" in name:
        name = name.replace("_query_dense", "query")
    if "_key_dense" in name:
        name = name.replace("_key_dense", "key")
    if "_value_dense" in name:
        name = name.replace("_value_dense", "value")
    if "mlp.mlp_layers.0.dense1" in name:
        name = name.replace("mlp.mlp_layers.0.dense1", "intermediate.dense")
    if "mlp.mlp_layers.0.dense2" in name:
        name = name.replace("mlp.mlp_layers.0.dense2", "output.dense")
    if "mlp.layernorms.0" in name:
        name = name.replace("mlp.layernorms.0", "layernorm_after")
    # output layer norm
    if "model.encoder.output_ln" in name:
        name = name.replace("model.encoder.output_ln", "layernorm")
    
    # handle qkv
    if "attention.output.dense" in name:
        if "kernel" in name:
            # (12, 64, 768) -> (768, 768) for weights 
            param = np.reshape(param, (param.shape[-1], -1))

    if ("query" in name or "key" in name or "value" in name):
        # print("Updating param for parameter:", name)
        # print("Old shape of param", param.shape)
        if "kernel" in name:
            # (768, 12, 64) -> (768, 768) for weights, or 
            param = np.reshape(param, (param.shape[0], -1))
        elif "bias" in name:
            # (12, 64) -> (768,) for biases
            param = param.flatten()
        # print("New shape of param:", param.shape)
    
    # rename kernel, gamma and beta (+ important: transpose if kernel!)
    if "kernel" in name:
        name = name.replace("kernel", "weight")
        if "patch_embeddings" in name:
            # important: conv2d layers have a special transpose
            param = np.transpose(param, axes=(3, 2, 0, 1))
        else:
            param = np.transpose(param)
    if "gamma" in name:
        name = name.replace("gamma", "weight")
    if "beta" in name:
        name = name.replace("beta", "bias")

    # add prefix
    # name = "pix2seq." + name

    return name, param


def convert_pix2seq_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path):
    config = get_pix2seq_config(checkpoint_path)
    model = Pix2SeqModel(config)
    model.eval()

    # Load weights from TF model
    tf_path = os.path.abspath(checkpoint_path)
    init_vars = tf.train.list_variables(tf_path)
    tf_vars = dict()
    for name, shape in init_vars:
        if "model" in name and "optimizer" not in name.lower():
            print(f"Loading TF weight {name} with shape {shape}")
            array = tf.train.load_variable(tf_path, name)
            tf_vars[name] = array.squeeze()

    # Rename keys
    state_dict = {}
    for name, param in tf_vars.items():
        if "decoder" in name or name.startswith("model/proj"):
            continue
        name, param = rename_key(name, param)
        state_dict[name] = torch.from_numpy(param)
    
    model.load_state_dict(state_dict)
    model.eval()

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    with open('/home/niels/checkpoints/pix2seq/pixel_values.npy', 'rb') as f:
        pixel_values = np.load(f)
        pixel_values = torch.from_numpy(pixel_values)
        pixel_values = pixel_values.permute(0, 3, 1, 2)
        print("Shape of pixel values:", pixel_values.shape)

    with torch.no_grad():
        outputs = model(pixel_values)

    # TODO assert outputs on cats image
    last_hidden_state = outputs.last_hidden_state
    print("Shape of last hidden states:", last_hidden_state.shape)

    print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)

    # print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    # feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        type=str,
        default="vit-base",
        help="Name of the Pix2Seq model you'd like to convert.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/niels/checkpoints/pix2seq/ckpt-112728.index",
        help="Path to the Pix2Seq checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_pix2seq_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path)