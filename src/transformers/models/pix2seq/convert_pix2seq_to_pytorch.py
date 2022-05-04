import argparse
import json
import os
import logging

import torch
import tensorflow as tf
import numpy as np
from PIL import Image

import requests
from huggingface_hub import hf_hub_download
from transformers import AutoFeatureExtractor, Pix2SeqConfig, Pix2SeqModel

logger = logging.get_logger(__name__)


def get_pix2seq_config(pix2seq_name):
    config = Pix2SeqConfig()
    name_split = pix2seq_name.split("_")

    model_size = name_split[1]
    img_size = int(name_split[4])
    window_size = int(name_split[3][-1])

    if model_size == "tiny":
        embed_dim = 96
        depths = (2, 2, 6, 2)
        num_heads = (3, 6, 12, 24)
    elif model_size == "small":
        embed_dim = 96
        depths = (2, 2, 18, 2)
        num_heads = (3, 6, 12, 24)
    elif model_size == "base":
        embed_dim = 128
        depths = (2, 2, 18, 2)
        num_heads = (4, 8, 16, 32)
    else:
        embed_dim = 192
        depths = (2, 2, 18, 2)
        num_heads = (6, 12, 24, 48)

    if "in22k" in pix2seq_name:
        num_classes = 21841
    else:
        num_classes = 1000
        repo_id = "datasets/huggingface/label-files"
        filename = "imagenet-1k-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}

    config.image_size = img_size
    config.num_labels = num_classes
    config.embed_dim = embed_dim
    config.num_heads = num_heads

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
    if "mha" in name:
        name = name.replace("mha", "attention.attention")
    if "_query_dense" in name:
        name = name.replace("_query_dense", "query")
    if "_output_dense" in name:
        name = name.replace("_output_dense", "output.dense")
    if "mlp.mlp_layers.0.dense1" in name:
        name = name.replace("mlp.mlp_layers.0.dense1", "intermediate.dense")
    if "mlp.mlp_layers.0.dense2" in name:
        name = name.replace("mlp.mlp_layers.0.dense2", "output.dense")
    if "mlp.layernorms.0" in name:
        name = name.replace("mlp.layernorms.0", "layernorm_after")
    
    # TODO rename kernel -> weight, gamma -> weight, beta -> bias
    if "kernel" in name:
        name = name.replace("kernel", "weight")
        param = np.transpose(param)
    if "gamma" in name:
        name = name.replace("gamma", "weight")
    if "beta" in name:
        name = name.replace("beta", "bias")
    
    # add prefix
    # name = "pix2seq." + name

    return name, param


def convert_state_dict(orig_state_dict, model):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)

        if "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[1])
            block_num = int(key_split[3])
            dim = model.pix2seq.encoder.layers[layer_num].blocks[block_num].attention.self.all_head_size

            if "weight" in key:
                orig_state_dict[
                    f"pix2seq.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.weight"
                ] = val[:dim, :]
                orig_state_dict[f"pix2seq.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.weight"] = val[
                    dim : dim * 2, :
                ]
                orig_state_dict[
                    f"pix2seq.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.weight"
                ] = val[-dim:, :]
            else:
                orig_state_dict[f"pix2seq.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.bias"] = val[
                    :dim
                ]
                orig_state_dict[f"pix2seq.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.bias"] = val[
                    dim : dim * 2
                ]
                orig_state_dict[f"pix2seq.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.bias"] = val[
                    -dim:
                ]
        else:
            orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def convert_pix2seq_checkpoint(checkpoint_path, pytorch_dump_folder_path):
    config = get_pix2seq_config(checkpoint_path)
    model = Pix2SeqModel(config)
    model.eval()

    # Load weights from TF model
    tf_path = os.path.abspath(checkpoint_path)
    init_vars = tf.train.list_variables(tf_path)
    tf_vars = []
    for name, shape in init_vars:
        if "model" in name and "optimizer" not in name.lower():
            logger.info(f"Loading TF weight {name} with shape {shape}")
            array = tf.train.load_variable(tf_path, name)
            tf_vars.append((name, array.squeeze()))

    # Rename keys
    state_dict = {}
    for name, param in tf_vars.items():
        name, param = rename_key(name, param)
        state_dict[rename_key(name)] = param
    
    model.load_state_dict(state_dict)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    feature_extractor = AutoFeatureExtractor()
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = feature_extractor(images=image, return_tensors="pt")

    outputs = model(**inputs).logits

    # TODO assert outputs

    print(f"Saving model {pix2seq_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)

    print(f"Saving feature extractor to {pytorch_dump_folder_path}")
    feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the Pix2Seq checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_pix2seq_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path)