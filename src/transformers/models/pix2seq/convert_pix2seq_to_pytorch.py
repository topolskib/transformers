import argparse
import os

import numpy as np
import tensorflow as tf
import torch

from transformers import Pix2SeqConfig
from transformers.models.pix2seq.modeling_pix2seq import Pix2SeqForConditionalGeneration


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
        name = name.replace("model.encoder.stem_conv", "encoder.embeddings.patch_embeddings.projection")
    if "model.encoder.stem_conv.bias" in name:
        name = name.replace("model.encoder.stem_conv.bias", "encoder.embeddings.patch_embeddings.projection")
    # stem layernorm
    if "model.encoder.stem_ln" in name:
        name = name.replace("model.encoder.stem_ln", "encoder.embeddings.patch_embeddings.layer_norm")
    if "model.encoder.stem_ln.beta" in name:
        name = name.replace("model.encoder.stem_ln", "encoder.embeddings.patch_embeddings.layer_norm")
    # encoder projection
    if "model.proj_ln" in name:
        name = name.replace("model.proj_ln", "projection.layernorm")
    if "model.proj_mlp.layernorms.0" in name:
        name = name.replace("model.proj_mlp.layernorms.0", "projection.projection_mlp.layernorm")
    if "model.proj_mlp.mlp_layers.0" in name:
        name = name.replace("model.proj_mlp.mlp_layers.0", "projection.projection_mlp")
    if "model.proj" in name:
        name = name.replace("model.proj", "projection.projection")
    # decoder embeddings, output bias and layernorm
    if "model.decoder.ar_decoder.Stoken_embedding" in name:
        name = name.replace("model.decoder.ar_decoder.Stoken_embedding", "decoder.embed_tokens.weight")
    if "model.decoder.ar_decoder.Sseq_pos_embedding" in name:
        name = name.replace("model.decoder.ar_decoder.Sseq_pos_embedding", "decoder.embed_positions.embeddings")
    if "model.decoder.ar_decoder.Soutp_bias" in name:
        name = name.replace("model.decoder.ar_decoder.Soutp_bias", "lm_head.bias")
    if "model.decoder.output_ln" in name:
        name = name.replace("model.decoder.output_ln", "output_layernorm")
    # decoder layers
    if "model.decoder.decoder.dec_layers" in name:
        name = name.replace("model.decoder.decoder.dec_layers", "decoder.layers")
    if "self_ln" in name:
        name = name.replace("self_ln", "self_attn_layer_norm")
    if "cross_ln" in name:
        name = name.replace("cross_ln", "encoder_attn_layer_norm")
    if "self_mha" in name:
        name = name.replace("self_mha", "self_attn")
    if "cross_mha" in name:
        name = name.replace("cross_mha", "encoder_attn")
    if "_query_dense" in name:
        name = name.replace("_query_dense", "q_proj")
    if "_key_dense" in name:
        name = name.replace("_key_dense", "k_proj")
    if "_value_dense" in name:
        name = name.replace("_value_dense", "v_proj")
    if "_output_dense" in name:
        name = name.replace("_output_dense", "out_proj")
    if "mlp.mlp_layers.0.dense1" in name:
        name = name.replace("mlp.mlp_layers.0.dense1", "fc1")
    if "mlp.mlp_layers.0.dense2" in name:
        name = name.replace("mlp.mlp_layers.0.dense2", "fc2")
    if "mlp.layernorms.0" in name:
        name = name.replace("mlp.layernorms.0", "layernorm")
    # encoder layers
    if "model.encoder.transformer_encoder.enc_layers" in name:
        name = name.replace("model.encoder.transformer_encoder.enc_layers", "encoder.layer")
    if "mha_ln" in name:
        name = name.replace("mha_ln", "layernorm_before")
    if "mha" in name:
        name = name.replace("mha", "self_attn")
    # output layer norm
    if "model.encoder.output_ln" in name:
        name = name.replace("model.encoder.output_ln", "layernorm")

    # handle qkv
    if "out_proj" in name:
        if "kernel" in name:
            # (12, 64, 768) -> (768, 768) for weights
            param = np.reshape(param, (param.shape[-1], -1))

    if "q_proj" in name or "k_proj" in name or "v_proj" in name:
        if "kernel" in name:
            # (768, 12, 64) -> (768, 768) for weights, or
            param = np.reshape(param, (param.shape[0], -1))
        elif "bias" in name:
            # (12, 64) -> (768,) for biases
            param = param.flatten()

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
    if (not name.startswith("lm_head")) and ("output_layernorm" not in name):
        name = "model." + name

    return name, param


def convert_pix2seq_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path):
    config = get_pix2seq_config(checkpoint_path)
    model = Pix2SeqForConditionalGeneration(config)
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
        name, param = rename_key(name, param)
        state_dict[name] = torch.from_numpy(param)

    # Set weights of lm head
    state_dict["lm_head.weight"] = state_dict["model.decoder.embed_tokens.weight"]

    model.load_state_dict(state_dict)
    model.eval()

    with open("/home/niels/checkpoints/pix2seq/pixel_values.npy", "rb") as f:
        pixel_values = np.load(f)
        pixel_values = torch.from_numpy(pixel_values)
        pixel_values = pixel_values.permute(0, 3, 1, 2)
        print("Shape of pixel values:", pixel_values.shape)

    prompt = torch.tensor([[10]])

    with torch.no_grad():
        outputs = model(pixel_values, decoder_input_ids=prompt)

    expected_slice = torch.tensor([[-4.3100, 2.0648, -0.2276], [-3.3209, 1.9841, 0.9853], [-3.5164, 2.3270, 0.6971]])

    encoder_last_hidden_state = outputs.encoder_last_hidden_state
    assert encoder_last_hidden_state.shape == (1, 1600, 768)
    assert torch.allclose(encoder_last_hidden_state[0, :3, :3], expected_slice, atol=1e-4)

    expected_slice_logits = torch.tensor([-8.0233, -7.5682, -7.5682])
    assert torch.allclose(outputs.logits[:, -1, :][0, :3], expected_slice_logits, atol=1e-4)

    print("Everything ok!")

    # In the original code, the max length is set to config.max_instances_per_image_test * 5 + 1,
    # with max_instances_per_image_test = 10 in the demo Colab notebook

    outputs = model.generate(
        pixel_values, max_length=51, use_cache=True, output_scores=True, return_dict_in_generate=True
    )

    print("Generated ids:", outputs.sequences)
    print("Shape of scores:", torch.cat(outputs.scores, dim=0).unsqueeze(0).shape)

    model.push_to_hub("nielsr/pix2seq-simple")


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
