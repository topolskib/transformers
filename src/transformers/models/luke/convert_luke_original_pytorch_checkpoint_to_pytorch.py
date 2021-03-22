# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert LUKE checkpoint."""

import argparse
import json
import os

import torch

from transformers import LukeConfig, LukeEntityAwareAttentionModel, LukeTokenizer, RobertaTokenizer
from transformers.tokenization_utils_base import AddedToken


def prepare_luke_batch_inputs(tokenizer):
    # Taken from Open Entity dev set
    # Very important to put on one line!
    text = "Top seed Ana Ivanovic said on Thursday she could hardly believe her luck as a fortuitous netcord helped the new world number one avoid a humiliating second- round exit at Wimbledon ."
    span = (39, 42)

    ENTITY_TOKEN = "<ent>"
    max_mention_length = 30

    def preprocess_and_tokenize(text, start, end=None):
        target_text = text[start:end]

        # for some reason, I had to add .strip() here (otherwise there's an additional token with id 1437 added)
        return tokenizer.tokenize(target_text.strip(), add_prefix_space=True)

    tokens = [tokenizer.cls_token]
    tokens += preprocess_and_tokenize(text, 0, span[0])
    mention_start = len(tokens)
    tokens.append(ENTITY_TOKEN)
    tokens += preprocess_and_tokenize(text, span[0], span[1])
    tokens.append(ENTITY_TOKEN)
    mention_end = len(tokens)

    tokens += preprocess_and_tokenize(text, span[1])
    tokens.append(tokenizer.sep_token)

    encoding = {}
    encoding["input_ids"] = tokenizer.convert_tokens_to_ids(tokens)
    encoding["attention_mask"] = [1] * len(tokens)
    encoding["token_type_ids"] = [0] * len(tokens)

    encoding["entity_ids"] = [1]
    encoding["entity_attention_mask"] = [1]
    encoding["entity_token_type_ids"] = [0]
    entity_position_ids = list(range(mention_start, mention_end))[:max_mention_length]
    entity_position_ids += [-1] * (max_mention_length - mention_end + mention_start)
    encoding["entity_position_ids"] = [entity_position_ids]

    return encoding


@torch.no_grad()
def convert_luke_checkpoint(checkpoint_path, metadata_path, entity_vocab_path, pytorch_dump_folder_path, model_size):

    # Load configuration defined in the metadata file
    with open(metadata_path) as metadata_file:
        metadata = json.load(metadata_file)
    config = LukeConfig(**metadata["model_config"])

    # Load in the weights from the checkpoint_path
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Load the entity vocab file
    entity_vocab = load_entity_vocab(entity_vocab_path)

    tokenizer = RobertaTokenizer.from_pretrained(metadata["model_config"]["bert_model_name"])

    # Add special tokens to the token vocabulary for downstream tasks
    entity_token_1 = AddedToken("<ent>", lstrip=False, rstrip=False)
    entity_token_2 = AddedToken("<ent2>", lstrip=False, rstrip=False)
    tokenizer.add_special_tokens(dict(additional_special_tokens=[entity_token_1, entity_token_2]))
    config.vocab_size += 2

    print("Saving tokenizer to {}".format(pytorch_dump_folder_path))
    tokenizer.save_pretrained(pytorch_dump_folder_path)
    with open(os.path.join(pytorch_dump_folder_path, LukeTokenizer.vocab_files_names["entity_vocab_file"]), "w") as f:
        json.dump(entity_vocab, f)

    tokenizer = LukeTokenizer.from_pretrained(pytorch_dump_folder_path)

    # Initialize the embeddings of the special tokens
    word_emb = state_dict["embeddings.word_embeddings.weight"]
    ent_emb = word_emb[tokenizer.convert_tokens_to_ids(["@"])[0]].unsqueeze(0)
    ent2_emb = word_emb[tokenizer.convert_tokens_to_ids(["#"])[0]].unsqueeze(0)
    state_dict["embeddings.word_embeddings.weight"] = torch.cat([word_emb, ent_emb, ent2_emb])

    # Initialize the query layers of the entity-aware self-attention mechanism
    for layer_index in range(config.num_hidden_layers):
        for matrix_name in ["query.weight", "query.bias"]:
            prefix = "encoder.layer." + str(layer_index) + ".attention.self."
            state_dict[prefix + "w2e_" + matrix_name] = state_dict[prefix + matrix_name]
            state_dict[prefix + "e2w_" + matrix_name] = state_dict[prefix + matrix_name]
            state_dict[prefix + "e2e_" + matrix_name] = state_dict[prefix + matrix_name]

    # Initialize the embedding of the [MASK2] entity using that of the [MASK] entity for downstream tasks
    entity_emb = state_dict["entity_embeddings.entity_embeddings.weight"]
    entity_emb[entity_vocab["[MASK2]"]] = entity_emb[entity_vocab["[MASK]"]]

    model = LukeEntityAwareAttentionModel(config=config).eval()

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(missing_keys) == 1 and missing_keys[0] == "embeddings.position_ids"
    assert all(key.startswith("entity_predictions") or key.startswith("lm_head") for key in unexpected_keys)

    # Check outputs
    encoding = prepare_luke_batch_inputs(tokenizer)
    # convert all values to PyTorch tensors
    for key, value in encoding.items():
        encoding[key] = torch.as_tensor(encoding[key]).unsqueeze(0)

    outputs = model(**encoding)

    # Verify word hidden states
    if model_size == "large":
        expected_shape = torch.Size((1, 42, 1024))
        expected_slice = torch.tensor(
            [[0.0301, 0.0980, 0.0092], [0.2718, -0.2413, -0.9446], [-0.1382, -0.2608, -0.3927]]
        )
    else:  # base
        expected_shape = torch.Size((1, 42, 768))
        expected_slice = torch.tensor([[0.0024, 0.1318, -0.0156], [0.1413, 0.3313, -0.1206], [0.1098, 0.5391, 0.1195]])

    assert outputs.last_hidden_state.shape == expected_shape
    assert torch.allclose(outputs.last_hidden_state[0, :3, :3], expected_slice, atol=1e-4)

    # Verify entity hidden states
    if model_size == "large":
        expected_shape = torch.Size((1, 1, 1024))
        expected_slice = torch.tensor([[0.3251, 0.3981, -0.0689]])
    else:  # base
        expected_shape = torch.Size((1, 1, 768))
        expected_slice = torch.tensor([[0.2170, 0.1851, -0.0291]])

    assert outputs.entity_last_hidden_state.shape == expected_shape
    assert torch.allclose(outputs.entity_last_hidden_state[0, :3, :3], expected_slice, atol=1e-4)

    # Finally, save our PyTorch model and tokenizer
    print("Saving PyTorch model to {}".format(pytorch_dump_folder_path))
    model.save_pretrained(pytorch_dump_folder_path)


def load_entity_vocab(entity_vocab_path):
    entity_vocab = {}
    with open(entity_vocab_path, "r", encoding="utf-8") as f:
        for (index, line) in enumerate(f):
            title, _ = line.rstrip().split("\t")
            entity_vocab[title] = index

    return entity_vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--checkpoint_path", type=str, help="Path to a pytorch_model.bin file.")
    parser.add_argument(
        "--metadata_path", default=None, type=str, help="Path to a metadata.json file, defining the configuration."
    )
    parser.add_argument(
        "--entity_vocab_path",
        default=None,
        type=str,
        help="Path to an entity_vocab.tsv file, containing the entity vocabulary.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to where to dump the output PyTorch model."
    )
    parser.add_argument(
        "--model_size", default="base", type=str, choices=["base", "large"], help="Size of the model to be converted."
    )
    args = parser.parse_args()
    convert_luke_checkpoint(
        args.checkpoint_path,
        args.metadata_path,
        args.entity_vocab_path,
        args.pytorch_dump_folder_path,
        args.model_size,
    )
