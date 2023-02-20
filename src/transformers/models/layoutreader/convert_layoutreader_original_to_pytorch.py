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
"""Convert LayoutReader checkpoints from the original repository.

URL: https://github.com/microsoft/unilm/tree/master/layoutreader"""


import argparse

import torch
from huggingface_hub import hf_hub_download

from transformers import LayoutReaderConfig, LayoutReaderForSeq2SeqDecoding


def rename_key(name):
    if "bert" in name:
        name = name.replace("bert", "layoutreader")
    if "cls.predictions" in name:
        name = name.replace("cls.predictions", "cls")

    return name


def convert_state_dict(orig_state_dict):
    for key in orig_state_dict.copy().keys():
        val = orig_state_dict.pop(key)
        orig_state_dict[rename_key(key)] = val

    return orig_state_dict


def convert_layoutreader_checkpoint(checkpoint_path, pytorch_dump_folder_path, push_to_hub=False):
    original_state_dict = torch.load(checkpoint_path, map_location="cpu")
    new_state_dict = convert_state_dict(original_state_dict)

    config = LayoutReaderConfig()
    model = LayoutReaderForSeq2SeqDecoding(config)
    model.eval()
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    assert missing_keys == []
    assert unexpected_keys == ["crit_mask_lm_smoothed.one_hot"]

    # Generate
    filepath = hf_hub_download(repo_id="nielsr/layoutreader-dummy-data", repo_type="dataset", filename="batch.pt")
    batch = torch.load(filepath, map_location="cpu")
    input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = tuple(batch)

    model.generate(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        attention_mask=input_mask,
        task_idx=task_idx,
        mask_qkv=mask_qkv,
    )

    # TODO assert values
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    # feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/{}".format(swin_name.replace("_", "-")))
    # image = Image.open(requests.get(url, stream=True).raw)
    # inputs = feature_extractor(images=image, return_tensors="pt")

    # timm_outs = timm_model(inputs["pixel_values"])
    # hf_outs = model(**inputs).logits

    # assert torch.allclose(timm_outs, hf_outs, atol=1e-3)

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and tokenizer to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        # tokenizer.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Saving model and tokenizer to the hub")
        model.push_to_hub("nielsr/layoutreader-readingbank")
        # tokenizer.push_to_hub(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_path",
        default="/Users/nielsrogge/Documents/LayoutReader/pytorch_model.bin",
        type=str,
        help="Path to the original PyTorch checkpoint.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_layoutreader_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
