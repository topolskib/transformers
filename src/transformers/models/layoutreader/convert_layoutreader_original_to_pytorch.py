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
    input_ids, token_type_ids, position_ids, input_mask, mask_qkv, _ = tuple(batch)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids[:, :, 0],
            bbox=input_ids[:, :, 1:],
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=input_mask,
            mask_qkv=mask_qkv,
        )

    # assert values
    # fmt: off
    expected_output_ids = [  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
          15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  62,  72,  73,  74,
          25,  26,  27,  28,  29,  44,  45,  46,  47,  48,  63,  64,  65,  66,
          67,  75,  76,  77,  78,  79,  93,  94,  95,  96,  97, 103, 104, 105,
         106, 107,  30,  31,  32,  33,  34,  49,  50,  51,  52,  53,  54,  55,
          56,  57,  68,  69,  80,  81,  82,  83,  84,  85,  86,  87,  88,  98,
          99,  35,  58,  70,  89, 100,  36,  59,  71,  90, 101,  37,  38,  39,
          40,  41,  42,  43,  60,  61,  91,  92, 102, 156, 157, 176, 177, 178,
         109, 110, 111, 126, 127, 128, 129, 130, 140, 141, 142, 143, 144, 158,
         159, 160, 161, 108, 112, 113, 131, 132, 145, 146, 147, 148, 149, 162,
         163, 164, 165, 166, 167, 168, 169, 179, 122, 123, 125, 133, 150, 170,
         114, 115, 124, 134, 151, 171, 116, 135, 136, 137, 152, 153, 172, 180,
         117, 118, 119, 120, 121, 138, 139, 154, 155, 173, 174, 175, 181, 182,
         280, 281, 300, 301, 302, 183, 184, 185, 186, 187, 196, 197, 198, 199,
         217, 218, 219, 220, 221, 233, 234, 235, 236, 237, 244, 245, 246, 247,
         248, 253, 254, 255, 269, 270, 271, 272, 273, 282, 283, 284, 285, 286,
         303, 304, 305, 306, 307, 315, 316, 317, 318, 319, 345, 346, 347, 348,
         355, 356, 357, 188, 189, 190, 191, 192, 193, 194, 195, 205, 206, 207,
         208, 209, 211, 212, 213, 214, 215, 216, 227, 228, 229, 230, 231, 232,
         243, 251, 252, 278, 279, 292, 293, 294, 295, 296, 297, 298, 299, 308,
         309, 320, 321, 322, 323, 324, 325, 326, 333, 334, 335, 336, 337, 338,
         339, 340, 341, 342, 343, 344, 353, 354, 200, 250, 264, 265, 268, 290,
         291, 327, 351, 352, 364, 201, 222, 238, 249, 256, 257, 267, 274, 287,
         328, 349, 350, 365, 210, 226, 242, 258, 266, 275, 288, 329, 358, 202,
         203, 204, 223, 224, 225, 239, 240, 241, 259, 260, 261, 262, 263, 276,
         277, 289, 330, 331, 332, 359, 360, 361, 362, 363, 310, 311, 312, 313,
         314, 366, 367, 368, 387, 369, 370, 371, 372, 373, 374, 381, 375, 376,
         377, 378, 379, 380, 382, 383, 384, 385, 386, 388, 389, 390, 391, 392,
         393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 210, 266,
         275, 288, 329, 180, 266, 266, 266, 266, 266, 266, 266, 266, 266, 266,
         266, 266, 266, 266, 266, 266, 266,  36, 180, 266, 402, 403, 404,  36,
          37,  38,  39,  40,  41,  42,  43,  60,  61, 402, 180, 266, 329, 330,
         331, 332, 242, 266, 266, 266, 266, 266, 266, 266, 266, 266, 266, 266,
         266, 266, 266, 116, 116, 116, 210, 266, 242, 242, 242, 242, 242, 242,
         242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242,
         242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242, 242,
         242, 242, 242, 242, 242, 242, 242]
    # fmt: on

    assert expected_output_ids == output_ids[0].tolist()
    print("Looks ok!")

    # try beam search
    model.config.beam_size = 5
    with torch.no_grad():
        output_ids = model.beam_search(
            input_ids=input_ids[:, :, 0],
            bbox=input_ids[:, :, 1:],
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=input_mask,
            mask_qkv=mask_qkv,
        )
        print("Output ids after beam search:", output_ids)

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
