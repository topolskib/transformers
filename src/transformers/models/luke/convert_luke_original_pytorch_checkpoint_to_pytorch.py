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
import os
from pathlib import Path

import torch
from packaging import version

from ...utils import logging
from . import LukeConfig, LukeEntityAwareAttentionModel


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


@torch.no_grad()
def convert_luke_checkpoint(checkpoint_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    # First, define our model 
    config = LukeConfig()
    model = LukeEntityAwareAttentionModel(config=config)

    # Second, load in the weights from the checkpoint_path
    model.load_state_dict(torch.load(checkpoint_path))
    
    # Finally, save our PyTorch model
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "checkpoint_path", type=str, help="path to a pytorch_model.bin on local filesystem."
    )
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    convert_luke_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path)