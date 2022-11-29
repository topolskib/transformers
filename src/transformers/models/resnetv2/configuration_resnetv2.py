# coding=utf-8
# Copyright 2022 Microsoft Research, Inc. and The HuggingFace Inc. team. All rights reserved.
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
""" ResNetv2 model configuration"""

from collections import OrderedDict
from typing import Mapping

from packaging import version

from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging


logger = logging.get_logger(__name__)

RESNETV2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/resnetv2-50": "https://huggingface.co/google/resnetv2-50/resolve/main/config.json",
}


class ResNetv2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ResNetv2Model`]. It is used to instantiate an
    ResNetv2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the ResNetv2
    [google/resnetnv2-50](https://huggingface.co/google/resnetnv2-50) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        embedding_size (`int`, *optional*, defaults to 64):
            Dimensionality (hidden size) for the embedding layer.
        hidden_sizes (`List[int]`, *optional*, defaults to `[256, 512, 1024, 2048]`):
            Dimensionality (hidden size) at each stage.
        depths (`List[int]`, *optional*, defaults to `[3, 4, 6, 3]`):
            Depth (number of layers) for each stage.
        layer_type (`str`, *optional*, defaults to `"preact"`):
            The layer to use, it can be either `"preact"` or `"bottleneck"`.
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in each block. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"`
            are supported.
        downsample_in_first_stage (`bool`, *optional*, defaults to `False`):
            If `True`, the first stage will downsample the inputs using a `stride` of 2.
        drop_path_rate (`float`, *optional*, defaults to 0.0):
            The drop path rate for the stochastic depth.
        output_stride (`int`, *optional*, defaults to 32):
            The output stride of the model.
        width_factor (`int`, *optional*, defaults to 1):
            The width factor for the model.

    Example:
    ```python
    >>> from transformers import ResNetv2Config, ResNetv2Model

    >>> # Initializing a ResNetv2 resnetv2-50 style configuration
    >>> configuration = ResNetv2Config()

    >>> # Initializing a model (with random weights) from the resnetv2-50 style configuration
    >>> model = ResNetv2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    model_type = "resnetv2"
    layer_types = ["preact", "bottleneck"]

    def __init__(
        self,
        num_channels=3,
        embedding_size=64,
        hidden_sizes=[256, 512, 1024, 2048],
        depths=[3, 4, 6, 3],
        layer_type="preact",
        hidden_act="relu",
        downsample_in_first_stage=False,
        drop_path_rate=0.0,
        output_stride=32,
        width_factor=1,
        **kwargs
    ):
        super().__init__(**kwargs)
        if layer_type not in self.layer_types:
            raise ValueError(f"layer_type={layer_type} is not one of {','.join(self.layer_types)}")
        self.num_channels = num_channels
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.layer_type = layer_type
        self.hidden_act = hidden_act
        self.downsample_in_first_stage = downsample_in_first_stage
        self.drop_path_rate = drop_path_rate
        self.output_stride = output_stride
        self.width_factor = width_factor


class ResNetv2OnnxConfig(OnnxConfig):

    torch_onnx_minimum_version = version.parse("1.11")

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
            ]
        )

    @property
    def atol_for_validation(self) -> float:
        return 1e-3
