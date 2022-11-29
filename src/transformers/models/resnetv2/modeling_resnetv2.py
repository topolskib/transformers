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
""" PyTorch ResNetv2 model."""

from functools import partial
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithNoAttention,
    BaseModelOutputWithPoolingAndNoAttention,
    ImageClassifierOutputWithNoAttention,
)
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_resnetv2 import ResNetv2Config


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ResNetv2Config"
_FEAT_EXTRACTOR_FOR_DOC = "AutoFeatureExtractor"

# Base docstring
_CHECKPOINT_FOR_DOC = "google/resnetnv2-50"
_EXPECTED_OUTPUT_SHAPE = [1, 2048, 7, 7]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "google/resnetnv2-50"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tiger cat"

RESNETV2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/resnetnv2-50",
    # See all ResNetv2 models at https://huggingface.co/models?filter=resnetv2
]


def _num_groups(num_channels, num_groups, group_size):
    if group_size:
        assert num_channels % group_size == 0
        return num_channels // group_size
    return num_groups


def get_padding_value(padding, kernel_size, **kwargs) -> Tuple[Tuple, bool]:
    dynamic = False
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == "same":
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = get_padding(kernel_size, **kwargs)
            else:
                # dynamic 'SAME' padding, has runtime/GPU memory overhead
                padding = 0
                dynamic = True
        elif padding == "valid":
            # 'VALID' padding, same as padding=0
            padding = 0
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = get_padding(kernel_size, **kwargs)
    return padding, dynamic


def create_conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop("padding", "")
    kwargs.setdefault("bias", False)
    padding, is_dynamic = get_padding_value(padding, kernel_size, **kwargs)
    if is_dynamic:
        return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
    else:
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)


class BatchNormAct2d(nn.BatchNorm2d):
    """BatchNorm + Activation

    This module performs BatchNorm + Activation in a manner that will remain backwards
    compatible with weights trained with separate bn, act. This is why we inherit from BN
    instead of composing it as a .bn member.
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        apply_act=True,
        act_layer=nn.ReLU,
        inplace=True,
        drop_layer=None,
        device=None,
        dtype=None,
    ):
        try:
            factory_kwargs = {"device": device, "dtype": dtype}
            super(BatchNormAct2d, self).__init__(
                num_features,
                eps=eps,
                momentum=momentum,
                affine=affine,
                track_running_stats=track_running_stats,
                **factory_kwargs,
            )
        except TypeError:
            # NOTE for backwards compat with old PyTorch w/o factory device/dtype support
            super(BatchNormAct2d, self).__init__(
                num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats
            )
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        # act_layer = get_act_layer(act_layer)  # string -> nn.Module
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        # cut & paste of torch.nn.BatchNorm2d.forward impl to avoid issues with torchscript and tracing
        assert x.ndim == 4, f"expected 4D input (got {x.ndim}D input)"

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        x = nn.functional.batch_norm(
            x,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
        x = self.drop(x)
        x = self.act(x)
        return x


class ResNetv2GroupNormActivation(nn.GroupNorm):
    # NOTE num_channel and num_groups order flipped for easier layer swaps / binding of fixed args
    def __init__(
        self,
        num_channels,
        num_groups=32,
        eps=1e-5,
        affine=True,
        group_size=None,
        apply_act=True,
        act_layer=nn.ReLU,
        inplace=True,
        drop_layer=None,
    ):
        super(ResNetv2GroupNormActivation, self).__init__(
            _num_groups(num_channels, num_groups, group_size), num_channels, eps=eps, affine=affine
        )
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        # act_layer = get_act_layer(act_layer)  # string -> nn.Module
        if act_layer is not None and apply_act:
            act_args = dict(inplace=True) if inplace else {}
            self.act = act_layer(**act_args)
        else:
            self.act = nn.Identity()
        self._fast_norm = False  # TODO add support for fast norm

    def forward(self, x):
        # if self._fast_norm:
        #     x = fast_group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        # else:
        x = nn.functional.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        x = self.drop(x)
        x = self.act(x)
        return x


# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class StdConv2d(nn.Conv2d):
    """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """

    def __init__(
        self, in_channel, out_channels, kernel_size, stride=1, padding=None, dilation=1, groups=1, bias=False, eps=1e-6
    ):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channel,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.eps = eps

    def forward(self, x):
        weight = nn.functional.batch_norm(
            self.weight.reshape(1, self.out_channels, -1), None, None, training=True, momentum=0.0, eps=self.eps
        ).reshape_as(self.weight)
        x = nn.functional.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class ResNetv2Embeddings(nn.Module):
    """
    ResNetv2 Embeddings (stem) composed of a single aggressive convolution.
    """

    def __init__(self, config: ResNetv2Config):
        super().__init__()
        self.convolution = nn.Conv2d(
            config.num_channels, config.embedding_size, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.pooler = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.num_channels = config.num_channels

    def forward(self, pixel_values: Tensor) -> Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )

        embedding = self.convolution(pixel_values)
        embedding = self.pooler(embedding)

        return embedding


# Copied from transformers.models.convnext.modeling_convnext.drop_path
def drop_path(input, drop_prob: float = 0.0, training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


# Copied from transformers.models.convnext.modeling_convnext.ConvNextDropPath
class ResNetv2DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


def make_div(v, divisor=8):
    min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ResNetv2PreActivationBottleneckLayer(nn.Module):
    """Pre-activation (v2) bottleneck block.
    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        bottle_ratio=0.25,
        stride=1,
        dilation=1,
        first_dilation=None,
        groups=1,
        act_layer=None,
        conv_layer=None,
        norm_layer=None,
        proj_layer=None,
        drop_path_rate=0.0,
    ):
        super().__init__()

        first_dilation = first_dilation or dilation
        conv_layer = conv_layer or StdConv2d
        norm_layer = norm_layer or partial(ResNetv2GroupNormActivation, num_groups=32)
        out_channels = out_channels or in_channels
        mid_channels = make_div(out_channels * bottle_ratio)

        if proj_layer is not None:
            self.downsample = proj_layer(
                in_channels,
                out_channels,
                stride=stride,
                dilation=dilation,
                first_dilation=first_dilation,
                preact=True,
                conv_layer=conv_layer,
                norm_layer=norm_layer,
            )
        else:
            self.downsample = None

        self.norm1 = norm_layer(in_channels)
        self.conv1 = conv_layer(in_channels, mid_channels, 1)
        self.norm2 = norm_layer(mid_channels)
        self.conv2 = conv_layer(mid_channels, mid_channels, 3, stride=stride, dilation=first_dilation, groups=groups)
        self.norm3 = norm_layer(mid_channels)
        self.conv3 = conv_layer(mid_channels, out_channels, 1)
        self.drop_path = ResNetv2DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x, print_values=False):
        x_preact = self.norm1(x)

        if print_values:
            print("Hidden states after first norm:", x_preact[0, 0, :3, :3])

        # shortcut branch
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x_preact, print_values)

        if print_values:
            print("Hidden states after downsample:", shortcut[0, 0, :3, :3])

        # residual branch
        x = self.conv1(x_preact)
        x = self.conv2(self.norm2(x))
        x = self.conv3(self.norm3(x))
        x = self.drop_path(x)
        return x + shortcut


class ResNetv2DownsampleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        dilation=1,
        first_dilation=None,
        preact=True,
        conv_layer=None,
        norm_layer=None,
    ):
        super(ResNetv2DownsampleConv, self).__init__()
        self.conv_layer = conv_layer
        self.conv = conv_layer(in_channels, out_channels, 1, stride=stride)
        self.norm = nn.Identity() if preact else norm_layer(out_channels, apply_act=False)

    def forward(self, x, print_values=False):
        if print_values:
            print("Conv layer:", self.conv_layer)
            print("Hidden states before downsample conv:", x[0, 0, :3, :3])

        z = self.conv(x)

        if print_values:
            print("Hidden states after downsample conv:", z[0, 0, :3, :3])

        return self.norm(self.conv(x))


class ResNetv2ConvLayer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: str = "relu"
    ):
        super().__init__()
        self.convolution = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False
        )
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = ACT2FN[activation] if activation is not None else nn.Identity()

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class ResNetv2ShortCut(nn.Module):
    """
    ResNet shortcut, used to project the residual features to the correct size. If needed, it is also used to
    downsample the input using `stride=2`.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.normalization = nn.BatchNorm2d(out_channels)

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        return hidden_state


class ResNetv2BottleNeckLayer(nn.Module):
    """
    A classic ResNet's bottleneck layer composed by three `3x3` convolutions.

    The first `1x1` convolution reduces the input by a factor of `reduction` in order to make the second `3x3`
    convolution faster. The last `1x1` convolution remaps the reduced features to `out_channels`.
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, activation: str = "relu", reduction: int = 4
    ):
        super().__init__()
        should_apply_shortcut = in_channels != out_channels or stride != 1
        reduces_channels = out_channels // reduction
        self.shortcut = (
            ResNetv2ShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else nn.Identity()
        )
        self.layer = nn.Sequential(
            ResNetv2ConvLayer(in_channels, reduces_channels, kernel_size=1),
            ResNetv2ConvLayer(reduces_channels, reduces_channels, stride=stride),
            ResNetv2ConvLayer(reduces_channels, out_channels, kernel_size=1, activation=None),
        )
        self.activation = ACT2FN[activation]

    def forward(self, hidden_state):
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


class ResNetv2Stage(nn.Module):
    """
    A ResNet v2 stage composed by stacked layers.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        dilation,
        depth,
        bottle_ratio=0.25,
        groups=1,
        avg_down=False,
        layer_dpr=None,
        layer_fn=ResNetv2PreActivationBottleneckLayer,
        act_layer=None,
        conv_layer=None,
        norm_layer=None,
        **layer_kwargs
    ):
        super().__init__()

        first_dilation = 1 if dilation in (1, 2) else 2
        layer_kwargs = dict(act_layer=act_layer, conv_layer=conv_layer, norm_layer=norm_layer)
        if avg_down:
            # TODO add support for avg_down
            raise NotImplementedError("avg_down is not implemented")
        proj_layer = ResNetv2DownsampleConv
        prev_chs = in_channels
        self.layers = nn.Sequential()
        for layer_idx in range(depth):
            drop_path_rate = layer_dpr[layer_idx] if layer_dpr else 0.0
            stride = stride if layer_idx == 0 else 1
            self.layers.add_module(
                str(layer_idx),
                layer_fn(
                    prev_chs,
                    out_channels,
                    stride=stride,
                    dilation=dilation,
                    bottle_ratio=bottle_ratio,
                    groups=groups,
                    first_dilation=first_dilation,
                    proj_layer=proj_layer,
                    drop_path_rate=drop_path_rate,
                    **layer_kwargs,
                ),
            )
            prev_chs = out_channels
            first_dilation = dilation
            proj_layer = None

    def forward(self, input: Tensor, print_values=False) -> Tensor:
        hidden_state = input
        for idx, layer in enumerate(self.layers):
            if idx == 0 and print_values:
                print(f"Hidden states before block {idx}", hidden_state[0, 0, :3, :3])
            hidden_state = layer(hidden_state, print_values=idx == 0)
            if idx == 0 and print_values:
                print(f"Hidden states after block {idx}", hidden_state[0, 0, :3, :3])
        return hidden_state


class ResNetv2Encoder(nn.Module):
    def __init__(
        self,
        config: ResNetv2Config,
        act_layer=nn.ReLU,
        conv_layer=create_conv2d_pad,
        norm_layer=BatchNormAct2d,
    ):
        super().__init__()
        self.stages = nn.ModuleList([])

        prev_chs = config.embedding_size
        curr_stride = 4
        dilation = 1
        block_dprs = [
            x.tolist() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths)).split(config.depths)
        ]
        block_fn = ResNetv2PreActivationBottleneckLayer if config.layer_type == "preact" else ResNetv2BottleNeckLayer
        for stage_idx, (d, c, bdpr) in enumerate(zip(config.depths, config.hidden_sizes, block_dprs)):
            out_chs = make_div(c * config.width_factor)
            stride = 1 if stage_idx == 0 else 2
            if curr_stride >= config.output_stride:
                dilation *= stride
                stride = 1
            stage = ResNetv2Stage(
                prev_chs,
                out_chs,
                stride=stride,
                dilation=dilation,
                depth=d,
                avg_down=False,
                act_layer=act_layer,
                conv_layer=conv_layer,
                norm_layer=norm_layer,
                block_dpr=bdpr,
                block_fn=block_fn,
            )
            prev_chs = out_chs
            curr_stride *= stride
            self.stages.add_module(str(stage_idx), stage)

    def forward(
        self, hidden_state: Tensor, output_hidden_states: bool = False, return_dict: bool = True
    ) -> BaseModelOutputWithNoAttention:
        hidden_states = () if output_hidden_states else None

        for idx, stage_module in enumerate(self.stages):
            if output_hidden_states:
                hidden_states = hidden_states + (hidden_state,)

            hidden_state = stage_module(hidden_state, print_values=idx == 0)

            print(f"Hidden states after stage {idx}: ", hidden_state.shape)
            print(f"Hidden states after stage {idx}: ", hidden_state[0, 0, :3, :3])

        if output_hidden_states:
            hidden_states = hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(v for v in [hidden_state, hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=hidden_state,
            hidden_states=hidden_states,
        )


# Copied from transformers.models.resnet.modeling_resnet.ResNetPreTrainedModel with ResNet->ResNetv2,resnet->resnetv2
class ResNetv2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ResNetv2Config
    base_model_prefix = "resnetv2"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ResNetv2Model):
            module.gradient_checkpointing = value


RESNETV2_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ResNetv2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

RESNETV2_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoFeatureExtractor`]. See
            [`AutoFeatureExtractor.__call__`] for details.

        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare ResNetv2 model outputting raw features without any specific head on top.",
    RESNETV2_START_DOCSTRING,
)
class ResNetv2Model(ResNetv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedder = ResNetv2Embeddings(config)

        self.encoder = ResNetv2Encoder(config)
        self.norm = BatchNormAct2d(config.hidden_sizes[-1])

        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(RESNETV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndNoAttention,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self, pixel_values: Tensor, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None
    ) -> BaseModelOutputWithPoolingAndNoAttention:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        embedding_output = self.embedder(pixel_values)

        print("Shape of embeddings:", embedding_output.shape)
        print("First values of embeddings:", embedding_output[0, 0, :3, :3])

        encoder_outputs = self.encoder(
            embedding_output, output_hidden_states=output_hidden_states, return_dict=return_dict
        )

        last_hidden_state = encoder_outputs[0]

        last_hidden_state = self.norm(last_hidden_state)

        print("Shape of final embeddings:", last_hidden_state.shape)
        print("Final embeddings:", last_hidden_state[0, 0, :3, :3])

        pooled_output = self.pooler(last_hidden_state)

        print("Pooled output:", pooled_output.shape)
        print("Pool output:", pooled_output[0, 0, :3, :3])

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


@add_start_docstrings(
    """
    ResNetv2 Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for
    ImageNet.
    """,
    RESNETV2_START_DOCSTRING,
)
class ResNetv2ForImageClassification(ResNetv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.resnetv2 = ResNetv2Model(config)
        # classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.hidden_sizes[-1], config.num_labels) if config.num_labels > 0 else nn.Identity(),
        )
        # initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(RESNETV2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_FEAT_EXTRACTOR_FOR_DOC,
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutputWithNoAttention,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ImageClassifierOutputWithNoAttention:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.resnetv2(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooled_output)

        loss = None

        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return (loss,) + output if loss is not None else output

        return ImageClassifierOutputWithNoAttention(loss=loss, logits=logits, hidden_states=outputs.hidden_states)
