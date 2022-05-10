# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for Pix2Seq."""

from typing import Optional, Union

import numpy as np

# TODO add dependency check
import torch
from PIL import Image

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ImageFeatureExtractionMixin,
    ImageInput,
    is_torch_tensor,
)
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)

BASE_VOCAB_SHIFT = 100


def dequantize(boxes, bins):
    """Dequantization of discrete tokens of coordinates in [0, bins-1]."""
    boxes = boxes.float()
    boxes = boxes / (bins - 1)
    return boxes


def seq_to_bbox(seq, quantization_bins, seq_format="yxyx_name"):
    """Returns [0, 1] normalized yxyx bbox from token sequence."""

    # [batch, 5*num_instances]
    assert seq.ndim == 2, seq.shape.as_list()
    # [batch, num_instances, 1]
    if seq_format.startswith("name"):
        ymin = torch.unsqueeze(seq[:, 1::5], -1)
        xmin = torch.unsqueeze(seq[:, 2::5], -1)
        ymax = torch.unsqueeze(seq[:, 3::5], -1)
        xmax = torch.unsqueeze(seq[:, 4::5], -1)
    else:
        ymin = torch.unsqueeze(seq[:, 0::5], -1)
        xmin = torch.unsqueeze(seq[:, 1::5], -1)
        ymax = torch.unsqueeze(seq[:, 2::5], -1)
        xmax = torch.unsqueeze(seq[:, 3::5], -1)
    if seq_format in ["name_cycxhw", "cycxhw_name"]:
        ycnt, xcnt, ysize, xsize = ymin, xmin, ymax, xmax
        ymin = ycnt - ysize // 2
        xmin = xcnt - xsize // 2
        ymax = ycnt + ysize // 2
        xmax = xcnt + xsize // 2
    quantized_box = torch.cat([ymin, xmin, ymax, xmax], dim=-1)
    quantized_box = dequantize(quantized_box, quantization_bins)
    return torch.minimum(torch.maximum(quantized_box, torch.tensor(0)), torch.tensor(1))


class Pix2SeqFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    r"""
    Constructs a Pix2Seq feature extractor.

    This feature extractor inherits from [`FeatureExtractionMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the input to a certain `size`.
        size (`int` or `Tuple(int)`, *optional*, defaults to 224):
            Resize the input to the given size. If a tuple is provided, it should be (width, height). If only an
            integer is provided, then the input will be resized to (size, size). Only has an effect if `do_resize` is
            set to `True`.
        resample (`int`, *optional*, defaults to `PIL.Image.BILINEAR`):
            An optional resampling filter. This can be one of `PIL.Image.NEAREST`, `PIL.Image.BOX`,
            `PIL.Image.BILINEAR`, `PIL.Image.HAMMING`, `PIL.Image.BICUBIC` or `PIL.Image.LANCZOS`. Only has an effect
            if `do_resize` is set to `True`.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether or not to normalize the input with mean and standard deviation.
        image_mean (`List[int]`, defaults to `[0.5, 0.5, 0.5]`):
            The sequence of means for each channel, to be used when normalizing images.
        image_std (`List[int]`, defaults to `[0.5, 0.5, 0.5]`):
            The sequence of standard deviations for each channel, to be used when normalizing images.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize=True,
        size=224,
        resample=Image.BILINEAR,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_STANDARD_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_STANDARD_STD

    def __call__(
        self, images: ImageInput, return_tensors: Optional[Union[str, TensorType]] = None, **kwargs
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several image(s).

        <Tip warning={true}>

        NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
        PIL images.

        </Tip>

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `'np'`):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model, of shape (batch_size, num_channels, height,
              width).
        """
        # Input type checking for clearer error
        valid_images = False

        # Check that images has a valid type
        if isinstance(images, (Image.Image, np.ndarray)) or is_torch_tensor(images):
            valid_images = True
        elif isinstance(images, (list, tuple)):
            if len(images) == 0 or isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]):
                valid_images = True

        if not valid_images:
            raise ValueError(
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example), "
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]

        # transformations (resizing + normalization)
        if self.do_resize and self.size is not None:
            images = [self.resize(image=image, size=self.size, resample=self.resample) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]

        # return as BatchFeature
        data = {"pixel_values": images}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs

    def decode_object_seq_to_bbox(self, logits, pred_seq, quantization_bins, coord_vocab_shift):
        """Decode objects (label & bbox) for seq from `build_response_seq_from_bbox`.

        Assume yxyxc format with truncation at the end for any uneven extra tokens. Replace class tokens with argmax
        instead of sampling.

        Args:
            logits: `float` output logits in shape of (bsz, max_seq_len, vocab_size).
            pred_seq: `int` pred sequence in shape of (bsz, max_seq_len).
            quantization_bins: `int` for bins.
            coord_vocab_shift: `int`, shifting coordinates by a specified integer.

        Returns:
            pred_class: `int` of shape (bsz, max_instances_per_image). pred_bbox: `float` of shape (bsz,
            max_instances_per_image, 4). pred_score: `float` of shape (bsz, max_instances_per_image).
        """
        _, seqlen, vocab_size = logits.shape

        if seqlen % 5 != 0:  # truncate out the last few tokens.
            pred_seq = pred_seq[..., : -(seqlen % 5)]
            logits = logits[..., : -(seqlen % 5), :]

        pred_class_p = torch.softmax(logits, dim=-1)[:, 4::5]  # (bsz, instances, vocab_size)

        print("Shape of pred_class_p:", pred_class_p.shape)

        mask_s1 = [0.0] * BASE_VOCAB_SHIFT  # reserved.
        mask_s2 = [1.0] * (coord_vocab_shift - BASE_VOCAB_SHIFT)  # labels.
        mask_s3 = [0] * (vocab_size - coord_vocab_shift)  # coordinates and others.
        mask = torch.tensor(mask_s1 + mask_s2 + mask_s3)
        pred_class = torch.argmax(pred_class_p * mask[None, None, :], -1)

        print("Shape of pred_class:", pred_class.shape)

        pred_score = torch.sum(pred_class_p * torch.nn.functional.one_hot(pred_class, vocab_size), -1)
        pred_class = torch.maximum(pred_class - BASE_VOCAB_SHIFT, torch.tensor(0))
        pred_bbox = seq_to_bbox(pred_seq - coord_vocab_shift, quantization_bins)
        
        return pred_class, pred_bbox, pred_score
