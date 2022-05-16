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

import copy
import functools
import operator
from typing import Optional, Union

import numpy as np
from PIL import Image

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...image_utils import ImageFeatureExtractionMixin, ImageInput, is_torch_tensor
from ...utils import TensorType, is_torch_available, logging


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)

FAKE_CLASS_TOKEN = 30
BASE_VOCAB_SHIFT = 100


def uniform_tensor(shape, r1=0, r2=1):
    return (r1 - r2) * torch.rand(*shape) + r2


def quantize(coordinates, bins):
    """Quantization of (normalized) coordinates in [0, 1]."""
    coordinates = torch.round(coordinates * (bins - 1)).long()
    coordinates = torch.clamp(coordinates, min=0, max=bins - 1)
    return coordinates


def dequantize(boxes, bins):
    """Dequantization of discrete tokens of coordinates in [0, bins-1]."""
    boxes = boxes.float()
    boxes = boxes / (bins - 1)
    return boxes


def shape_as_list(t):
    # Assumes rank of `t` is statically known.
    shape = list(t.shape)
    dynamic_shape = t.shape
    return [shape[i] if shape[i] is not None else dynamic_shape[i] for i in range(len(shape))]


def flatten_non_batch_dims(t, out_rank):
    """Merge last few dims to have out_rank."""
    if t.ndim == out_rank:
        return t
    if t.ndim < out_rank:
        raise ValueError("Tensor has rank %d. Expected at least %d" % (t.ndim, out_rank))
    shape_list = shape_as_list(t)
    split = out_rank - 1
    inner_dims = shape_list[:split]
    new_last_dim = functools.reduce(operator.mul, shape_list[split:])
    out_shape = inner_dims + [new_last_dim]
    return torch.reshape(t, out_shape)


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
        size (`int` or `Tuple(int)`, *optional*, defaults to 640):
            Resize the input to the given size. If a tuple is provided, it should be (width, height). If only an
            integer is provided, then the input will be resized to (size, size). Only has an effect if `do_resize` is
            set to `True`.
        resample (`int`, *optional*, defaults to `PIL.Image.BILINEAR`):
            An optional resampling filter. This can be one of `PIL.Image.NEAREST`, `PIL.Image.BOX`,
            `PIL.Image.BILINEAR`, `PIL.Image.HAMMING`, `PIL.Image.BICUBIC` or `PIL.Image.LANCZOS`. Only has an effect
            if `do_resize` is set to `True`.
        do_rescale (`bool`, *optional*, defaults to `True`):
             Whether or not to apply the scaling factor (to make pixel values floats between 0. and 1.).
        quantization_bins (`int`, *optional*, defaults to `1000`):
            Bins.
        coord_vocab_shift (`int`, *optional*, defaults to `1000`):
            Shift coordinates by this number.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize=True,
        size=640,
        resample=Image.BILINEAR,
        do_rescale=True,
        quantization_bins=1000,
        coord_vocab_shift=1000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.quantization_bins = quantization_bins
        self.coord_vocab_shift = coord_vocab_shift

    def _resize(self, image, size_longer_side, resample=Image.BILINEAR):
        """Resize the image while preserving aspect ratio.

        Args:
            image: image.
            size_longer_side: size for the longer side of the image.
            resample: resampling method.

        Returns:
            resized image.
        """
        if not isinstance(image, Image.Image):
            image = self.to_pil_image(image)

        w, h = image.size

        if w > h:
            original_width = size_longer_side
            original_height = int(size_longer_side * h / w)

        else:
            original_height = size_longer_side
            original_width = int(size_longer_side * w / h)

        return self.resize(image, (original_height, original_width), resample=resample)

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

        # transformations (resizing + rescaling)
        if self.do_resize and self.size is not None:
            # images = [
            #     self._resize(image=image, size_longer_side=self.size, resample=self.resample) for image in images
            # ]
            images = [self.resize(image=image, size=self.size, resample=self.resample) for image in images]
        if self.do_rescale:
            images = [self.to_numpy_array(image=image) for image in images]

        # return as BatchFeature
        data = {"pixel_values": images}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs

    def build_response_seq_from_bbox(self, bbox, label, noise_bbox_weight=1.0, class_label_corruption="rand_cls"):
        """
        Build target seq from bounding bboxes for object detection. Objects are serialized using the format of yxyxc.

        Args:
            bbox: `float` bounding box of shape (bsz, n, 4).
            label: `int` label of shape (bsz, n).
            quantization_bins: `int`.
            noise_bbox_weight: `float` on the token weights for noise bboxes.
            coord_vocab_shift: `int`, shifting coordinates by a specified integer.
            class_label_corruption: `string` specifying how labels are corrupted for the
            input_seq.

        Returns:
            discrete sequences with shape (bsz, seqlen).
        """
        # Bbox and label quantization.
        is_padding = torch.unsqueeze(torch.eq(label, 0), -1)
        quantized_bbox = quantize(bbox, self.quantization_bins)
        quantized_bbox = quantized_bbox + self.coord_vocab_shift
        quantized_bbox = torch.where(is_padding, torch.zeros_like(quantized_bbox), quantized_bbox)
        new_label = torch.unsqueeze(label + BASE_VOCAB_SHIFT, -1)
        new_label = torch.where(is_padding, torch.zeros_like(new_label), new_label)
        lb_shape = list(new_label.shape)

        # Bbox and label serialization.
        response_seq = torch.cat([quantized_bbox, new_label], dim=-1)
        response_seq = flatten_non_batch_dims(response_seq, 2)
        rand_cls = BASE_VOCAB_SHIFT + uniform_tensor(lb_shape, 0, self.coord_vocab_shift - BASE_VOCAB_SHIFT).type(
            new_label.dtype
        )
        fake_cls = FAKE_CLASS_TOKEN + torch.zeros_like(new_label)
        rand_n_fake_cls = torch.where(uniform_tensor(lb_shape) > 0.5, rand_cls, fake_cls)
        real_n_fake_cls = torch.where(uniform_tensor(lb_shape) > 0.5, new_label, fake_cls)
        real_n_rand_n_fake_cls = torch.where(uniform_tensor(lb_shape) > 0.5, new_label, rand_n_fake_cls)
        label_mapping = {
            "none": new_label,
            "rand_cls": rand_cls,
            "real_n_fake_cls": real_n_fake_cls,
            "rand_n_fake_cls": rand_n_fake_cls,
            "real_n_rand_n_fake_cls": real_n_rand_n_fake_cls,
        }
        new_label_m = label_mapping[class_label_corruption]
        new_label_m = torch.where(is_padding, torch.zeros_like(new_label_m), new_label_m)
        response_seq_class_m = torch.cat([quantized_bbox, new_label_m], dim=-1)
        response_seq_class_m = flatten_non_batch_dims(response_seq_class_m, 2)

        # Get token weights.
        is_real = torch.ne(new_label, FAKE_CLASS_TOKEN).float()
        bbox_weight = is_real.repeat([1, 1, 4])
        label_weight = is_real + (1.0 - is_real) * noise_bbox_weight
        token_weights = torch.cat([bbox_weight, label_weight], -1)
        token_weights = flatten_non_batch_dims(token_weights, 2)

        return response_seq, response_seq_class_m, token_weights

    def pad_to_max_len(self, data, max_len, dim):
        """Pad the data tensor to max length on dim."""

        shape = shape_as_list(data)
        padding_shape, new_shape = copy.copy(shape), copy.copy(shape)
        padding_shape[dim] = max_len - padding_shape[dim]
        new_shape[dim] = max_len
        paddings = torch.zeros(padding_shape, dtype=data.dtype)
        return torch.reshape(torch.cat([data, paddings], axis=dim), new_shape)

    def decode_object_seq_to_bbox(self, logits, pred_seq):
        """Decode objects (label & bbox) for seq from `build_response_seq_from_bbox`.

        Assume yxyxc format with truncation at the end for any uneven extra tokens. Replace class tokens with argmax
        instead of sampling.

        Args:
            logits: `float` output logits in shape of (bsz, max_seq_len, vocab_size).
            pred_seq: `int` pred sequence in shape of (bsz, max_seq_len).

        Returns:
            pred_class: `int` of shape (bsz, max_instances_per_image). pred_bbox: `float` of shape (bsz,
            max_instances_per_image, 4). pred_score: `float` of shape (bsz, max_instances_per_image).
        """
        _, seqlen, vocab_size = logits.shape

        if seqlen % 5 != 0:  # truncate out the last few tokens.
            pred_seq = pred_seq[..., : -(seqlen % 5)]
            logits = logits[..., : -(seqlen % 5), :]

        pred_class_p = torch.softmax(logits, dim=-1)[:, 4::5]  # (bsz, instances, vocab_size)

        mask_s1 = [0.0] * BASE_VOCAB_SHIFT  # reserved.
        mask_s2 = [1.0] * (self.coord_vocab_shift - BASE_VOCAB_SHIFT)  # labels.
        mask_s3 = [0] * (vocab_size - self.coord_vocab_shift)  # coordinates and others.
        mask = torch.tensor(mask_s1 + mask_s2 + mask_s3)
        pred_class = torch.argmax(pred_class_p * mask[None, None, :], -1)

        pred_score = torch.sum(pred_class_p * torch.nn.functional.one_hot(pred_class, vocab_size), -1)
        pred_class = torch.maximum(pred_class - BASE_VOCAB_SHIFT, torch.tensor(0))
        pred_bbox = seq_to_bbox(pred_seq - self.coord_vocab_shift, self.quantization_bins)

        return pred_class, pred_bbox, pred_score
