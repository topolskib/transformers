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
"""Image processor class for MaskFormer."""

from typing import Dict, List, Optional, Union

import numpy as np

from transformers.utils.generic import TensorType

from ...image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from ...image_transforms import normalize, resize, to_channel_dimension_format
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    is_batched,
    to_numpy_array,
    valid_images,
)
from ...utils import logging


logger = logging.get_logger(__name__)


class MaskFormerImageProcessor(BaseImageProcessor):
    r"""
    Constructs a MaskFormer image processor.

    Args:
        ...
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_reduce_label: bool = True,
        do_pad: bool = True,
        size_divisibility: int = 512,
        do_create_binary_mask: bool = True,
        ignore_label: int = 0, # TODO verify this
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.do_reduce_label = do_reduce_label
        self.do_pad = pad
        self.size_divisibility = size_divisibility
        self.do_create_binary_mask = do_create_binary_mask
        self.ignore_label = ignore_label

    def reduce_label(segmentation_map):
        # TODO input numpy, output numpy?
        image = np.asarray(Image.open(segmentation_map))
        
        assert image.dtype == np.uint8
        
        image = image - 1  # 0 (ignore) becomes 255. others are shifted by 1
        
        return Image.fromarray(image)
    
    def pad(image, seg_seg_gt):
        # TODO input numpy, output numpy?
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if segmentation_map is not None:
            segmentation_map = torch.as_tensor(segmentation_map.astype("long"))

        if self.size_divisibility > 0:
            image_size = (image.shape[-2], image.shape[-1])
            padding_size = [
                0,
                self.size_divisibility - image_size[1],
                0,
                self.size_divisibility - image_size[0],
            ]
            image = F.pad(image, padding_size, value=128).contiguous()
            if segmentation_map is not None:
                segmentation_map = F.pad(segmentation_map, padding_size, value=self.ignore_label).contiguous()

        return image, segmentation_map

    def create_binary_mask(segmentation_map, ignore_label):
        # TODO input numpy, output numpy?
        classes = np.unique(segmentation_map)
        # remove ignored region
        classes = classes[classes != self.ignore_label]
        gt_classes = np.array(classes, dtype=np.int64)

        masks = []
        for class_id in classes:
            masks.append(segmentation_map == class_id)

        if len(masks) == 0:
            # Some images don't have annotations (all ignored)
            gt_masks = np.zeros((0, segmentation_map.shape[-2], segmentation_map.shape[-1]))
        else:
            gt_masks = np.stack([masks])

        return gt_classes, gt_masks

    def normalize(
        self,
        image: np.ndarray,
        mean: Union[float, List[float]],
        std: Union[float, List[float]],
        data_format: Optional[Union[str, ChannelDimension]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Normalize an image. image = (image - image_mean) / image_std.

        Args:
            image (`np.ndarray`):
                Image to normalize.
            mean (`float` or `List[float]`):
                Image mean to use for normalization.
            std (`float` or `List[float]`):
                Image standard deviation to use for normalization.
            data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.

        Returns:
            `np.ndarray`: The normalized image.
        """
        return normalize(image, mean=mean, std=std, data_format=data_format, **kwargs)

    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput = None,
        do_resize: Optional[bool] = None,
        size: Dict[str, int] = None,
        resample: PILImageResampling = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Union[str, ChannelDimension] = ChannelDimension.FIRST,
        **kwargs,
    ):
        """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Images to preprocess.
            segmentation_maps (`ImageInput`, *optional*):
                Segmentation maps to preprocess.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Dictionary in the format `{"height": h, "width": w}` specifying the size of the output image after
                resizing.
            resample (`PILImageResampling` filter, *optional*, defaults to `self.resample`):
                `PILImageResampling` filter to use if resizing the image e.g. `PILImageResampling.BILINEAR`. Only has
                an effect if `do_resize` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use if `do_normalize` is set to `True`.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        resample = resample if resample is not None else self.resample
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        size = size if size is not None else self.size
        size_dict = get_size_dict(size)

        if not is_batched(images):
            images = [images]

        if not valid_images(images):
            raise ValueError(
                "Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, "
                "torch.Tensor, tf.Tensor or jax.ndarray."
            )

        if do_resize and size is None:
            raise ValueError("Size must be specified if do_resize is True.")

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]
        segmenmtation_maps = [to_numpy_array(segmentation_map) for segmentation_map in segmentation_maps]

        # Step 1: reduce labels
        if self.do_reduce_label and segmentation_maps is not None:
            segmentation_maps = [self.reduce_label(segmentation_map) for segmentation_map in segmentation_maps]

        # Step 2: pad
        if self.do_pad:
            images = [self.pad(image) for image in images]
            if segmentation_maps is not None:
                segmentation_maps = [self.pad(segmentation_map) for segmentation_map in segmentation_maps]
        
        # Step 3: create binary masks
        if do_create_binary_mask and segmentation_maps is not None:
            classes_and_masks = [self.create_binary_mask(segmentation_map) for segmentation_map in segmentation_maps]

        # Step 4: normalize images, prepare targets

        images = [to_channel_dimension_format(image, data_format) for image in images]

        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)