# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Feature extractor class for SegFormer."""

from collections import abc
from typing import List, Optional, Union

import numpy as np
from PIL import Image

from ...feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from ...file_utils import TensorType
from ...image_utils import ImageFeatureExtractionMixin, is_torch_tensor
from ...utils import logging


logger = logging.get_logger(__name__)


# 2 functions below taken from https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/misc.py
def is_seq_of(seq, expected_type, seq_type=None):
    """
    Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_list_of(seq, expected_type):
    """
    Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)


# 2 functions below taken from https://github.com/open-mmlab/mmcv/blob/master/mmcv/image/geometric.py
def _scale_size(size, scale):
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | tuple(float)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    return int(w * float(scale[0]) + 0.5), int(h * float(scale[1]) + 0.5)


def rescale_size(old_size, scale, return_scale=False):
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


class SegFormerFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    r"""
    Constructs a SegFormer feature extractor.

    This feature extractor inherits from :class:`~transformers.FeatureExtractionMixin` which contains most of the main
    methods. Users should refer to this superclass for more information regarding those methods.

    Args:
        do_resize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to resize/rescale the input based on a certain :obj:`image_scale`.
        keep_ratio (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to keep the aspect ratio when resizing the image.
        image_scale (:obj:`float` or :obj:`Tuple(int)`, `optional`, defaults to (2048, 512)):
            In case :obj:`keep_ratio` is set to :obj:`True`, the scaling factor or maximum size. If it is a float number, then 
            the image will be rescaled by this factor, else if it is a tuple of 2 integers (width, height), then the image will be 
            rescaled as large as possible within the scale. 
            In case :obj:`keep_ratio` is set to :obj:`False`, the target size (w, h) to which the image will be resized.
            
            Only has an effect if :obj:`do_resize` is set to :obj:`True`.
        size_divisor (:obj:`int`, `optional`, defaults to 32):
            The integer by which both sides of an image should be divisible. Only has an effect if :obj:`do_resize` 
            is set to :obj:`True`.
        resample (:obj:`int`, `optional`, defaults to :obj:`PIL.Image.BILINEAR`):
            An optional resampling filter. This can be one of :obj:`PIL.Image.NEAREST`, :obj:`PIL.Image.BOX`,
            :obj:`PIL.Image.BILINEAR`, :obj:`PIL.Image.HAMMING`, :obj:`PIL.Image.BICUBIC` or :obj:`PIL.Image.LANCZOS`.
            Only has an effect if :obj:`do_resize` is set to :obj:`True`.
        do_normalize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to randomly crop the input to a certain obj:`crop_size`.
        crop_size (:obj:`int`, `optional`, defaults to 512):
            The crop size to use. Only has an effect if :obj:`do_random_crop` is set to :obj:`True`.
        do_normalize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to normalize the input with mean and standard deviation.
        image_mean (:obj:`int`, `optional`, defaults to :obj:`[0.485, 0.456, 0.406]`):
            The sequence of means for each channel, to be used when normalizing images. Defaults to the ImageNet mean.
        image_std (:obj:`int`, `optional`, defaults to :obj:`[0.229, 0.224, 0.225]`):
            The sequence of standard deviations for each channel, to be used when normalizing images. Defaults to the
            ImageNet std.
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize=True,
        keep_ratio=True,
        image_scale=(2048, 512),
        size_divisor=32,
        resample=Image.BILINEAR,
        do_random_crop=True,
        crop_size=(512, 512),
        do_normalize=True,
        image_mean=None,
        image_std=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.keep_ratio = keep_ratio
        self.image_scale = image_scale
        self.size_divisor = size_divisor
        self.resample = resample
        self.do_random_crop = do_random_crop
        self.crop_size = crop_size
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.485, 0.456, 0.406]  # ImageNet mean
        self.image_std = image_std if image_std is not None else [0.229, 0.224, 0.225]  # ImageNet std

    def _align(self, image, size_divisor, resample=None):
        align_w = int(np.ceil(image.size[0] / self.size_divisor)) * self.size_divisor
        align_h = int(np.ceil(image.size[1] / self.size_divisor)) * self.size_divisor
        if resample == None:
            image = self.resize(image=image, size=(align_w, align_h))
        else:
            image = self.resize(image=image, size=(align_w, align_h), resample=resample)
        return image

    def align_resize_image(self, image, size, resample):
        """
        This class is based on PIL's ``resize`` method, the only difference is it ensures the long and short sides 
        are divisible by ``self.size_divisor``.  

        If ``self.keep_ratio`` equals ``True``, then it replicates mmcv.rescale, else it replicates mmcv.resize.       
        """
        if not isinstance(image, Image.Image):
            image = self.to_pil_image(image)
        
        if self.keep_ratio:     
            w, h = image.size
            # calculate new size 
            new_size = rescale_size((w, h), scale=self.image_scale, return_scale=False)
            image = self.resize(image=image, size=new_size, resample=resample)
            image = self._align(image, self.size_divisor)
        else:
            image = self.resize(image=image, size=self.image_scale, resample=resample)
            w, h = image.size
            assert int(np.ceil(h / self.size_divisor)) * self.size_divisor == h and \
                   int(np.ceil(w / self.size_divisor)) * self.size_divisor == w, \
                   "img size doesn't align. h:{} w:{}".format(h,w)

        return image

    def resize_segmentation_map(self, segmentation_map):
        """Resize semantic segmentation map.
        
        This method is equal to _resize defined above, with the only difference that PIL.Image.NEAREST is used instead of None.
        """
        if not isinstance(segmentation_map, Image.Image):
            segmentation_map = self.to_pil_image(segmentation_map)
        
        if self.keep_ratio:
            w, h = segmentation_map.size
            # calculate new size 
            new_size = rescale_size((w, h), scale=self.image_scale, return_scale=False)
            gt_seg = self.resize(image=segmentation_map, size=new_size, resample=Image.NEAREST)
            gt_seg = self._align(gt_seg, self.size_divisor, resample=Image.NEAREST)
        else:
            gt_seg = self.resize(image=segmentation_map, size=self.image_scale, resample=Image.NEAREST)
            w, h = gt_seg.size
            assert (
                int(np.ceil(h / self.size_divisor)) * self.size_divisor == h
                and int(np.ceil(w / self.size_divisor)) * self.size_divisor == w
            ), "gt_seg size not align. h:{} w:{}".format(h, w)
        
        return gt_seg
    
    def _pad_img(self, image):
        """Pad images according to ``self.size``."""
        padded_image = nn.functional.pad(image, pad=self.crop_size)
        
        return padded_image

    def _pad_seg(self, segmentation_map):
        """Pad masks according to ``results['pad_shape']``."""
        padded_segmentation_map = nn.functional.pad(segmentation_map, pad=results['pad_shape'][:2], value=255)

        return padded_segmentation_map
    
    def __call__(
        self,
        images: Union[
            Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]  # noqa
        ],
        segmentation_maps: Union[Image.Image, List[Image.Image]] = None,
        align: bool = False,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several image(s).

        .. warning::

           NumPy arrays and PyTorch tensors are converted to PIL images when resizing, so the most efficient is to pass
           PIL images.

        Args:
            images (:obj:`PIL.Image.Image`, :obj:`np.ndarray`, :obj:`torch.Tensor`, :obj:`List[PIL.Image.Image]`, :obj:`List[np.ndarray]`, :obj:`List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.

            segmentation_maps (:obj:`PIL.Image.Image`, :obj:`List[PIL.Image.Image]`):
                The corresponding semantic segmentation maps with the pixel-wise annotations.
            
            align (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to ensure the long and short sides of an image are divisible by :obj:`size_divisor` when resizing.
            
            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`, defaults to :obj:`'np'`):
                If set, will return tensors of a particular framework. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return NumPy :obj:`np.ndarray` objects.
                * :obj:`'jax'`: Return JAX :obj:`jnp.ndarray` objects.

        Returns:
            :class:`~transformers.BatchFeature`: A :class:`~transformers.BatchFeature` with the following fields:

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
                "Images must of type `PIL.Image.Image`, `np.ndarray` or `torch.Tensor` (single example),"
                "`List[PIL.Image.Image]`, `List[np.ndarray]` or `List[torch.Tensor]` (batch of examples)."
            )

        is_batched = bool(
            isinstance(images, (list, tuple))
            and (isinstance(images[0], (Image.Image, np.ndarray)) or is_torch_tensor(images[0]))
        )

        if not is_batched:
            images = [images]

        # transformations (resizing + normalization)
        if self.do_resize and self.image_scale is not None:
            if align:
                images = [self.align_resize_image(image=image, size=self.image_scale, resample=self.resample) for image in images]
            else:
                images = [self.resize(image=image, size=self.image_scale, resample=self.resample) for image in images]
            if segmentation_maps is not None:
                segmentation_maps = [self.resize_segmentation_map(map) for map in segmentation_maps]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]

        # return as BatchFeature
        data = {"pixel_values": images}

        if segmentation_maps is not None:
            data["labels"] = segmentation_maps 

        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs

    def show_result_pyplot(image, outputs, fig_size=(15, 10)):
        """Visualize the segmentation results on the image.
        """
        plt.figure(figsize=fig_size)
        plt.imshow(mmcv.bgr2rgb(img))
        plt.show()
