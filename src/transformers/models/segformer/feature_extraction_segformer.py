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


# taken from https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/misc.py
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


class AlignedResize(object):
    """
    Resize images & corresponding segmentation maps.

    This class is based on ``Resize``, the only difference is it ensures the long and short sides are divisible by
    ``size_divisor``.
    """

    def __init__(self, img_scale=None, multiscale_mode="range", ratio_range=None, keep_ratio=True, size_divisor=32):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ["value", "range"]

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.size_divisor = size_divisor

    @staticmethod
    def random_select(img_scales):
        """
        Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``, where ``img_scale`` is the selected image scale
            and ``scale_idx`` is the selected index in the given candidates.
        """

        assert is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """
        Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower and uper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where ``img_scale`` is sampled scale and None is just
            a placeholder to be consistent with :func:`random_select`.
        """

        assert is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(min(img_scale_long), max(img_scale_long) + 1)
        short_edge = np.random.randint(min(img_scale_short), max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """
        Randomly sample an img_scale when ``ratio_range`` is specified. A ratio will be randomly sampled from the range
        specified by ``ratio_range``. Then it would be multiplied with ``img_scale`` to generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where ``scale`` is sampled ratio multiplied with
            ``img_scale`` and None is just a placeholder to be consistent with :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """
        Randomly sample an img_scale according to ``ratio_range`` and ``multiscale_mode``. If ``ratio_range`` is
        specified, a ratio will be sampled and be multiplied with ``img_scale``. If multiple scales are specified by
        ``img_scale``, a scale will be sampled according to ``multiscale_mode``. Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into ``results``, which would be used by subsequent
            pipelines.
        """

        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results["img"].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h), self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == "range":
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == "value":
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results["scale"] = scale
        results["scale_idx"] = scale_idx

    def _align(self, img, size_divisor, resample=None):
        align_h = int(np.ceil(img.shape[0] / size_divisor)) * size_divisor
        align_w = int(np.ceil(img.shape[1] / size_divisor)) * size_divisor
        if resample == None:
            img = self.resize(image=img, size=(align_w, align_h))
        else:
            img = self.resize(image=img, size=(align_w, align_h), resample=resample)
        return img

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        if self.keep_ratio:
            img = self.resize(image=results["img"], size=results["scale"])
            #### align ####
            img = self._align(img, self.size_divisor)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the self.resize in the future
            new_h, new_w = img.shape[:2]
            h, w = results["img"].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = self.resize(image=results["img"], size=results["scale"])

            h, w = img.shape[:2]
            assert (
                int(np.ceil(h / self.size_divisor)) * self.size_divisor == h
                and int(np.ceil(w / self.size_divisor)) * self.size_divisor == w
            ), "img size not align. h:{} w:{}".format(h, w)
        results["img"] = img
        results["img_shape"] = img.shape
        results["pad_shape"] = img.shape  # in case that there is no padding
        results["keep_ratio"] = self.keep_ratio

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get("seg_fields", []):
            if self.keep_ratio:
                gt_seg = self.resize(image=results[key], size=results["scale"], resample=Image.NEAREST)
                gt_seg = self._align(gt_seg, self.size_divisor, resample=Image.NEAREST)
            else:
                gt_seg = self.resize(image=results[key], size=results["scale"], resample=Image.NEAREST)
                h, w = gt_seg.shape[:2]
                assert (
                    int(np.ceil(h / self.size_divisor)) * self.size_divisor == h
                    and int(np.ceil(w / self.size_divisor)) * self.size_divisor == w
                ), "gt_seg size not align. h:{} w:{}".format(h, w)
            results[key] = gt_seg

    def __call__(self, results):
        """
        Call function to resize images, bounding boxes, masks, semantic segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', 'keep_ratio' keys are added into result
            dict.
        """

        if "scale" not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_seg(results)
        return results


class SegFormerFeatureExtractor(FeatureExtractionMixin, ImageFeatureExtractionMixin):
    r"""
    Constructs a SegFormer feature extractor.

    This feature extractor inherits from :class:`~transformers.FeatureExtractionMixin` which contains most of the main
    methods. Users should refer to this superclass for more information regarding those methods.


    Args:
        do_resize (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether to resize the input to a certain :obj:`size`.
        size (:obj:`int` or :obj:`Tuple(int)`, `optional`, defaults to 512):
            Resize the input to the given size. If a tuple is provided, it should be (width, height). If only an
            integer is provided, then the input will be resized to (size, size). Only has an effect if :obj:`do_resize`
            is set to :obj:`True`.
        resample (:obj:`int`, `optional`, defaults to :obj:`PIL.Image.BILINEAR`):
            An optional resampling filter. This can be one of :obj:`PIL.Image.NEAREST`, :obj:`PIL.Image.BOX`,
            :obj:`PIL.Image.BILINEAR`, :obj:`PIL.Image.HAMMING`, :obj:`PIL.Image.BICUBIC` or :obj:`PIL.Image.LANCZOS`.
            Only has an effect if :obj:`do_resize` is set to :obj:`True`.
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
        size=512,
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
        self.image_mean = image_mean if image_mean is not None else [0.485, 0.456, 0.406]  # ImageNet mean
        self.image_std = image_std if image_std is not None else [0.229, 0.224, 0.225]  # ImageNet std

    def __call__(
        self,
        images: Union[
            Image.Image, np.ndarray, "torch.Tensor", List[Image.Image], List[np.ndarray], List["torch.Tensor"]  # noqa
        ],
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
        if self.do_resize and self.size is not None:
            images = [self.resize(image=image, size=self.size, resample=self.resample) for image in images]
        if self.do_normalize:
            images = [self.normalize(image=image, mean=self.image_mean, std=self.image_std) for image in images]

        # return as BatchFeature
        data = {"pixel_values": images}
        encoded_inputs = BatchFeature(data=data, tensor_type=return_tensors)

        return encoded_inputs
