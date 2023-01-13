# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import random
from PIL import Image
from torchvision.transforms import (
    Resize,
    Compose,
    ToTensor,
    Normalize,
    ColorJitter,
    RandomHorizontalFlip,
    CenterCrop,
    RandomAffine,
    RandomGrayscale,
    GaussianBlur,
)


def build_train_test_transforms(
    height,
    width,
    transforms_train="random_flip",
    transforms_test="resize",
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    **kwargs
):
    """Build train and test transformation functions.

    Args:
        height (int): target image height.
        width (int): target image width.
        transforms_train (str or list of str, optional): transformations
            applied to training data. Default is 'random_flip'.
        transforms_test (str or list of str, optional): transformations
            applied to test data. Default is 'resize'.
        norm_mean (list or None, optional): normalization mean values.
            Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation
            values. Default is ImageNet standard deviation values.

    Returns:
        transform_tr: transformation function for training
        transform_te: transformation function for testing
    """
    transform_tr = build_transforms(
        height=height, width=width, transforms=transforms_train
    )

    transform_te = build_transforms(
        height=height, width=width, transforms=transforms_test
    )
    return transform_tr, transform_te


class Random2DTranslation(object):
    """Randomly translates the input image with a probability.

    Specifically, given a predefined shape (height, width), the input is first
    resized with a factor of 1.125, leading to (height*1.125, width*1.125),
    then a random crop is performed. Such operation is done with a probability.

    Args:
        height (int): target image height.
        width (int): target image width.
        p (float, optional): probability that this operation takes place.
            Default is 0.5.
        interpolation (int, optional): desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)

        new_width, new_height = int(round(self.width * 1.125)), int(
            round(self.height * 1.125)
        )
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img


def build_transforms(
    height,
    width,
    transforms="random_flip",
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    **kwargs
):
    """Build transformation functions.

    Args:
        height (int): target image height.
        width (int): target image width.
        transforms (str or list of str, optional): transformations applied to
            input data. Default is 'random_flip'.
        norm_mean (list or None, optional): normalization mean values.
            Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation
            values. Default is ImageNet standard deviation values.

    Returns:
        transform_tr: transformation function
    """
    if transforms is None:
        transforms = []

    if isinstance(transforms, str):
        transforms = [transforms]

    if not isinstance(transforms, list):
        raise ValueError(
            "transforms must be a list of strings, but found to be {}".format(
                type(transforms)
            )
        )

    if len(transforms) > 0:
        transforms = [t.lower() for t in transforms]

    if norm_mean is None or norm_std is None:
        norm_mean = [0.485, 0.456, 0.406]  # imagenet mean
        norm_std = [0.229, 0.224, 0.225]  # imagenet std
    normalize = Normalize(mean=norm_mean, std=norm_std)

    print("Building transforms ...")
    transform_tr = []

    if "center_crop" in transforms:
        print("+ center crop with size {}".format(max(height, width)))
        transform_tr += [CenterCrop(max(height, width))]

    if "resize" in transforms:
        print("+ resize to {}x{}".format(height, width))
        transform_tr += [Resize((height, width))]
        # transform_tr += [Resize(height, max_size=height)]

    if "random_flip" in transforms:
        print("+ random flip")
        transform_tr += [RandomHorizontalFlip()]

    if "random_crop" in transforms:
        print(
            "+ random crop (enlarge to {}x{} and "
            "crop {}x{})".format(
                int(round(height * 1.125)), int(round(width * 1.125)), height, width
            )
        )
        transform_tr += [Random2DTranslation(height, width)]

    if "random_affine" in transforms:
        print("+ random affine")
        transform_tr += [
            RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                shear=(5, 5),
                # resample=0,
                # fillcolor=0,
            )
        ]

    if "color_jitter" in transforms:
        print("+ color jitter: brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1")
        transform_tr += [
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
        ]

    if "random_grayscale" in transforms:
        print("+ random grayscale: p=0.2")
        transform_tr += [RandomGrayscale(p=0.2)]

    if "blur" in transforms:
        print("+ blur: kernel_size=11, sigma=(0.1, 2.0)")
        transform_tr += [GaussianBlur(kernel_size=11, sigma=(0.1, 2.0))]

    print("+ to torch tensor of range [0, 1]")
    transform_tr += [ToTensor()]

    print("+ normalization (mean={}, std={})".format(norm_mean, norm_std))
    transform_tr += [normalize]

    transform_tr = Compose(transform_tr)

    return transform_tr
