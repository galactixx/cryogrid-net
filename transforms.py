import random
from typing import Tuple

import torch
import torchvision.transforms.functional as F
from PIL import Image


class RandomPairHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self, img: Image.Image, target: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            img = torch.flip(img, dims=[2])
            target = torch.flip(target, dims=[2])
        return img, target


class RandomPairVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self, img: Image.Image, target: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            img = torch.flip(img, dims=[1])
            target = torch.flip(target, dims=[1])
        return img, target


class RandomPairBrightness:
    def __init__(self, p: float = 0.5, strength=(0.7, 1.3)) -> None:
        self.p = p
        self.strength = strength

    def __call__(
        self, img: Image.Image, target: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            factor = random.uniform(*self.strength)
            img = F.adjust_brightness(img, factor)
        return img, target


class RandomPairContrast:
    def __init__(self, p: float = 0.5, strength=(0.7, 1.3)) -> None:
        self.p = p
        self.strength = strength

    def __call__(
        self, img: Image.Image, target: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.p:
            factor = random.uniform(*self.strength)
            img = F.adjust_contrast(img, factor)
        return img, target


class PairCompose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(
        self, img: Image.Image, target: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        for t in self.transforms:
            img, target = t(img, target)
        return img, target
