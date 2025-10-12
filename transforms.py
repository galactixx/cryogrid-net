"""
Data augmentation transforms for CryogridNet training.

Implements paired image and target augmentation transforms that maintain
spatial correspondence between input images and heatmap targets. Includes
horizontal/vertical flips, brightness/contrast adjustments, and composition
class for applying multiple transforms sequentially.
"""

import random
from typing import Tuple

import torch
import torchvision.transforms.functional as F


class RandomPairHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self, img: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            img = torch.flip(img, dims=[2])
            target = torch.flip(target, dims=[2])
        return img, target


class RandomPairVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self, img: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            img = torch.flip(img, dims=[1])
            target = torch.flip(target, dims=[1])
        return img, target


class RandomPairBrightness:
    def __init__(self, p: float = 0.5, strength=(0.7, 1.3)) -> None:
        self.p = p
        self.strength = strength

    def __call__(
        self, img: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            factor = random.uniform(*self.strength)
            img = F.adjust_brightness(img, factor)
        return img, target


class RandomPairContrast:
    def __init__(self, p: float = 0.5, strength=(0.7, 1.3)) -> None:
        self.p = p
        self.strength = strength

    def __call__(
        self, img: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < self.p:
            factor = random.uniform(*self.strength)
            img = F.adjust_contrast(img, factor)
        return img, target


class PairCompose:
    def __init__(self, transforms: list) -> None:
        self.transforms = transforms

    def __call__(
        self, img: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class PairHorizontalFlip:
    def __call__(
        self, img: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img = torch.flip(img, dims=[3])
        target = torch.flip(target, dims=[3])
        return img, target


class PairVerticalFlip:
    def __call__(
        self, img: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img = torch.flip(img, dims=[2])
        target = torch.flip(target, dims=[2])
        return img, target
