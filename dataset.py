"""
Dataset classes and data structures for CryogridNet.

Defines core data structures for grid box images and slot center annotations.
Includes GridBoxDataset class for loading images, generating rhombus-shaped
heatmaps, and applying data augmentation transforms.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from constants import INET_MEAN, INET_STD, SCALE_X, SCALE_Y
from transforms import (
    PairCompose,
    RandomPairBrightness,
    RandomPairContrast,
    RandomPairHorizontalFlip,
    RandomPairVerticalFlip,
)


@dataclass(frozen=True)
class SlotCenterPoint:
    """Represents a single slot center point with coordinate scaling."""

    x: int
    y: int

    @property
    def scaled_y(self) -> float:
        """Convert original Y coordinate to resized image coordinates."""
        return self.y * SCALE_Y

    @property
    def scaled_x(self) -> float:
        """Convert original X coordinate to resized image coordinates."""
        return self.x * SCALE_X


@dataclass(frozen=True)
class ImageCenters:
    """Container for all slot center points in a single image."""

    path: Path
    position: int
    left: SlotCenterPoint
    top: SlotCenterPoint
    right: SlotCenterPoint
    bottom: SlotCenterPoint

    @property
    def points(self) -> List[int]:
        """Return all slot center coordinates as a flattened list."""
        return [
            self.left.scaled_x,
            self.left.scaled_y,
            self.top.scaled_x,
            self.top.scaled_y,
            self.right.scaled_x,
            self.right.scaled_y,
            self.bottom.scaled_x,
            self.bottom.scaled_y,
        ]


class GridBoxDataset(Dataset):
    """PyTorch Dataset for loading grid box images and generating heatmap targets."""

    def __init__(
        self,
        centers: List[ImageCenters],
        out_size: Tuple[int, int] = (512, 960),
        rhombus_rx: float = 17.0,
        rhombus_ry: float = 11.0,
        softness: float = 10.0,
        train: bool = True,
    ) -> None:
        super().__init__()
        self.centers = centers
        self.out_h, self.out_w = out_size
        self.rx = rhombus_rx
        self.ry = rhombus_ry
        self.softness = softness
        self.train = train

        self.normalize = transforms.Normalize(INET_MEAN, INET_STD)
        self.resizer = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((512, 960)),
                transforms.ToTensor(),
            ]
        )

        self.transforms = PairCompose(
            [
                RandomPairHorizontalFlip(),
                RandomPairVerticalFlip(),
                RandomPairBrightness(),
                RandomPairContrast(),
            ]
        )

    def __len__(self) -> int:
        return len(self.centers)

    def _make_rhombus_heatmap(self, cx: float, cy: float) -> np.ndarray:
        """Generate rhombus-shaped heatmap centered at (cx, cy)."""
        ys = np.arange(self.out_h, dtype=np.float32)[:, None]
        xs = np.arange(self.out_w, dtype=np.float32)[None, :]

        # Calculate rhombus distance metric
        s = (np.abs(xs - cx) / (self.rx + 1e-8)) + (np.abs(ys - cy) / (self.ry + 1e-8))

        if self.softness > 0:
            # Soft rhombus with exponential decay outside boundary
            contrib = np.exp(-self.softness * (s - 1.0).clip(min=0.0) ** 2)
            contrib = np.where(s <= 1.0, 1.0, contrib).astype(np.float32)
        else:
            # Hard rhombus boundary
            contrib = (s <= 1.0).astype(np.float32)
        return contrib

    def __getitem__(
        self, idx: int
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        center = self.centers[idx]

        # Load and convert image from BGR to RGB
        img = cv2.imread(center.path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Get scaled coordinates for heatmap generation
        pts = np.array(center.points, dtype=np.float32).reshape(-1, 2)

        # Generate heatmaps for each slot center
        heatmaps = []
        for cx, cy in pts:
            hm = self._make_rhombus_heatmap(cx, cy)
            heatmaps.append(hm)

        heatmaps = np.stack(heatmaps, axis=0)
        heatmap_t = torch.from_numpy(heatmaps)

        img_t = self.resizer(img)
        if self.train:
            img_t, heatmap_t = self.transforms(img_t, heatmap_t)
        img_t = self.normalize(img_t)

        pts_tensor = torch.tensor(center.points, dtype=torch.float32)
        return center.position, img_t, heatmap_t, pts_tensor
