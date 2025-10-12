"""
Data preprocessing and splitting utilities for CryogridNet.

Handles creation of image center annotations from CSV data and implements
train/validation/test splitting based on grid box positions. Provides DataSplit
dataclass with cached DataLoader properties for efficient data loading.
"""

import random
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import List, Literal, Set, Tuple, TypeAlias

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from constants import SEED
from dataset import GridBoxDataset, ImageCenters, SlotCenterPoint


@dataclass(frozen=True)
class DataSplit:
    """Container for train/validation/test data splits with cached DataLoaders."""

    train: List[ImageCenters]
    val: List[ImageCenters]
    test: List[ImageCenters]

    @cached_property
    def train_loader(self) -> DataLoader:
        """Create DataLoader for training data with shuffling and augmentation."""

        def worker_init_fn(worker_id: int) -> None:
            np.random.seed(SEED + worker_id)
            random.seed(SEED + worker_id)

        dataset = GridBoxDataset(centers=self.train)
        g = torch.Generator()
        g.manual_seed(SEED)
        return DataLoader(
            dataset,
            shuffle=True,
            batch_size=16,
            generator=g,
            num_workers=2,
            worker_init_fn=worker_init_fn,
        )

    @cached_property
    def val_loader(self) -> DataLoader:
        """Create DataLoader for validation data without augmentation."""
        dataset = GridBoxDataset(centers=self.val, train=False)
        return DataLoader(dataset, shuffle=False, batch_size=16, num_workers=2)

    @cached_property
    def test_loader(self) -> DataLoader:
        """Create DataLoader for test data without augmentation."""
        dataset = GridBoxDataset(centers=self.test, train=False)
        return DataLoader(dataset, shuffle=False, batch_size=16, num_workers=2)


def create_image_centers(path: Path, data: pd.DataFrame) -> List[ImageCenters]:
    """Convert CSV annotations to ImageCenters objects."""
    SlotType: TypeAlias = Literal["L", "T", "R", "B"]

    def create_point(points: pd.DataFrame, slot: SlotType) -> SlotCenterPoint:
        slot_data = points[points.Slot == slot]
        x = slot_data.X.iloc[0]
        y = slot_data.Y.iloc[0]
        return SlotCenterPoint(x=x, y=y)

    image_centers: List[ImageCenters] = []
    images = data.File.unique().tolist()

    for image_name in images:
        points = data[data.File == image_name]
        # Extract position number from position string (e.g., "position-000001" -> 1)
        position = points.iloc[0, 0]
        position = int(position.split("-")[-1])

        # Create center points for each slot
        l = create_point(points=points, slot="L")
        t = create_point(points=points, slot="T")
        r = create_point(points=points, slot="R")
        b = create_point(points=points, slot="B")

        image_path = path / image_name
        centers = ImageCenters(
            path=image_path,
            position=position,
            left=l,
            top=t,
            right=r,
            bottom=b,
        )
        image_centers.append(centers)
    return image_centers


def split_data(centers: List[ImageCenters]) -> DataSplit:
    """Split data by grid box positions to prevent data leakage."""

    def get_centers(positions: Set[int]) -> List[ImageCenters]:
        return [center for center in centers if center.position in positions]

    # Get unique positions and split them
    pos = list(set(center.position for center in centers))
    pos.sort()

    # Split positions: 70% train, 15% val, 15% test
    train_pos, temp_pos = train_test_split(pos, test_size=0.3, random_state=SEED)
    test_pos, val_pos = train_test_split(temp_pos, test_size=0.5, random_state=SEED)

    # Create splits based on position groups
    train_centers = get_centers(positions=train_pos)
    val_centers = get_centers(positions=val_pos)
    test_centers = get_centers(positions=test_pos)

    return DataSplit(train_centers, val_centers, test_centers)
