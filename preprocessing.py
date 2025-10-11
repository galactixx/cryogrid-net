from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import List, Literal, Set, Tuple, TypeAlias

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from constants import SEED
from dataset import GridBoxDataset, ImageCenters, SlotCenterPoint


@dataclass(frozen=True)
class DataSplit:
    train: List[ImageCenters]
    val: List[ImageCenters]
    test: List[ImageCenters]

    @cached_property
    def train_loader(self) -> DataLoader:
        dataset = GridBoxDataset(centers=self.train)
        g = torch.Generator()
        g.manual_seed(SEED)
        return DataLoader(dataset, shuffle=True, batch_size=16, generator=g)

    @cached_property
    def val_loader(self) -> DataLoader:
        dataset = GridBoxDataset(centers=self.val, train=False)
        return DataLoader(dataset, shuffle=False, batch_size=16)

    @cached_property
    def test_loader(self) -> DataLoader:
        dataset = GridBoxDataset(centers=self.test, train=False)
        return DataLoader(dataset, shuffle=False, batch_size=16)


def create_image_centers(path: Path, data: pd.DataFrame) -> List[ImageCenters]:
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
        position = points.iloc[0, 0]
        position = int(position.split("-")[-1])

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
    def get_centers(positions: Set[str]) -> List[ImageCenters]:
        return [center for center in centers if center.position in positions]

    pos = list(set(center.position for center in centers))
    pos.sort()
    train_pos, temp_pos = train_test_split(pos, test_size=0.3, random_state=SEED)

    test_pos, val_pos = train_test_split(temp_pos, test_size=0.5, random_state=SEED)

    train_centers = get_centers(positions=train_pos)
    val_centers = get_centers(positions=val_pos)
    test_centers = get_centers(positions=test_pos)

    return DataSplit(train_centers, val_centers, test_centers)
