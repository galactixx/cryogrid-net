"""
Model evaluation and testing script for CryogridNet.

Implements comprehensive model evaluation using test-time augmentation (TTA)
to improve prediction accuracy. Loads pre-trained model from HuggingFace Hub,
evaluates on test dataset with flip augmentations, and generates visualization
plots showing predicted slot centers overlaid on test images.
"""

import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from constants import RESIZE_H, RESIZE_W, SEED
from gridbox_net import GridBoxMobileNet
from preprocessing import DataSplit, create_image_centers, split_data
from transforms import PairHorizontalFlip
from utils import get_dataset_paths, seed_everything
from visualize import Point, visualize_slot_points

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

horizontal_flip = PairHorizontalFlip()


def init_points(batch_size: int) -> Dict[int, Dict[int, List[Point]]]:
    """Initialize nested dictionary structure for collecting TTA predictions."""
    return {bidx: {idx: list() for idx in range(4)} for bidx in range(batch_size)}


transforms = [
    (
        lambda i, t: (i, t),
        lambda pred: torch.tensor([pred % RESIZE_W, pred // RESIZE_W], device=device),
    ),
    (
        lambda i, t: horizontal_flip(i, t),
        lambda pred: torch.tensor(
            [RESIZE_W - 1 - (pred % RESIZE_W), pred // RESIZE_W],
            device=device,
        ),
    ),
]


def tta_evaluate(
    model: GridBoxMobileNet,
    data: DataSplit,
    display: bool = True,
    save: bool = False,
    path: Optional[Path] = None,
) -> float:
    """Evaluate model with test-time augmentation (TTA)."""
    positions: Dict[int, List[float]] = dict()

    cur_idx = 0
    with torch.no_grad():
        for img_pos, imgs, targets, pts in tqdm(data.test_loader, desc="Evaluation"):
            imgs, targets, pts = imgs.to(device), targets.to(device), pts.to(device)

            # Initialize points collection for TTA averaging
            points: Dict[int, Dict[int, List[Point]]] = init_points(imgs.size(0))

            # Apply each TTA transform
            for t, aug in transforms:
                imgs_new, targets_new = t(imgs, targets)
                logits = model(imgs_new)

                # Convert logits to predictions
                B, C = logits.size(0), logits.size(1)
                logits = logits.view(B, C, -1)
                preds = logits.argmax(dim=2)

                for img_idx in range(B):
                    pos = img_pos[img_idx]
                    if pos not in positions:
                        positions[pos] = list()

                    img_pred = preds[img_idx, :].clone()
                    img_pts = pts[img_idx, :]

                    # Calculate distance error for this augmentation
                    img_dist = 0
                    for idx in range(C):
                        pred = img_pred[idx]
                        pt = aug(pred)

                        dist = torch.dist(pt, img_pts[idx * 2 : idx * 2 + 2])
                        img_dist += dist.item()

                        point = Point(pt[0].item(), pt[1].item())
                        points[img_idx][idx].append(point)

                    positions[pos].append(img_dist / C)

            # Visualize predictions if requested
            if display:
                index = random.randint(0, imgs.size(0) - 1)

                # Average predictions across all TTA transforms
                img_points_agg: List[Point] = [None] * 4
                for pt_idx, img_points in points[index].items():
                    num_points = len(img_points)
                    x = y = 0
                    for img_point in img_points:
                        x += img_point.x
                        y += img_point.y
                    x /= num_points
                    y /= num_points
                    img_points_agg[pt_idx] = Point(x, y)

                visualize_slot_points(
                    img_t=imgs[index],
                    points=img_points_agg,
                    save=save,
                    path=path,
                    filename=f"preds_{data.test[cur_idx + index].path.name}",
                )

                cur_idx += imgs.size(0)

        # Calculate average distance across all positions
        running_dist = sum(np.mean(dists) for dists in positions.values())
        eval_dist = running_dist / len(positions)

    return eval_dist


if __name__ == "__main__":
    seed_everything(seed=SEED)
    images_path, annots_path = get_dataset_paths()
    annots = pd.read_csv(annots_path)

    model_path = hf_hub_download(
        "galactixx/gridbox-net", "gridbox-mobilenetv2.pth", token=False
    )

    model = GridBoxMobileNet()
    model.to(device)

    weights = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(weights, strict=True)
    model.eval()

    centers = create_image_centers(images_path, annots)
    data = split_data(centers)
    test_dist = tta_evaluate(model=model, data=data)
    print(f"Test avg distance: {test_dist:.3f}...")
