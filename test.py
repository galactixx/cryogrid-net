"""
Model evaluation and testing script for CryogridNet.

Implements comprehensive model evaluation using test-time augmentation (TTA)
to improve prediction accuracy. Loads pre-trained model from HuggingFace Hub,
evaluates on test dataset with flip augmentations, and generates visualization
plots showing predicted slot centers overlaid on test images.
"""

import argparse
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

from constants import NUM_CENTERS, RESIZE_H, RESIZE_W, SEED
from gridbox_net import GridBoxDenseNet, GridBoxMobileNet, GridBoxResNet
from preprocessing import DataSplit, create_image_centers, split_data
from transforms import PairHorizontalFlip, PairVerticalFlip
from utils import get_dataset_paths, seed_everything
from visualize import visualize_slot_points

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

horizontal_flip = PairHorizontalFlip()
vertical_flip = PairVerticalFlip()


def init_points(batch_size: int) -> Dict[int, Dict[int, List[torch.Tensor]]]:
    """Initialize nested dictionary structure for collecting TTA predictions."""
    return {
        bidx: {idx: list() for idx in range(NUM_CENTERS)} for bidx in range(batch_size)
    }


transforms = [
    (
        lambda i, t: (i, t),
        lambda pred: torch.tensor([pred % RESIZE_W, pred // RESIZE_W], device=device),
    ),
    (
        lambda i, t: horizontal_flip(i, t),
        lambda pred: torch.tensor(
            [RESIZE_W - 1 - pred % RESIZE_W, pred // RESIZE_W],
            device=device,
        ),
    ),
    (
        lambda i, t: vertical_flip(i, t),
        lambda pred: torch.tensor(
            [pred % RESIZE_W, RESIZE_H - 1 - (pred // RESIZE_W)],
            device=device,
        ),
    ),
]


def tta_evaluate(
    encoder: str,
    model: torch.nn.Module,
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

            points: Dict[int, Dict[int, List[torch.Tensor]]] = init_points(imgs.size(0))

            for t, aug in transforms:
                imgs_new, targets_new = t(imgs, targets)
                logits = model(imgs_new)

                B, C = logits.size(0), logits.size(1)
                logits = logits.view(B, C, -1)
                preds = logits.argmax(dim=2)

                for img_idx in range(B):
                    img_pred = preds[img_idx, :].clone()

                    for idx in range(C):
                        pred = img_pred[idx]
                        pt = aug(pred)
                        point = torch.tensor([pt[0], pt[1]], device=device)
                        points[img_idx][idx].append(point)

            index = random.randint(0, imgs.size(0) - 1)

            for img_idx, t_points in points.items():
                pos = img_pos[img_idx]
                if pos not in positions:
                    positions[pos] = list()

                img_pts = pts[img_idx, :]
                img_points_agg: List[torch.Tensor] = [None] * NUM_CENTERS
                for pt_idx, img_points in t_points.items():
                    img_points_agg[pt_idx] = torch.stack(img_points).float().mean(dim=0)

                agg_pts = torch.stack(img_points_agg)
                gt_pts = img_pts.view(NUM_CENTERS, 2)
                img_dist = (agg_pts - gt_pts).norm(dim=1).mean().item()
                positions[pos].append(img_dist)

                if display and img_idx == index:
                    visualize_slot_points(
                        img_t=imgs[index],
                        points=img_points_agg,
                        save=save,
                        path=path,
                        filename=f"{encoder}_preds_{data.test[cur_idx + index].path.name}",
                    )

            cur_idx += imgs.size(0)

        running_dist = sum(np.mean(dists) for dists in positions.values())
        eval_dist = running_dist / len(positions)

    return eval_dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a UNet heatmap regression model using different CNN encoders."
    )
    parser.add_argument(
        "--encoder",
        choices=["mobilenetv2", "densenet121", "resnet18"],
        required=True,
        help="The pretrained CNN encoder to use.",
    )
    args = parser.parse_args()
    filename = f"{args.encoder}.bin"

    if args.encoder == "mobilenetv2":
        model = GridBoxMobileNet()
    elif args.encoder == "resnet18":
        model = GridBoxResNet()
    else:
        model = GridBoxDenseNet()

    seed_everything(seed=SEED)
    images_path, annots_path = get_dataset_paths()
    annots = pd.read_csv(annots_path)

    model_path = hf_hub_download("galactixx/gridbox-net", filename, token=False)

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
    test_dist = tta_evaluate(encoder=args.encoder, model=model, data=data)
    print(f"Test avg distance: {test_dist:.3f}...")
