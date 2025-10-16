"""
Training script for CryogridNet with progressive unfreezing.

Implements complete training pipeline using Focal Loss for class imbalance
in heatmap regression. Features progressive unfreezing of encoder layers,
exponential moving average (EMA), mixed precision training, and early stopping.
Includes comprehensive validation evaluation and automatic model checkpointing.
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Tuple, TypeAlias

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from tqdm.auto import tqdm

from constants import RESIZE_W, SEED
from dataset import ImageCenters
from gridbox_net import GridBoxDenseNet, GridBoxMobileNet, GridBoxResNet
from preprocessing import create_image_centers, split_data
from utils import ParamGroup, ProgressiveUnfreezer, get_dataset_paths, seed_everything

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in heatmap regression."""

    def __init__(
        self,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = Fnn.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def train_evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    ema: ExponentialMovingAverage,
    criterion: FocalLoss,
) -> Tuple[float, float]:
    """Evaluate model on validation/test data using EMA weights."""
    model.eval()
    ema.store()
    ema.copy_to()

    running_loss = 0
    positions: Dict[int, List[float]] = dict()

    with torch.no_grad():
        for img_pos, imgs, targets, pts in tqdm(loader, desc="Evaluation"):
            imgs, targets, pts = (imgs.to(device), targets.to(device), pts.to(device))

            logits = model(imgs)
            loss = criterion(logits, targets)

            B, C = logits.size(0), logits.size(1)
            logits = logits.view(B, C, -1)
            preds = logits.argmax(dim=2)

            for img_idx in range(B):
                pos = img_pos[img_idx]
                if pos not in positions:
                    positions[pos] = []

                img_pred = preds[img_idx, :]
                img_pts = pts[img_idx, :]

                img_dist = 0
                for idx in range(C):
                    pred = img_pred[idx]
                    pt = torch.tensor(
                        [pred % RESIZE_W, pred // RESIZE_W], device=device
                    )
                    dist = torch.dist(pt, img_pts[idx * 2 : idx * 2 + 2])
                    img_dist += dist.item()
                positions[pos].append(img_dist / C)
            running_loss += loss.item() * B

        running_dist = sum(np.mean(dists) for pos, dists in positions.items())
        eval_dist = running_dist / len(positions)

    ema.restore()

    eval_loss = running_loss / len(loader.dataset)
    return eval_loss, eval_dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a UNet heatmap regression model using different CNN encoders."
    )
    parser.add_argument(
        "--encoder",
        choices=["mobilenetv2", "densenet121", "resnet18"],
        required=True,
        help="The pretrained CNN encoder to use.",
    )
    args = parser.parse_args()

    seed_everything(seed=SEED)
    images_path, annots_path = get_dataset_paths()

    annots = pd.read_csv(annots_path)
    centers = create_image_centers(images_path, annots)
    data = split_data(centers)

    if args.encoder == "mobilenetv2":
        model = GridBoxMobileNet()
        encoder = model.backbone.encoder.features

        stage1 = encoder[0:2]
        stage2 = encoder[2:4]
        stage3 = encoder[4:7]
        stage4 = encoder[7:14]
        stage5 = encoder[14:19]
    elif args.encoder == "resnet18":
        model = GridBoxResNet()
        encoder = model.backbone.encoder

        stage1 = torch.nn.Sequential(
            encoder.conv1,
            encoder.bn1,
        )
        stage2 = encoder.layer1
        stage3 = encoder.layer2
        stage4 = encoder.layer3
        stage5 = encoder.layer4
    else:
        model = GridBoxDenseNet()
        encoder = model.backbone.encoder.features

        stage1 = torch.nn.Sequential(
            encoder.conv0,
            encoder.norm0,
            encoder.relu0,
            encoder.pool0,
        )
        stage2 = encoder.denseblock1
        stage3 = torch.nn.Sequential(encoder.transition1, encoder.denseblock2)
        stage4 = torch.nn.Sequential(encoder.transition2, encoder.denseblock3)
        stage5 = torch.nn.Sequential(
            encoder.transition3, encoder.denseblock4, encoder.norm5
        )

    model.to(device)

    # Freeze all stages initially
    for stage in [stage1, stage2, stage3, stage4, stage5]:
        for param in stage.parameters():
            param.requires_grad = False
        # Keep BatchNorm in eval mode for frozen stages
        for _, module in stage.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

    unfreezer = ProgressiveUnfreezer(
        [
            ParamGroup(
                epoch=1,
                layer=stage5,
                lr=5e-4,
                decay=1e-4,
            ),
            ParamGroup(
                epoch=3,
                layer=stage4,
                lr=1e-4,
                decay=1e-4,
            ),
            ParamGroup(
                epoch=5,
                layer=stage3,
                lr=1e-5,
                decay=1e-4,
            ),
        ],
    )

    ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

    EPOCHS = 100
    optimizer = AdamW(
        [
            {
                "params": filter(lambda p: p.requires_grad, model.parameters()),
                "lr": 1e-3,
                "weight_decay": 1e-4,
            }
        ]
    )
    unfreezer.optimizer = optimizer

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3)
    criterion = FocalLoss()

    patience = 7
    best_loss = float("inf")
    no_improve = 0

    scaler = GradScaler()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for _, imgs, targets, _ in tqdm(
            data.train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"
        ):
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()

            with autocast():
                logits = model(imgs)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(data.train_loader.dataset)

        val_loss, dist = train_evaluate(model, data.val_loader, ema, criterion)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{EPOCHS}.. "
            f"Train BCE loss: {train_loss:.3f}.. "
            f"Val BCE loss: {val_loss:.3f}.. "
            f"Val avg distance: {dist:.3f}..."
        )

        if val_loss < best_loss:
            no_improve = 0
            best_loss = val_loss
            ema.copy_to()
            torch.save(model.state_dict(), f"{args.encoder}.bin")
            ema.restore()
        else:
            no_improve += 1
            if no_improve >= patience:
                break

        unfreezer.unfreeze(epoch=epoch + 1)
        torch.cuda.empty_cache()
