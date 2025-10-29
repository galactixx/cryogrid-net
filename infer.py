"""
Inference script for GridBox-Net models.

Performs inference on individual gridbox images using pretrained UNet-based
heatmap regression models with multiple CNN encoders (ResNet18, DenseNet121,
MobileNetV2). Employs test-time augmentation to improve accuracy and
visualizes detected slot points on the input image.
"""

import argparse
import os
from pathlib import Path
from test import transforms
from typing import Dict, List

import torch
from huggingface_hub import hf_hub_download

from constants import NUM_CENTERS, RESIZE_H, RESIZE_W, SEED
from dataset import preprocess_image_inference
from model import GridBoxDenseNet, GridBoxMobileNet, GridBoxResNet
from transforms import HorizontalFlip, VerticalFlip
from utils import seed_everything
from visualize import visualize_slot_points

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

horizontal_flip = HorizontalFlip()
vertical_flip = VerticalFlip()


def init_points() -> Dict[int, List[torch.Tensor]]:
    """Initialize nested dictionary structure for collecting TTA predictions."""
    return {idx: list() for idx in range(NUM_CENTERS)}


transforms = [
    (
        lambda i: i,
        lambda pred: torch.tensor([pred % RESIZE_W, pred // RESIZE_W], device=device),
    ),
    (
        lambda i: horizontal_flip(i),
        lambda pred: torch.tensor(
            [RESIZE_W - 1 - pred % RESIZE_W, pred // RESIZE_W],
            device=device,
        ),
    ),
    (
        lambda i: vertical_flip(i),
        lambda pred: torch.tensor(
            [pred % RESIZE_W, RESIZE_H - 1 - (pred // RESIZE_W)],
            device=device,
        ),
    ),
]

if __name__ == "__main__":
    seed_everything(seed=SEED)
    parser = argparse.ArgumentParser(
        description="Test a single image on a UNet heatmap regression model using different CNN encoders."
    )
    parser.add_argument(
        "--encoder",
        choices=["mobilenetv2", "densenet121", "resnet18"],
        required=True,
        help="The pretrained CNN encoder to use.",
    )
    parser.add_argument("path", type=str, help="Path to input the gridbox image.")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        parser.error(f"Path does not exist: {args.path}")

    filename = f"{args.encoder}.bin"
    if args.encoder == "mobilenetv2":
        model = GridBoxMobileNet()
    elif args.encoder == "resnet18":
        model = GridBoxResNet()
    else:
        model = GridBoxDenseNet()

    model_path = hf_hub_download("galactixx/gridbox-net", filename, token=False)
    model.to(device)

    weights = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(weights, strict=True)
    model.eval()

    path = Path(args.path)
    img = preprocess_image_inference(path=path)
    img = img.unsqueeze(0)

    with torch.no_grad():
        points: Dict[int, List[torch.Tensor]] = init_points()

        for t, aug in transforms:
            imgs_new = t(img)
            logits = model(imgs_new)

            B, C = logits.size(0), logits.size(1)
            logits = logits.view(B, C, -1)
            preds = logits.argmax(dim=2).squeeze(0)

            for idx in range(C):
                pred = preds[idx]
                pt = aug(pred)
                point = torch.tensor([pt[0], pt[1]], device=device)
                points[idx].append(point)

        points_agg = torch.stack(
            [
                torch.stack(points[pt_idx]).float().mean(dim=0)
                for pt_idx in points.keys()
            ]
        )

        visualize_slot_points(img_t=img[0], points=points_agg)
