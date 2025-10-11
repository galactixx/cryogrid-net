import warnings

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from constants import RESIZE_H, RESIZE_W
from gridbox_net import GridBoxMobileNet
from preprocessing import create_image_centers, split_data
from transforms import PairHorizontalFlip, PairVerticalFlip
from utils import get_dataset_paths

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

horizontal_flip = PairHorizontalFlip()
vertical_flip = PairVerticalFlip()

transforms = [
    (
        lambda i, t: (i, t),
        lambda pred: torch.tensor([pred % RESIZE_W, pred // RESIZE_W], device=device),
    ),
    (
        lambda i, t: horizontal_flip(i, t),
        lambda pred: torch.tensor(
            [RESIZE_W - pred % RESIZE_W, pred // RESIZE_W],
            device=device,
        ),
    ),
    (
        lambda i, t: vertical_flip(i, t),
        lambda pred: torch.tensor(
            [pred % RESIZE_W, RESIZE_H - 1 - pred // RESIZE_W],
            device=device,
        ),
    ),
]


def tta_evaluate(model: GridBoxMobileNet, loader: DataLoader) -> float:
    model.eval()
    positions: Dict[int, List[float]] = dict()

    with torch.no_grad():
        for img_pos, imgs, targets, pts in tqdm(loader, desc="Evaluation"):
            imgs, targets, pts = imgs.to(device), targets.to(device), pts.to(device)

            for t, aug in transforms:
                imgs_new, targets_new = t(imgs, targets)
                logits = model(imgs_new)

                B, C = logits.size(0), logits.size(1)
                logits = logits.view(B, C, -1)
                preds = logits.argmax(dim=2)

                for img_idx in range(B):
                    pos = img_pos[img_idx]
                    if pos not in positions:
                        positions[pos] = []

                    img_pred = preds[img_idx, :].clone()
                    img_pts = pts[img_idx, :]

                    img_dist = 0
                    for idx in range(C):
                        pred = img_pred[idx]
                        pt = aug(pred)

                        dist = torch.dist(pt, img_pts[idx * 2 : idx * 2 + 2])
                        img_dist += dist.item()
                    positions[pos].append(img_dist / C)

        running_dist = sum(np.mean(dists) for dists in positions.values())
        eval_dist = running_dist / len(positions)

    return eval_dist


if __name__ == "__main__":
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

    centers = create_image_centers(images_path, annots)
    dataloaders = split_data(centers)
    test_dist = tta_evaluate(model=model, loader=dataloaders.test)
    print(f"Test avg distance: {test_dist:.3f}...")
