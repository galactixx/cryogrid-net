"""
Visualization utilities for CryogridNet predictions and results.

Provides functions for visualizing slot center predictions on grid box images.
Includes utilities for denormalizing images from ImageNet preprocessing, plotting
predicted slot centers with customizable markers, and saving visualization results.
"""

import os
from pathlib import Path
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from constants import INET_MEAN, INET_STD, NUM_CENTERS


def visualize_slot_points(
    img_t: torch.Tensor,
    points: List[torch.Tensor],
    save: bool = False,
    path: Optional[Path] = None,
    filename: Optional[str] = None,
    point_color: str = "red",
    point_size: int = 30,
) -> None:
    """Visualize predicted slot centers overlaid on the input image."""
    # Denormalize image from ImageNet preprocessing
    img = img_t.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * np.array(INET_STD) + np.array(INET_MEAN)).clip(0, 1)

    labels = ["Left", "Top", "Right", "Bottom"]
    plt.figure(figsize=(4 * NUM_CENTERS, NUM_CENTERS))

    col = 1

    # Plot each individual point
    for i, label in enumerate(labels):
        pt = points[i].cpu()
        ax = plt.subplot(1, NUM_CENTERS, col)
        ax.imshow(img)
        ax.scatter(pt[0], pt[1], c=point_color, s=point_size, edgecolors="black")
        ax.set_title(f"{label} Point")
        ax.axis("off")
        col += 1

    plt.tight_layout()
    if save:
        img_path = Path(os.getcwd() if path is None else path)
        plt.savefig(img_path / filename, dpi=300, bbox_inches="tight")
    plt.show()
