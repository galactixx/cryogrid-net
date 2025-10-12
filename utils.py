"""
Utility functions and classes for CryogridNet project management.

Provides essential utilities including dataset path resolution from HuggingFace Hub,
progressive layer unfreezing for transfer learning, reproducible random seeding,
and parameter group management for optimizer configuration.
"""

import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.optim import AdamW

from constants import ANNOTATIONS_FILENAME, IMAGES_FOLDER


def get_dataset_paths() -> Tuple[Path, Path]:
    snapshot_path = snapshot_download(
        "galactixx/cryogrid-boxes",
        repo_type="dataset",
        local_dir="data",
        token=False,
        local_dir_use_symlinks=False,
    )
    snapshot_path = Path(snapshot_path)
    images_path = snapshot_path / IMAGES_FOLDER
    annots_path = snapshot_path / ANNOTATIONS_FILENAME
    return images_path, annots_path


def unfreeze_layer(layer: torch.nn.Module) -> None:
    """Unfreeze parameters and enable training for BatchNorm in a layer prefix."""
    for param in layer.parameters():
        param.requires_grad = True

    for _, module in layer.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()


@dataclass(frozen=True)
class ParamGroup:
    """Optimizer hyperparameters associated with a layer to unfreeze at an epoch."""

    epoch: int
    layer: torch.nn.Module
    lr: float
    decay: float

    @property
    def group(self) -> Dict[str, Any]:
        return {
            "params": self.layer.parameters(),
            "lr": self.lr,
            "weight_decay": self.decay,
        }


class ProgressiveUnfreezer:
    """Manage progressive unfreezing by epoch and add optimizer param groups."""

    def __init__(self, params: List[ParamGroup]) -> None:
        self.params = deque(sorted(params, key=lambda x: x.epoch))
        self._optimizer: Optional[AdamW] = None

    @property
    def optimizer(self) -> AdamW:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: AdamW) -> None:
        self._optimizer = optimizer

    def unfreeze(self, epoch: int) -> None:
        """If the current epoch matches the schedule, unfreeze next layer."""
        # Nothing left to unfreeze
        if not len(self.params):
            return None

        top = self.params[0]
        if epoch == top.epoch:
            unfreeze_layer(layer=top.layer)
            # Start optimizing the newly unfrozen parameters
            self.optimizer.add_param_group(top.group)
            self.params.popleft()
            print("Unfreezing layer...")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducibility (incl. CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Make CUDA algorithms deterministic where possible
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
