"""
Neural network architecture definitions for CryogridNet.

Implements U-Net architecture with MobileNetV2 backbone for slot center detection.
Provides GridBoxNet and GridBoxMobileNet classes that output 4-channel heatmaps
corresponding to the four slot positions (left, top, right, bottom).
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class GridBoxNet(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: bool = True,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = backbone
        self.encoder_weights = "imagenet" if pretrained else None
        self.num_classes = num_classes

        self.backbone = smp.Unet(
            encoder_name=self.encoder,
            encoder_weights=self.encoder_weights,
            in_channels=3,
            classes=self.num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return x


class GridBoxMobileNet(GridBoxNet):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__(backbone="mobilenet_v2", pretrained=pretrained, num_classes=4)
