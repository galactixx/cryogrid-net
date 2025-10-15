"""
Neural network architecture definitions for CryogridNet.

Implements U-Net architectures with DenseNet121 and MobileNetV2 encoders for
slot center detection. Provides GridBoxNet, GridBoxDenseNet, and
GridBoxMobileNet classes that output 4-channel heatmaps corresponding
to the four slot positions (left, top, right, bottom).
"""

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class GridBoxNet(nn.Module):
    """Base U-Net model for slot center detection with configurable encoder."""

    def __init__(
        self,
        encoder: str,
        pretrained: bool = True,
        num_classes: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_weights = "imagenet" if pretrained else None
        self.num_classes = num_classes

        self.backbone = smp.Unet(
            encoder_name=self.encoder,
            encoder_weights=self.encoder_weights,
            in_channels=3,
            classes=self.num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the U-Net model."""
        x = self.backbone(x)
        return x


class GridBoxMobileNet(GridBoxNet):
    """MobileNetV2-based U-Net model optimized for slot center detection."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__(encoder="mobilenet_v2", pretrained=pretrained, num_classes=4)


class GridBoxDenseNet(GridBoxNet):
    """DenseNet121-based U-Net model optimized for slot center detection."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__(encoder="densenet121", pretrained=pretrained, num_classes=4)
