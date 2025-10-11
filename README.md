# GridBox Vision Pipeline

A computer vision pipeline that automatically detects the center coordinates of each slot in a grid box used for CRYO-EM protein freezing. Uses U-Net with MobileNetV2 backbone to predict four slot centers (left, top, right, bottom) via heatmap regression.

## Usage

```bash
python train_mobilenetv2.py  # Training
python test_mobilenetv2.py   # Testing
```

## Overview

This model automates slot center detection in grid boxes for cryo-electron microscopy (CRYO-EM) protein sample preparation. While showing promising results, the project was pivoted to a hardware-based pipeline solution.

## Problem Statement

CRYO-EM workflows require precise identification of slot centers in grid boxes for:
- Automated sample positioning
- Precise robotic handling  
- Quality control

The challenge: detect four distinct slot centers despite variations in lighting, orientation, and positioning.

## Model Architecture

- **Backbone**: MobileNetV2 encoder
- **Architecture**: U-Net decoder for spatial localization
- **Output**: 4-channel heatmap prediction (one per slot center)
- **Input**: RGB images resized to 960×512 pixels
- **Detection**: Rhombus-shaped heatmaps for robust center prediction

## Dataset

- **Images**: Grid box photographs with varying conditions
- **Annotations**: CSV with slot center coordinates (X, Y) for each slot type (L, T, R, B)
- **Format**: Each image contains 4 slot center annotations
- **Split**: Train/validation/test based on grid box positions

```
Position,File,Slot,X,Y
position-000001,img_000001.jpg,B,1186,817
position-000001,img_000001.jpg,L,997,711
```

## Training

- **Loss**: Focal Loss (α=1.0, γ=2.0) for class imbalance
- **Strategy**: Progressive unfreezing of encoder layers
- **Augmentation**: Paired transforms (flips, brightness/contrast)
- **Optimization**: AdamW with EMA and mixed precision

## Results

- **Dataset**: 996 images collected
- **Test Performance**: 5.379 pixel average error
- **Relative Error**: 0.56% (5.379/960 max dimension of 960×512 image)