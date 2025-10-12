"""
Configuration constants for the CryogridNet project.

Contains global constants for reproducibility, ImageNet normalization,
image dimensions, and file naming conventions.
"""

# Random seed for reproducible results
SEED = 42

# ImageNet normalization values (RGB channels)
INET_MEAN = [0.485, 0.456, 0.406]
INET_STD = [0.229, 0.224, 0.225]

# Target image dimensions for model input
RESIZE_W, RESIZE_H = 960, 512
# Original image dimensions
W, H = 1920, 1080

# Scaling factors for coordinate transformation
SCALE_X, SCALE_Y = RESIZE_W / W, RESIZE_H / H

# Dataset file naming conventions
ANNOTATIONS_FILENAME = "annotations.csv"
IMAGES_FOLDER = "images"
