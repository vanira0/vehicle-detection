"""
Image preprocessing utilities for inference.

Functions for loading, resizing, normalizing, and converting images
to the format expected by each model type.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk as a RGB numpy array.

    Args:
        path: Path to the image file.

    Returns:
        Image as numpy array (H, W, 3) in RGB format.

    Raises:
        FileNotFoundError: If the image file doesn't exist.
        ValueError: If the image cannot be decoded.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_image(
    image: np.ndarray,
    target_size: int,
    keep_aspect_ratio: bool = True,
) -> np.ndarray:
    """
    Resize an image to target size.

    Args:
        image: Input image (H, W, 3).
        target_size: Target size for the longest edge (if keep_aspect_ratio)
                     or both dimensions (if not).
        keep_aspect_ratio: If True, resize the longest edge to target_size
                           while maintaining aspect ratio.

    Returns:
        Resized image.
    """
    h, w = image.shape[:2]

    if keep_aspect_ratio:
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
    else:
        new_h, new_w = target_size, target_size

    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def preprocess_image(
    image: np.ndarray,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    target_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Full preprocessing pipeline: resize → normalize → to tensor.

    Args:
        image: Input image (H, W, 3) in RGB, uint8 [0, 255].
        mean: Channel means for normalization.
        std: Channel stds for normalization.
        target_size: Optional resize target. If None, no resize.

    Returns:
        Preprocessed image tensor (1, 3, H, W), float32, normalized.
    """
    if target_size is not None:
        image = resize_image(image, target_size, keep_aspect_ratio=True)

    # Convert to float32 [0, 1]
    image = image.astype(np.float32) / 255.0

    # Normalize
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    image = (image - mean) / std

    # HWC → CHW, add batch dim
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    return tensor


def preprocess_for_detection(
    image: np.ndarray,
    target_size: int = 1024,
) -> torch.Tensor:
    """
    Preprocess an image for detection/segmentation models.
    Detection models (Mask R-CNN) expect images as float32 tensors
    in [0, 1] range WITHOUT normalization (the model handles it internally).

    Args:
        image: Input image (H, W, 3) in RGB, uint8 [0, 255].
        target_size: Target size for the longest edge.

    Returns:
        Image tensor (3, H, W), float32, in [0, 1].
    """
    image = resize_image(image, target_size, keep_aspect_ratio=True)
    image = image.astype(np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1)
    return tensor
