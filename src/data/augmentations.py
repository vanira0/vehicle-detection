"""
Albumentations-based augmentation pipelines.

Supports both classification (image-only) and instance segmentation
(image + masks + bboxes) augmentations. All pipelines are configurable
from YAML configs.

Key advantage of Albumentations over torchvision transforms:
polygon masks are correctly transformed alongside the image during
rotation, flip, and affine operations.
"""

from typing import Optional

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.config import Config


def build_classification_augmentation(
    config: Config,
    is_train: bool = True,
    image_size: int = 224,
) -> A.Compose:
    """
    Build an augmentation pipeline for the Gatekeeper classification model.

    Args:
        config: Config object with `augmentation` section.
        is_train: If True, apply training augmentations. If False, only
                  resize + normalize (for validation/test).
        image_size: Target image size (square).

    Returns:
        Albumentations Compose pipeline.
    """
    aug_cfg = config.augmentation

    transforms = []

    # Resize
    transforms.append(A.Resize(image_size, image_size))

    if is_train:
        # Horizontal flip
        hflip_prob = getattr(aug_cfg, "horizontal_flip", 0.5)
        if hflip_prob > 0:
            transforms.append(A.HorizontalFlip(p=hflip_prob))

        # Vertical flip
        vflip_prob = getattr(aug_cfg, "vertical_flip", 0.0)
        if vflip_prob > 0:
            transforms.append(A.VerticalFlip(p=vflip_prob))

        # Rotation
        rot_limit = getattr(aug_cfg, "rotation_limit", 0)
        if rot_limit > 0:
            transforms.append(A.Rotate(limit=rot_limit, p=0.5))

        # Color jitter
        cj = getattr(aug_cfg, "color_jitter", None)
        if cj:
            transforms.append(
                A.ColorJitter(
                    brightness=getattr(cj, "brightness", 0.2),
                    contrast=getattr(cj, "contrast", 0.2),
                    saturation=getattr(cj, "saturation", 0.2),
                    hue=getattr(cj, "hue", 0.1),
                    p=0.5,
                )
            )

        # Gaussian blur
        gb = getattr(aug_cfg, "gaussian_blur", None)
        if gb:
            transforms.append(
                A.GaussianBlur(
                    blur_limit=getattr(gb, "blur_limit", 3),
                    p=getattr(gb, "prob", 0.1),
                )
            )

    # Normalize (always applied)
    norm = getattr(aug_cfg, "normalize", None)
    if norm:
        transforms.append(
            A.Normalize(
                mean=getattr(norm, "mean", [0.485, 0.456, 0.406]),
                std=getattr(norm, "std", [0.229, 0.224, 0.225]),
            )
        )
    else:
        transforms.append(A.Normalize())

    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def build_segmentation_augmentation(
    config: Config,
    is_train: bool = True,
    image_size: int = 1024,
) -> A.Compose:
    """
    Build an augmentation pipeline for instance segmentation models
    (Damage and Parts). Correctly transforms masks and bounding boxes
    alongside the image.

    Args:
        config: Config object with `augmentation` section.
        is_train: If True, apply training augmentations.
        image_size: Target size for the longest edge.

    Returns:
        Albumentations Compose pipeline with bbox and mask support.
    """
    aug_cfg = config.augmentation

    transforms = []

    # Resize longest edge while maintaining aspect ratio
    transforms.append(A.LongestMaxSize(max_size=image_size))
    transforms.append(
        A.PadIfNeeded(
            min_height=None,
            min_width=None,
            pad_height_divisor=32,   # Ensure divisible by 32 for FPN
            pad_width_divisor=32,
            border_mode=0,           # cv2.BORDER_CONSTANT
            fill=0,
        )
    )

    if is_train:
        # Horizontal flip
        hflip_prob = getattr(aug_cfg, "horizontal_flip", 0.5)
        if hflip_prob > 0:
            transforms.append(A.HorizontalFlip(p=hflip_prob))

        # Random scale
        rs = getattr(aug_cfg, "random_scale", None)
        if rs:
            transforms.append(
                A.RandomScale(
                    scale_limit=(
                        getattr(rs, "min", 0.8) - 1.0,
                        getattr(rs, "max", 1.2) - 1.0,
                    ),
                    p=0.5,
                )
            )

        # Rotation
        rot_limit = getattr(aug_cfg, "rotation_limit", 0)
        if rot_limit > 0:
            transforms.append(A.Rotate(limit=rot_limit, border_mode=0, p=0.5))

        # Color jitter
        cj = getattr(aug_cfg, "color_jitter", None)
        if cj:
            transforms.append(
                A.ColorJitter(
                    brightness=getattr(cj, "brightness", 0.2),
                    contrast=getattr(cj, "contrast", 0.2),
                    saturation=getattr(cj, "saturation", 0.2),
                    hue=getattr(cj, "hue", 0.1),
                    p=0.5,
                )
            )

        # Gaussian blur
        gb = getattr(aug_cfg, "gaussian_blur", None)
        if gb:
            transforms.append(
                A.GaussianBlur(
                    blur_limit=getattr(gb, "blur_limit", 3),
                    p=getattr(gb, "prob", 0.1),
                )
            )

    # Normalize
    norm = getattr(aug_cfg, "normalize", None)
    if norm:
        transforms.append(
            A.Normalize(
                mean=getattr(norm, "mean", [0.485, 0.456, 0.406]),
                std=getattr(norm, "std", [0.229, 0.224, 0.225]),
            )
        )
    else:
        transforms.append(A.Normalize())

    transforms.append(ToTensorV2())

    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(
            format="pascal_voc",       # [x1, y1, x2, y2]
            label_fields=["labels"],
            min_area=1,
            min_visibility=0.1,
        ),
    )


def build_augmentation_pipeline(
    config: Config,
    is_train: bool = True,
) -> A.Compose:
    """
    Factory function that builds the correct augmentation pipeline
    based on the model stage (gatekeeper vs. segmentation).

    Args:
        config: Full experiment config.
        is_train: If True, apply training augmentations.

    Returns:
        Albumentations Compose pipeline.
    """
    stage = config.model.stage
    image_size = getattr(config.data, "image_size", 1024)

    if getattr(config.data, "annotation_format", "") == "folder" or stage in ["gatekeeper", "angle"]:
        return build_classification_augmentation(config, is_train, image_size)
    else:
        return build_segmentation_augmentation(config, is_train, image_size)
