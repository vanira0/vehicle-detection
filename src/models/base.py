"""
Abstract base classes for all models in the pipeline.

Every model must implement one of these interfaces. The Trainer
calls these methods without knowing which concrete model it's
training — that's the core of the strategy pattern.

Two base classes:
    - BaseDetector: For instance segmentation models (Stages 2 & 3)
    - BaseClassifier: For the binary classification gatekeeper (Stage 1)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from utils.config import Config


class BaseDetector(ABC):
    """
    Abstract base for all detection/segmentation models.

    Implementations handle:
        - Model construction (build)
        - Loss computation (compute_loss)
        - Inference post-processing (post_process)
        - Model-specific transforms (get_transform)
    """

    @abstractmethod
    def build(self, model_config: Config) -> nn.Module:
        """
        Construct and return the model.

        Args:
            model_config: The `model` section of the experiment config.

        Returns:
            A PyTorch nn.Module ready for training.
        """
        ...

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        images: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass in training mode + loss computation.

        Args:
            model: The model in training mode.
            images: List of image tensors.
            targets: List of target dicts with boxes, labels, masks.

        Returns:
            Dict of named losses (e.g., loss_classifier, loss_box_reg,
            loss_mask, loss_objectness, loss_rpn_box_reg).
        """
        ...

    @abstractmethod
    def post_process(
        self,
        predictions: List[Dict[str, torch.Tensor]],
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Post-process raw model outputs into standardized format.

        Args:
            predictions: Raw model output (list of dicts per image).
            confidence_threshold: Minimum confidence score to keep.
            nms_threshold: IoU threshold for non-maximum suppression.

        Returns:
            List of processed prediction dicts with keys:
                - boxes: np.ndarray (N, 4) in [x1, y1, x2, y2]
                - labels: np.ndarray (N,) of class indices
                - scores: np.ndarray (N,) of confidence scores
                - masks: np.ndarray (N, H, W) of binary masks
        """
        ...

    def get_transform(self, is_train: bool = True):
        """
        Return any model-specific data transforms.
        Default: None (use the shared augmentation pipeline).
        Override if the model needs custom preprocessing.
        """
        return None


class BaseClassifier(ABC):
    """
    Abstract base for classification models (Gatekeeper).
    """

    @abstractmethod
    def build(self, model_config: Config) -> nn.Module:
        """
        Construct and return the classifier model.

        Args:
            model_config: The `model` section of the experiment config.

        Returns:
            A PyTorch nn.Module ready for training.
        """
        ...

    @abstractmethod
    def compute_loss(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass + classification loss.

        Args:
            model: The classifier model in training mode.
            images: Batch of image tensors (B, 3, H, W).
            labels: Ground truth labels (B,).

        Returns:
            Dict with at least {"loss": tensor}.
        """
        ...

    @abstractmethod
    def predict(
        self,
        model: nn.Module,
        image: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Run inference on a single image.

        Args:
            model: The classifier model in eval mode.
            image: Single image tensor (1, 3, H, W) or (3, H, W).

        Returns:
            Dict with keys:
                - is_damaged: bool
                - confidence: float
                - class_probabilities: list of floats
        """
        ...

    def get_transform(self, is_train: bool = True):
        """Return model-specific transforms. Default: None."""
        return None
