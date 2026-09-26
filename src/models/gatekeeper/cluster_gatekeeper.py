"""
Instrument Cluster Gatekeeper -- Production Binary Classifier.

Classifies whether an image contains a valid vehicle instrument cluster (class 0)
or an invalid/unrelated image (class 1: car exterior, steering wheel, seats,
blurry photo, etc.).

Registered model name: cluster_gatekeeper

Supported architectures (config: model.architecture):
    resnet50           -- ResNet-50 IMAGENET1K_V2  (default, highest accuracy)
    mobilenet_v3_large -- MobileNetV3-Large        (lightweight deployment)
    efficientnet_b0    -- EfficientNet-B0          (efficient alternative)

Classification head attached to backbone features:
    Linear(in_features, 512) -> BatchNorm1d -> ReLU -> Dropout -> Linear(512, num_classes)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

from models.base import BaseClassifier
from models.registry import register_model
from utils.config import Config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom classification head
# ---------------------------------------------------------------------------

class ClusterClassificationHead(nn.Module):
    """
    Deep classification head replacing the backbone's native FC/classifier.

    Architecture:
        Linear(in_features, hidden_dim) -> BatchNorm1d -> ReLU
        -> Dropout(p=dropout_rate) -> Linear(hidden_dim, num_classes)

    Args:
        in_features:  Feature dimension from the backbone.
        hidden_dim:   Intermediate projection size.
        num_classes:  Output classes (2 for binary gatekeeper).
        dropout_rate: Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 512,
        num_classes: int = 2,
        dropout_rate: float = 0.4,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------

def _build_backbone(
    architecture: str,
    pretrained: bool,
    dropout_rate: float,
    num_classes: int,
) -> nn.Module:
    """
    Construct a torchvision backbone with the custom classification head.

    Args:
        architecture: Backbone name string.
        pretrained:   Load ImageNet pretrained weights.
        dropout_rate: Dropout for the custom head.
        num_classes:  Number of output classes.

    Returns:
        nn.Module with the custom head attached.

    Raises:
        ValueError: Unsupported architecture.
    """
    arch = architecture.lower().strip()

    if arch == "resnet50":
        weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model   = tv_models.resnet50(weights=weights)
        in_feat = model.fc.in_features
        model.fc = ClusterClassificationHead(in_feat, 512, num_classes, dropout_rate)

    elif arch in ("mobilenet_v3_large", "mobilenetv3_large", "mobilenet_v3"):
        weights = tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model   = tv_models.mobilenet_v3_large(weights=weights)
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = ClusterClassificationHead(in_feat, 256, num_classes, dropout_rate)

    elif arch in ("efficientnet_b0", "efficientnet"):
        weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model   = tv_models.efficientnet_b0(weights=weights)
        in_feat = model.classifier[-1].in_features
        model.classifier[-1] = ClusterClassificationHead(in_feat, 256, num_classes, dropout_rate)

    else:
        raise ValueError(
            f"Unsupported architecture: '{arch}'. "
            "Choose from: resnet50, mobilenet_v3_large, efficientnet_b0"
        )

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Built '%s' | params: %s | trainable: %s",
                arch, f"{n_total:,}", f"{n_trainable:,}")
    return model


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def _freeze_backbone(model: nn.Module) -> None:
    """Freeze backbone layers; only the custom head remains trainable."""
    head_roots = {"fc", "classifier"}
    frozen, kept = 0, 0
    for name, param in model.named_parameters():
        if name.split(".")[0] in head_roots:
            param.requires_grad = True;  kept   += 1
        else:
            param.requires_grad = False; frozen += 1
    logger.info("Backbone frozen: %d params frozen, %d head params trainable.", frozen, kept)


def _unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters for full end-to-end fine-tuning."""
    for p in model.parameters():
        p.requires_grad = True
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Backbone unfrozen: %s trainable params.", f"{n:,}")


# ---------------------------------------------------------------------------
# Registered model class
# ---------------------------------------------------------------------------

@register_model("cluster_gatekeeper")
class ClusterGatekeeperClassifier(BaseClassifier):
    """
    Production binary classifier for instrument cluster gatekeeper.

    Integrates into the vehicle-detection project via the model registry
    and BaseClassifier interface.

    Config keys (model section):
        architecture    -- resnet50 | mobilenet_v3_large | efficientnet_b0
        pretrained      -- bool (default True)
        num_classes     -- int  (default 2)
        dropout_rate    -- float (default 0.4)
        freeze_backbone -- bool (default True, freeze on build)
    """

    def build(self, model_config: Config) -> nn.Module:
        """
        Construct backbone + custom classification head.

        Args:
            model_config: The 'model' section of the experiment YAML.

        Returns:
            nn.Module ready for training.
        """
        arch     = getattr(model_config, "architecture",  "resnet50")
        pretrain = getattr(model_config, "pretrained",    True)
        nc       = getattr(model_config, "num_classes",   2)
        dropout  = getattr(model_config, "dropout_rate",  0.4)
        freeze   = getattr(model_config, "freeze_backbone", True)

        model = _build_backbone(arch, pretrain, dropout, nc)
        if freeze:
            _freeze_backbone(model)
        return model

    def compute_loss(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass + cross-entropy loss.

        Args:
            model:  Classifier in training mode.
            images: Batch tensors (B, 3, H, W).
            labels: Ground-truth class indices (B,).

        Returns:
            Dict{'loss': scalar tensor}.
        """
        logits = model(images)
        loss   = F.cross_entropy(logits, labels)
        return {"loss": loss}

    def predict(
        self,
        model: nn.Module,
        image: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Single-image inference.

        Args:
            model: Classifier in eval mode.
            image: (3,H,W) or (1,3,H,W) tensor.

        Returns:
            Dict with keys:
                is_valid_cluster  -- bool
                confidence        -- float
                class_probs       -- List[float]
                predicted_class   -- int
                label             -- str
        """
        model.eval()
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            logits = model(image)
            probs  = F.softmax(logits, dim=1).squeeze(0)

        idx        = int(probs.argmax().item())
        confidence = float(probs[idx].item())
        labels     = ["valid_cluster", "invalid"]

        return {
            "is_valid_cluster": idx == 0,
            "confidence":       confidence,
            "class_probs":      probs.cpu().tolist(),
            "predicted_class":  idx,
            "label":            labels[idx],
        }

    @staticmethod
    def freeze_backbone(model: nn.Module) -> None:
        """Freeze backbone; called by Trainer on epoch 0."""
        _freeze_backbone(model)

    @staticmethod
    def unfreeze_backbone(model: nn.Module) -> None:
        """Unfreeze backbone; called by Trainer at freeze_backbone_epochs."""
        _unfreeze_backbone(model)
