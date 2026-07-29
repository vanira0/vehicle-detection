"""
ResNet50-based multi-class classifier for the Angle Detection stage.

Determines the angle (orientation) of a vehicle from an image.
Uses transfer learning from ImageNet-pretrained ResNet50.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.base import BaseClassifier
from models.registry import register_model
from utils.config import Config


@register_model("resnet50_angle_classifier")
class ResNet50AngleClassifier(BaseClassifier):
    """
    ResNet50-based angle classifier.

    Config options (in model section):
        - num_classes: Number of output classes (default: 8)
        - pretrained: Whether to use ImageNet pretrained weights
        - dropout: Dropout rate before final FC layer
        - freeze_backbone_epochs: Number of epochs to freeze backbone
    """

    def build(self, model_config: Config) -> nn.Module:
        num_classes = getattr(model_config, "num_classes", 8)
        pretrained = getattr(model_config, "pretrained", True)
        dropout = getattr(model_config, "dropout", 0.5)

        # Load pretrained ResNet50
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base_model = models.resnet50(weights=weights)

        # Replace the final fully connected layer
        in_features = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        return base_model

    def compute_loss(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        return {"loss": loss}

    def predict(
        self,
        model: nn.Module,
        image: torch.Tensor,
    ) -> Dict[str, Any]:
        model.eval()
        with torch.no_grad():
            if image.dim() == 3:
                image = image.unsqueeze(0)
            logits = model(image)
            probs = F.softmax(logits, dim=1).squeeze(0)

        predicted_class = probs.argmax().item()
        confidence = probs[predicted_class].item()

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "class_probabilities": probs.cpu().tolist(),
        }

    @staticmethod
    def freeze_backbone(model: nn.Module) -> None:
        """Freeze all layers except the final FC layer."""
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

    @staticmethod
    def unfreeze_backbone(model: nn.Module) -> None:
        """Unfreeze all layers for fine-tuning."""
        for param in model.parameters():
            param.requires_grad = True
