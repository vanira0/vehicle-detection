"""
MobileNetV3-based binary classifier for the Gatekeeper stage.

Lightweight alternative to ResNet50 for resource-constrained
deployment scenarios. Faster inference with minimal accuracy loss.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.base import BaseClassifier
from models.registry import register_model
from utils.config import Config


@register_model("mobilenetv3_classifier")
class MobileNetV3Classifier(BaseClassifier):
    """
    MobileNetV3-Large gatekeeper classifier.

    Config options (in model section):
        - num_classes: Number of output classes (default: 2)
        - pretrained: Whether to use ImageNet pretrained weights
        - dropout: Dropout rate before final classifier
        - variant: 'large' or 'small' (default: 'large')
    """

    def build(self, model_config: Config) -> nn.Module:
        num_classes = getattr(model_config, "num_classes", 2)
        pretrained = getattr(model_config, "pretrained", True)
        dropout = getattr(model_config, "dropout", 0.2)
        variant = getattr(model_config, "variant", "large")

        if variant == "small":
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.mobilenet_v3_small(weights=weights)
        else:
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
            base_model = models.mobilenet_v3_large(weights=weights)

        # Replace the final classifier
        in_features = base_model.classifier[-1].in_features
        base_model.classifier[-1] = nn.Sequential(
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
            "is_damaged": predicted_class == 1,
            "confidence": confidence,
            "class_probabilities": probs.cpu().tolist(),
            "predicted_class": predicted_class,
        }
