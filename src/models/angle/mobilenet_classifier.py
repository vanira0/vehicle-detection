import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.base import BaseClassifier
from models.registry import register_model
from utils.config import Config
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

@register_model("mobilenet_v4_angle_classifier")
class MobileNetV4AngleClassifier(BaseClassifier):
    def build(self, model_config: Config) -> nn.Module:
        num_classes = getattr(model_config, "num_classes", 8)
        pretrained = getattr(model_config, "pretrained", True)
        dropout = getattr(model_config, "dropout", 0.5)

        try:
            import timm
            # Using timm's mobilenetv4
            base_model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=pretrained, num_classes=num_classes, drop_rate=dropout)
            logger.info("Using timm MobileNetV4")
            return base_model
        except ImportError:
            logger.warning("timm not found, falling back to torchvision MobileNetV3 Large")
            weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.mobilenet_v3_large(weights=weights)

            in_features = base_model.classifier[3].in_features
            base_model.classifier[3] = nn.Linear(in_features, num_classes)
            return base_model

    def compute_loss(self, model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        return {"loss": loss}

    def predict(self, model: nn.Module, image: torch.Tensor) -> Dict[str, Any]:
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
        for name, param in model.named_parameters():
            if "classifier" not in name and "head" not in name:
                param.requires_grad = False

    @staticmethod
    def unfreeze_backbone(model: nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad = True
