import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from models.base import BaseClassifier
from models.registry import register_model
from utils.config import Config
from typing import Dict, Any

@register_model("efficientnet_b0_angle_classifier")
class EfficientNetB0AngleClassifier(BaseClassifier):
    def build(self, model_config: Config) -> nn.Module:
        num_classes = getattr(model_config, "num_classes", 8)
        pretrained = getattr(model_config, "pretrained", True)
        dropout = getattr(model_config, "dropout", 0.5)

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.efficientnet_b0(weights=weights)

        in_features = base_model.classifier[1].in_features
        base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
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
            if "classifier" not in name:
                param.requires_grad = False

    @staticmethod
    def unfreeze_backbone(model: nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad = True
