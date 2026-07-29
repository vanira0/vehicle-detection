import torch.nn as nn
import torchvision.models as models
from models.angle.resnet_classifier import ResNet50AngleClassifier
from models.registry import register_model
from utils.config import Config

@register_model("resnet18_angle_classifier")
class ResNet18AngleClassifier(ResNet50AngleClassifier):
    def build(self, model_config: Config) -> nn.Module:
        num_classes = getattr(model_config, "num_classes", 8)
        pretrained = getattr(model_config, "pretrained", True)
        dropout = getattr(model_config, "dropout", 0.5)

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base_model = models.resnet18(weights=weights)

        in_features = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        return base_model
