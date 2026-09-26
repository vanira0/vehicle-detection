# Import all model implementations to trigger registration
from .resnet_classifier import ResNet50Classifier
from .mobilenet_classifier import MobileNetV3Classifier
from .cluster_gatekeeper import ClusterGatekeeperClassifier  # instrument cluster gatekeeper
