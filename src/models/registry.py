"""
Model registry for strategy pattern.

All models register themselves via the @register_model decorator.
The training loop looks up models by name from the config, enabling
complete model swapping without any code changes.

Usage:
    from models.registry import register_model, get_model

    @register_model("my_custom_model")
    class MyModel(BaseDetector):
        ...

    # At training time:
    model_cls = get_model(config.model.name)
    model_wrapper = model_cls()
    model = model_wrapper.build(config.model)
"""

from typing import Dict, Type

MODEL_REGISTRY: Dict[str, Type] = {}


def register_model(name: str):
    """
    Decorator to register a model class in the global registry.

    Args:
        name: Unique string identifier for the model. This name is
              used in YAML configs (model.name field) to select it.

    Usage:
        @register_model("maskrcnn_resnet50")
        class MaskRCNNResNet50(BaseDetector):
            ...
    """
    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{name}' is already registered by {MODEL_REGISTRY[name].__name__}. "
                f"Cannot register {cls.__name__} with the same name."
            )
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str):
    """
    Retrieve a registered model class by name.

    Args:
        name: Model name as specified in the config.

    Returns:
        The model class (not an instance).

    Raises:
        ValueError: If the model name is not in the registry.
    """
    if name not in MODEL_REGISTRY:
        available = sorted(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{name}' not found in registry. "
            f"Available models: {available}"
        )
    return MODEL_REGISTRY[name]


def list_models():
    """Return a sorted list of all registered model names."""
    return sorted(MODEL_REGISTRY.keys())
