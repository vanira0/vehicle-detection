"""
Training strategy abstractions.

Defines different training strategies that can be swapped via config:
    - StandardTrainingStrategy: Default PyTorch training loop
    - YOLOv8TrainingStrategy: Delegates to Ultralytics native training

The Trainer uses the strategy to handle model-specific training quirks
without polluting the core training loop.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from utils.config import Config


class BaseTrainingStrategy(ABC):
    """
    Abstract training strategy.

    Strategies handle model-specific training setups that don't fit
    the generic Trainer pattern (e.g., YOLOv8's native training API).
    """

    @abstractmethod
    def setup(self, config: Config, model_wrapper: Any) -> None:
        """Perform any strategy-specific setup."""
        ...

    @abstractmethod
    def train(self, config: Config, **kwargs) -> Dict[str, Any]:
        """Execute the full training loop."""
        ...


class StandardTrainingStrategy(BaseTrainingStrategy):
    """
    Standard PyTorch training strategy.

    Uses the generic Trainer class — no special handling needed.
    This is the default strategy for Mask R-CNN and classification models.
    """

    def setup(self, config: Config, model_wrapper: Any) -> None:
        # No special setup needed; the Trainer handles everything
        pass

    def train(self, config: Config, **kwargs) -> Dict[str, Any]:
        # This strategy delegates to the Trainer directly
        # The train.py script handles this case
        raise NotImplementedError(
            "StandardTrainingStrategy uses the generic Trainer. "
            "Call Trainer.fit() directly."
        )


class YOLOv8TrainingStrategy(BaseTrainingStrategy):
    """
    YOLOv8 native training strategy.

    Bypasses the generic Trainer and uses Ultralytics' built-in
    training loop, which includes its own augmentation, scheduling,
    and logging.
    """

    def setup(self, config: Config, model_wrapper: Any) -> None:
        self.model_wrapper = model_wrapper
        self.config = config

    def train(self, config: Config, **kwargs) -> Dict[str, Any]:
        data_yaml = kwargs.get("data_yaml")
        if data_yaml is None:
            raise ValueError(
                "YOLOv8TrainingStrategy requires 'data_yaml' kwarg "
                "pointing to a YOLO-format data.yaml file."
            )

        results = self.model_wrapper.train_with_ultralytics(
            data_yaml=data_yaml,
            model_config=config.model,
            training_config=config.training,
        )
        return {"results": results}


# Strategy registry
TRAINING_STRATEGIES = {
    "standard": StandardTrainingStrategy,
    "yolov8": YOLOv8TrainingStrategy,
}


def get_training_strategy(name: str = "standard") -> BaseTrainingStrategy:
    """Get a training strategy by name."""
    if name not in TRAINING_STRATEGIES:
        raise ValueError(
            f"Unknown training strategy: '{name}'. "
            f"Available: {list(TRAINING_STRATEGIES.keys())}"
        )
    return TRAINING_STRATEGIES[name]()
