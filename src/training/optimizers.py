"""
Optimizer factory.

Creates optimizers by name from config, supporting:
    - SGD (with momentum)
    - Adam
    - AdamW
    - RMSprop

Add new optimizers by extending the factory's `create` method.
"""

import torch
import torch.nn as nn

from utils.config import Config


class OptimizerFactory:
    """
    Factory that creates PyTorch optimizers from config.

    Usage:
        optimizer = OptimizerFactory.create(config.training.optimizer, model)
    """

    @staticmethod
    def create(optim_config: Config, model: nn.Module) -> torch.optim.Optimizer:
        """
        Create an optimizer from config.

        Args:
            optim_config: Config section for the optimizer with at least
                          'name' and 'lr' fields.
            model: The model whose parameters to optimize.

        Returns:
            A PyTorch optimizer instance.
        """
        name = getattr(optim_config, "name", "sgd").lower()
        lr = getattr(optim_config, "lr", 0.005)
        weight_decay = getattr(optim_config, "weight_decay", 0.0005)

        # Only optimize parameters that require gradients
        params = [p for p in model.parameters() if p.requires_grad]

        if name == "sgd":
            momentum = getattr(optim_config, "momentum", 0.9)
            nesterov = getattr(optim_config, "nesterov", False)
            return torch.optim.SGD(
                params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
            )

        elif name == "adam":
            betas = getattr(optim_config, "betas", (0.9, 0.999))
            if isinstance(betas, Config):
                betas = (0.9, 0.999)
            return torch.optim.Adam(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=tuple(betas),
            )

        elif name == "adamw":
            betas = getattr(optim_config, "betas", (0.9, 0.999))
            if isinstance(betas, Config):
                betas = (0.9, 0.999)
            return torch.optim.AdamW(
                params,
                lr=lr,
                weight_decay=weight_decay,
                betas=tuple(betas),
            )

        elif name == "rmsprop":
            momentum = getattr(optim_config, "momentum", 0.0)
            alpha = getattr(optim_config, "alpha", 0.99)
            return torch.optim.RMSprop(
                params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                alpha=alpha,
            )

        else:
            raise ValueError(
                f"Unknown optimizer: '{name}'. "
                f"Available: sgd, adam, adamw, rmsprop"
            )
