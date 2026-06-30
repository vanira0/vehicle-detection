"""
Learning rate scheduler factory.

Creates LR schedulers by name from config, supporting:
    - StepLR
    - CosineAnnealingLR
    - OneCycleLR
    - ReduceLROnPlateau
    - MultiStepLR
    - ExponentialLR

Add new schedulers by extending the factory's `create` method.
"""

import torch
from torch.optim.lr_scheduler import _LRScheduler

from utils.config import Config


class SchedulerFactory:
    """
    Factory that creates PyTorch LR schedulers from config.

    Usage:
        scheduler = SchedulerFactory.create(config.training.scheduler, optimizer)
    """

    @staticmethod
    def create(
        sched_config: Config,
        optimizer: torch.optim.Optimizer,
        steps_per_epoch: int = 0,
    ) -> _LRScheduler:
        """
        Create an LR scheduler from config.

        Args:
            sched_config: Config section for the scheduler with at least 'name'.
            optimizer: The optimizer to schedule.
            steps_per_epoch: Number of training steps per epoch (for OneCycleLR).

        Returns:
            A PyTorch LR scheduler instance.
        """
        name = getattr(sched_config, "name", "step").lower()

        if name == "step":
            step_size = getattr(sched_config, "step_size", 3)
            gamma = getattr(sched_config, "gamma", 0.1)
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )

        elif name == "multistep":
            milestones = getattr(sched_config, "milestones", [30, 60, 90])
            gamma = getattr(sched_config, "gamma", 0.1)
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=gamma
            )

        elif name == "cosine":
            T_max = getattr(sched_config, "T_max", 50)
            eta_min = getattr(sched_config, "eta_min", 0.0)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=eta_min
            )

        elif name == "onecycle":
            max_lr = getattr(sched_config, "max_lr", 0.01)
            epochs = getattr(sched_config, "epochs", 50)
            if steps_per_epoch <= 0:
                raise ValueError(
                    "OneCycleLR requires steps_per_epoch > 0. "
                    "Pass the number of training batches."
                )
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lr,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
            )

        elif name == "plateau":
            patience = getattr(sched_config, "patience", 5)
            factor = getattr(sched_config, "factor", 0.1)
            mode = getattr(sched_config, "mode", "min")
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode=mode, factor=factor, patience=patience
            )

        elif name == "exponential":
            gamma = getattr(sched_config, "gamma", 0.95)
            return torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=gamma
            )

        else:
            raise ValueError(
                f"Unknown scheduler: '{name}'. "
                f"Available: step, multistep, cosine, onecycle, plateau, exponential"
            )
