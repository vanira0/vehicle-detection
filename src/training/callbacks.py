"""
Training callbacks for checkpointing, early stopping, logging, etc.

Callbacks are pluggable hooks that run at training lifecycle points:
    - on_train_begin / on_train_end
    - on_epoch_end

Add custom callbacks by subclassing BaseCallback.
"""

import os
from typing import Any, Dict, List, Optional

import torch


class BaseCallback:
    """Base class for all callbacks."""

    def on_train_begin(self, trainer) -> None:
        pass

    def on_train_end(self, trainer) -> None:
        pass

    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        trainer,
    ) -> bool:
        """
        Called at the end of each epoch.

        Returns:
            True if training should stop (e.g., early stopping).
        """
        return False


class CallbackList:
    """Container for managing multiple callbacks."""

    def __init__(self, callbacks: List[BaseCallback]):
        self.callbacks = callbacks

    def on_train_begin(self, trainer) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(trainer)

    def on_train_end(self, trainer) -> None:
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        trainer,
    ) -> bool:
        should_stop = False
        for cb in self.callbacks:
            if cb.on_epoch_end(epoch, train_metrics, val_metrics, trainer):
                should_stop = True
        return should_stop


class CheckpointCallback(BaseCallback):
    """
    Save model checkpoints periodically and track the best model.

    Config options:
        - save_every: Save a checkpoint every N epochs
        - save_best: Save the best model based on a metric
        - metric: Metric name to track for best model
        - mode: 'max' (higher is better) or 'min' (lower is better)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_every: int = 5,
        save_best: bool = True,
        metric: str = "loss",
        mode: str = "min",
    ):
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.save_best = save_best
        self.metric = metric
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")

        os.makedirs(checkpoint_dir, exist_ok=True)

    def on_epoch_end(self, epoch, train_metrics, val_metrics, trainer) -> bool:
        # Periodic checkpoint
        if (epoch + 1) % self.save_every == 0:
            path = os.path.join(self.checkpoint_dir, f"epoch_{epoch:04d}.pth")
            trainer.save_checkpoint(path, {**train_metrics, **val_metrics})

        # Best model checkpoint
        if self.save_best:
            # Look for metric in val_metrics first, then train_metrics
            current_value = val_metrics.get(
                self.metric, train_metrics.get(self.metric)
            )

            if current_value is None:
                return False

            is_better = (
                (self.mode == "min" and current_value < self.best_value)
                or (self.mode == "max" and current_value > self.best_value)
            )

            if is_better:
                self.best_value = current_value
                trainer.best_metric = current_value
                path = os.path.join(self.checkpoint_dir, "best.pth")
                trainer.save_checkpoint(path, {**train_metrics, **val_metrics})
                trainer.logger.info(
                    f"New best model! {self.metric}: {current_value:.4f}"
                )

        return False


class EarlyStoppingCallback(BaseCallback):
    """
    Stop training if a monitored metric doesn't improve for `patience` epochs.
    """

    def __init__(
        self,
        patience: int = 10,
        metric: str = "loss",
        mode: str = "min",
        min_delta: float = 1e-4,
    ):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")

    def on_epoch_end(self, epoch, train_metrics, val_metrics, trainer) -> bool:
        current_value = val_metrics.get(
            self.metric, train_metrics.get(self.metric)
        )

        if current_value is None:
            return False

        if self.mode == "min":
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            trainer.logger.info(
                f"EarlyStopping: {self.counter}/{self.patience} "
                f"(best {self.metric}: {self.best_value:.4f})"
            )

        return self.counter >= self.patience


class LoggingCallback(BaseCallback):
    """
    Log training metrics to console with formatted output.
    """

    def on_epoch_end(self, epoch, train_metrics, val_metrics, trainer) -> bool:
        lr = trainer.optimizer.param_groups[0]["lr"]
        msg = f"Epoch {epoch:4d} | LR: {lr:.6f}"

        for k, v in train_metrics.items():
            msg += f" | train_{k}: {v:.4f}"

        for k, v in val_metrics.items():
            msg += f" | val_{k}: {v:.4f}"

        trainer.logger.info(msg)
        return False


class TensorBoardCallback(BaseCallback):
    """
    Log metrics to TensorBoard for visualization.
    """

    def __init__(self, log_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        except ImportError:
            self.writer = None

    def on_epoch_end(self, epoch, train_metrics, val_metrics, trainer) -> bool:
        if self.writer is None:
            return False

        for k, v in train_metrics.items():
            self.writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            self.writer.add_scalar(f"val/{k}", v, epoch)

        self.writer.add_scalar(
            "lr", trainer.optimizer.param_groups[0]["lr"], epoch
        )
        return False

    def on_train_end(self, trainer) -> None:
        if self.writer:
            self.writer.close()
