"""
Model-agnostic training loop.

The Trainer doesn't know which model it's training — it calls the
BaseDetector / BaseClassifier interface methods. This decouples the
training logic from the model implementation entirely.

Supports:
    - Detection models (Mask R-CNN etc.) via BaseDetector
    - Classification models (Gatekeeper) via BaseClassifier
    - Automatic Mixed Precision (AMP)
    - Gradient accumulation
    - Gradient clipping
    - Pluggable callbacks (checkpoint, early stopping, logging)
    - Automatic experiment directory management
"""

import os
import time
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseClassifier, BaseDetector
from utils.config import Config
from utils.logger import setup_logger, MetricsLogger
from utils.seed import set_seed

from .optimizers import OptimizerFactory
from .schedulers import SchedulerFactory
from .callbacks import CallbackList


class Trainer:
    """
    Generic trainer for any model implementing BaseDetector or BaseClassifier.

    Usage:
        config = Config.from_file("configs/damage/maskrcnn_resnet50.yaml")
        model_wrapper = get_model(config.model.name)()
        trainer = Trainer(config, model_wrapper, train_loader, val_loader)
        trainer.fit()
    """

    def __init__(
        self,
        config: Config,
        model_wrapper,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        callbacks: Optional[List] = None,
        evaluator=None,
    ):
        self.config = config
        self.model_wrapper = model_wrapper
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = evaluator

        # Determine if this is a classification or detection model
        self.is_classifier = isinstance(model_wrapper, BaseClassifier)

        # Setup device
        device_cfg = config.get("inference.device", None)
        if device_cfg:
            self.device = torch.device(device_cfg)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup experiment directory
        exp_name = getattr(config, "experiment_name", "experiment")
        runs_dir = config.get("output.runs_dir", "runs")
        self.experiment_dir = os.path.join(runs_dir, exp_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.experiment_dir, "metrics"), exist_ok=True)

        # Save frozen config snapshot
        config.save(os.path.join(self.experiment_dir, "config.yaml"))

        # Setup logging
        self.logger = setup_logger(
            name=exp_name,
            log_dir=os.path.join(self.experiment_dir, "logs"),
            level=config.get("logging.level", "INFO"),
        )
        self.metrics_logger = MetricsLogger(
            os.path.join(self.experiment_dir, "metrics")
        )

        # Set seed for reproducibility
        seed = getattr(config, "seed", 42)
        set_seed(seed)

        # Build model
        self.logger.info(f"Building model: {config.model.name}")
        self.model = model_wrapper.build(config.model).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

        # Setup optimizer and scheduler
        self.optimizer = OptimizerFactory.create(config.training.optimizer, self.model)
        self.scheduler = SchedulerFactory.create(
            config.training.scheduler, self.optimizer
        )

        # AMP scaler
        amp_enabled = getattr(config.training, "amp", True)
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and self.device.type == "cuda")
        self.amp_enabled = amp_enabled and self.device.type == "cuda"

        # Training params
        self.gradient_accumulation = getattr(config.training, "gradient_accumulation", 1)
        self.gradient_clip_norm = config.get("training.gradient_clip_norm", None)
        self.num_epochs = getattr(config.training, "epochs", 50)

        # Callbacks
        self.callbacks = CallbackList(callbacks or [])

        # State tracking
        self.current_epoch = 0
        self.best_metric = float("-inf")
        self.global_step = 0

        self.logger.info(f"Trainer initialized on device: {self.device}")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")

    def fit(self) -> Dict:
        """
        Main training loop. Runs for num_epochs, calling callbacks
        at each epoch boundary.

        Returns:
            Dict with final training metrics.
        """
        self.logger.info(f"Starting training for {self.num_epochs} epochs")
        self.callbacks.on_train_begin(self)

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Handle backbone freezing for gatekeeper
            if self.is_classifier:
                freeze_epochs = self.config.get("model.freeze_backbone_epochs", 0)
                if freeze_epochs and epoch == 0:
                    self.logger.info(f"Freezing backbone for {freeze_epochs} epochs")
                    if hasattr(self.model_wrapper, "freeze_backbone"):
                        self.model_wrapper.freeze_backbone(self.model)
                elif freeze_epochs and epoch == freeze_epochs:
                    self.logger.info("Unfreezing backbone for fine-tuning")
                    if hasattr(self.model_wrapper, "unfreeze_backbone"):
                        self.model_wrapper.unfreeze_backbone(self.model)
                        # Rebuild optimizer with all parameters
                        self.optimizer = OptimizerFactory.create(
                            self.config.training.optimizer, self.model
                        )

            # Train one epoch
            train_metrics = self._train_one_epoch(epoch)

            # Validate
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self._validate(epoch)

            # Update scheduler
            scheduler_name = getattr(self.config.training.scheduler, "name", "step")
            if scheduler_name == "plateau" and val_metrics:
                metric_name = self.config.get("training.checkpoint.metric", "loss")
                metric_val = val_metrics.get(metric_name, train_metrics.get("loss", 0))
                self.scheduler.step(metric_val)
            else:
                self.scheduler.step()

            # Log metrics
            all_metrics = {
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.metrics_logger.log(all_metrics)

            # Callbacks
            should_stop = self.callbacks.on_epoch_end(
                epoch, train_metrics, val_metrics, self
            )
            if should_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        self.callbacks.on_train_end(self)
        self.logger.info("Training complete!")
        return all_metrics

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        epoch_losses = {}
        num_batches = 0

        log_every = self.config.get("logging.log_every_n_steps", 10)

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.num_epochs - 1}",
            leave=True,
        )

        for batch_idx, batch in enumerate(pbar):
            # Dispatch to the right training method
            if self.is_classifier:
                loss_dict = self._train_step_classifier(batch)
            else:
                loss_dict = self._train_step_detector(batch)

            # Accumulate losses for logging
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value
            num_batches += 1

            # Gradient accumulation step
            total_loss = sum(loss_dict.values()) / self.gradient_accumulation

            self.scaler.scale(torch.tensor(total_loss)).backward() if False else None

            if (batch_idx + 1) % self.gradient_accumulation == 0:
                if self.gradient_clip_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            self.global_step += 1

            # Update progress bar
            pbar_losses = {k: f"{v / num_batches:.4f}" for k, v in epoch_losses.items()}
            pbar.set_postfix(pbar_losses)

        # Average losses
        avg_losses = {k: v / max(num_batches, 1) for k, v in epoch_losses.items()}
        self.logger.info(
            f"Epoch {epoch} | "
            + " | ".join(f"{k}: {v:.4f}" for k, v in avg_losses.items())
        )
        return avg_losses

    def _train_step_classifier(self, batch) -> Dict[str, float]:
        """Single training step for classification models."""
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        with torch.amp.autocast("cuda", enabled=self.amp_enabled):
            loss_dict = self.model_wrapper.compute_loss(self.model, images, labels)

        losses = sum(loss_dict.values())
        self.scaler.scale(losses).backward()

        return {k: v.item() for k, v in loss_dict.items()}

    def _train_step_detector(self, batch) -> Dict[str, float]:
        """Single training step for detection models."""
        images, targets = batch
        images = [img.to(self.device) for img in images]
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast("cuda", enabled=self.amp_enabled):
            loss_dict = self.model_wrapper.compute_loss(self.model, images, targets)

        losses = sum(loss for loss in loss_dict.values())
        self.scaler.scale(losses).backward()

        return {k: v.item() for k, v in loss_dict.items()}

    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()

        if self.evaluator is not None:
            return self.evaluator.evaluate(self.model, self.val_loader, self.device)

        # Basic validation loss computation
        val_losses = {}
        num_batches = 0

        for batch in self.val_loader:
            if self.is_classifier:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                # For classification, compute val loss
                self.model.train()  # Temporarily set to train for loss computation
                with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                    loss_dict = self.model_wrapper.compute_loss(self.model, images, labels)
                self.model.eval()

                # Also compute accuracy
                with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                    logits = self.model(images)
                    preds = logits.argmax(dim=1)
                    correct = (preds == labels).sum().item()
                    total = labels.size(0)
                    loss_dict["accuracy"] = torch.tensor(correct / total)

            else:
                images, targets = batch
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Detection models need train mode to return losses
                self.model.train()
                with torch.amp.autocast("cuda", enabled=self.amp_enabled):
                    loss_dict = self.model_wrapper.compute_loss(self.model, images, targets)
                self.model.eval()

            for key, value in loss_dict.items():
                v = value.item() if isinstance(value, torch.Tensor) else value
                if key not in val_losses:
                    val_losses[key] = 0.0
                val_losses[key] += v
            num_batches += 1

        avg_losses = {k: v / max(num_batches, 1) for k, v in val_losses.items()}
        self.logger.info(
            f"Validation | "
            + " | ".join(f"{k}: {v:.4f}" for k, v in avg_losses.items())
        )
        return avg_losses

    def save_checkpoint(self, path: str, metrics: Optional[Dict] = None) -> None:
        """Save a training checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "config": self.config.to_dict(),
        }
        if metrics:
            checkpoint["metrics"] = metrics
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint to resume training."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint.get("global_step", 0)
        self.best_metric = checkpoint.get("best_metric", float("-inf"))
        self.logger.info(f"Resumed from checkpoint: {path} (epoch {self.current_epoch})")
