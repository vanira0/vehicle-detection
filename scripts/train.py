#!/usr/bin/env python
"""
Train a model using a YAML configuration file.

Usage:
    # Train with a specific config
    python scripts/train.py --config configs/damage/maskrcnn_resnet50.yaml

    # Override specific parameters
    python scripts/train.py --config configs/damage/maskrcnn_resnet50.yaml \
        --set training.optimizer.lr=0.001 training.epochs=30

    # Resume from checkpoint
    python scripts/train.py --config configs/damage/maskrcnn_resnet50.yaml \
        --resume runs/damage_maskrcnn_v1/checkpoints/epoch_0015.pth
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from torch.utils.data import DataLoader

from data.augmentations import build_augmentation_pipeline
from data.dataset import ClassificationDataset, COCOSegmentationDataset, collate_fn
from evaluation.evaluator import Evaluator
from models.registry import get_model

# Import model packages to trigger registration
import models.gatekeeper
import models.angle
import models.damage      # noqa: F401
import models.parts       # noqa: F401
import models.vehicle     # noqa: F401
import models.yolo_segmentation # noqa: F401
import models.maskrcnn_segmentation # noqa: F401

from training.callbacks import (
    CallbackList,
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    TensorBoardCallback,
)
from training.trainer import Trainer
from utils.config import Config
from utils.logger import setup_logger
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a vehicle damage detection model"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML experiment config file",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--set", nargs="*", default=[],
        help="Override config values: --set key1=val1 key2=val2",
    )
    return parser.parse_args()


def resolve_data_loader_settings(data_cfg):
    """Return dataloader settings that are robust for this training setup."""
    configured_workers = getattr(data_cfg, "num_workers", None)
    if configured_workers is None:
        configured_workers = 0

    try:
        num_workers = int(configured_workers)
    except (TypeError, ValueError):
        num_workers = 0

    if num_workers < 0:
        num_workers = 0

    # The project occasionally crashes in Linux/WSL-style environments when
    # PyTorch multiprocessing workers load images and annotations. Using a
    # single process is more reliable for training runs.
    if num_workers > 0 and os.name == "posix":
        num_workers = 0

    pin_memory = getattr(data_cfg, "pin_memory", False)
    return num_workers, pin_memory


def build_data_loaders(config):
    """Build train and validation data loaders from config."""
    stage = config.model.stage
    data_cfg = config.data
    batch_size = config.training.batch_size
    num_workers, pin_memory = resolve_data_loader_settings(data_cfg)

    if getattr(data_cfg, "annotation_format", "") == "folder" or stage in ["gatekeeper", "angle"]:
        # Folder-based classification dataset
        train_transform = build_augmentation_pipeline(config, is_train=True)
        val_transform = build_augmentation_pipeline(config, is_train=False)

        train_root = os.path.join(data_cfg.root, "train")
        val_root = os.path.join(data_cfg.root, "val")
        if not os.path.exists(val_root) and os.path.exists(os.path.join(data_cfg.root, "valid")):
            val_root = os.path.join(data_cfg.root, "valid")

        class_names = getattr(data_cfg, "class_names", None)
        
        train_dataset = ClassificationDataset(
            root=train_root,
            transform=train_transform,
            class_names=class_names,
        )
        val_dataset = ClassificationDataset(
            root=val_root,
            transform=val_transform,
            class_names=class_names,
        )
        
        if len(train_dataset) == 0:
            raise RuntimeError(f"Found 0 images in {train_root}. Check if data.root is correct (maybe the dataset downloaded to a different folder like rf_ds_angle-3?) or if the folder structure matches a classification dataset.")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        # COCO segmentation dataset
        train_transform = build_augmentation_pipeline(config, is_train=True)
        val_transform = build_augmentation_pipeline(config, is_train=False)

        dataset_dir = getattr(data_cfg, "root", "front_kr-1")

        train_dataset = COCOSegmentationDataset(
            root=os.path.join(dataset_dir, "train"),
            annotation_file=os.path.join(dataset_dir, "train", "_annotations.coco.json"),
            transform=train_transform,
        )
        val_dataset = COCOSegmentationDataset(
            root=os.path.join(dataset_dir, "valid"),
            annotation_file=os.path.join(dataset_dir, "valid", "_annotations.coco.json"),
            transform=val_transform,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

    return train_loader, val_loader


def build_callbacks(config):
    """Build training callbacks from config."""
    exp_name = getattr(config, "experiment_name", "experiment")
    runs_dir = config.get("output.runs_dir", "runs")
    exp_dir = os.path.join(runs_dir, exp_name)

    callbacks = []

    # Logging callback (always enabled)
    callbacks.append(LoggingCallback())

    # Checkpoint callback
    ckpt_cfg = config.training.checkpoint
    callbacks.append(
        CheckpointCallback(
            checkpoint_dir=os.path.join(exp_dir, "checkpoints"),
            save_every=getattr(ckpt_cfg, "save_every", 5),
            save_best=getattr(ckpt_cfg, "save_best", True),
            metric=getattr(ckpt_cfg, "metric", "loss"),
            mode=getattr(ckpt_cfg, "mode", "min"),
        )
    )

    # Early stopping callback
    es_cfg = config.training.early_stopping
    if getattr(es_cfg, "enabled", True):
        callbacks.append(
            EarlyStoppingCallback(
                patience=getattr(es_cfg, "patience", 10),
                metric=getattr(es_cfg, "metric", "loss"),
                mode=getattr(es_cfg, "mode", "min"),
            )
        )

    # TensorBoard callback
    if config.get("logging.tensorboard", True):
        callbacks.append(
            TensorBoardCallback(
                log_dir=os.path.join(exp_dir, "logs", "tensorboard")
            )
        )

    return callbacks


def main():
    args = parse_args()

    # Load config with overrides
    config = Config.from_file(args.config, overrides=args.set)
    logger = setup_logger("train")

    logger.info(f"Experiment: {getattr(config, 'experiment_name', 'unnamed')}")
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Stage: {config.model.stage}")

    # Set seed
    set_seed(getattr(config, "seed", 42))

    # Get model from registry
    model_name = config.model.name
    logger.info(f"Loading model strategy: {model_name}")
    model_wrapper = get_model(model_name)()
    
    if hasattr(model_wrapper, "train_native"):
        logger.info("Model uses native training loop. Bypassing custom PyTorch Trainer.")
        model_wrapper.build(config.model)
        final_metrics = model_wrapper.train_native(config)
        logger.info(f"Native training finished. Results: {final_metrics}")
        return

    # Build data loaders
    logger.info("Building data loaders...")
    train_loader, val_loader = build_data_loaders(config)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Build callbacks
    callbacks = build_callbacks(config)

    # Build evaluator
    model_type = "classification" if config.model.stage in ["gatekeeper", "angle"] else "detection"
    evaluator = Evaluator(
        model_type=model_type,
        num_classes=getattr(config.model, "num_classes", 2),
    )

    # Create trainer
    trainer = Trainer(
        config=config,
        model_wrapper=model_wrapper,
        train_loader=train_loader,
        val_loader=val_loader,
        callbacks=callbacks,
        evaluator=evaluator,
    )

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train!
    final_metrics = trainer.fit()
    logger.info(f"Final metrics: {final_metrics}")


if __name__ == "__main__":
    main()
