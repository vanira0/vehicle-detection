#!/usr/bin/env python
"""
Instrument Cluster Gatekeeper -- Binary Classification Training Pipeline.

Classifies: valid_cluster (class 0) vs invalid (class 1).

Steps:
  1.  Parse config YAML with optional CLI --set overrides.
  2.  Validate dataset directory and class folder layout.
  3.  Build DataLoaders with ImageNet normalisation + rich augmentation.
  4.  Instantiate ClusterGatekeeperClassifier from the model registry.
  5.  Run AMP-accelerated training: AdamW + CosineAnnealingLR.
  6.  Track per-epoch: loss, accuracy, precision, recall, F1 (macro + per-class).
  7.  Save best_model.pth (on val F1) and last_model.pth each epoch.
  8.  Export confusion matrix PNG after each validation pass.
  9.  Provide standalone predict() usable by downstream pipeline stages.

Usage:
    python scripts/train_cluster_gatekeeper.py

    # Override config values
    python scripts/train_cluster_gatekeeper.py \
        --set training.epochs=30 model.architecture=mobilenet_v3_large

    # Inference only (skip training)
    python scripts/train_cluster_gatekeeper.py --infer-only \
        --weights runs/cluster_gatekeeper_resnet50_v1/weights/best_model.pth \
        --image path/to/image.jpg
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---- project path bootstrap ------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Trigger @register_model decorators for all models
import models.gatekeeper                      # noqa: F401
import models.gatekeeper.cluster_gatekeeper   # noqa: F401  registers cluster_gatekeeper
import models.yolo_segmentation               # noqa: F401
import models.instrument_cluster_detection    # noqa: F401
import models.maskrcnn_segmentation           # noqa: F401
import models.damage                          # noqa: F401
import models.parts                           # noqa: F401
import models.vehicle                         # noqa: F401
import models.angle                           # noqa: F401

from models.registry import get_model
from utils.config import Config
from utils.logger import setup_logger
from utils.seed import set_seed

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLASS_NAMES   = ["valid_cluster", "invalid"]

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Instrument Cluster Gatekeeper Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str,
                   default="configs/gatekeeper/instrument_cluster_gatekeeper.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="Runtime config overrides, e.g. --set training.epochs=30")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint .pth to resume training from.")
    p.add_argument("--infer-only", action="store_true",
                   help="Skip training and run single-image inference.")
    p.add_argument("--weights", type=str, default=None,
                   help="Weights path for --infer-only mode.")
    p.add_argument("--image", type=str, default=None,
                   help="Image path for --infer-only mode.")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _abs(rel: str) -> str:
    """Return absolute path anchored at the project root."""
    return rel if os.path.isabs(rel) else str(_PROJECT_ROOT / rel)


def configure_determinism(seed: int) -> None:
    """Seed all RNGs and enable deterministic CUDA for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def _build_train_transform(cfg: Config) -> transforms.Compose:
    """
    Build the training augmentation pipeline from config.

    Augmentation strategy for the gatekeeper:
      - Geometric: RandomCrop (oversample), HorizontalFlip, RandomRotation
      - Photometric: ColorJitter (brightness/contrast/saturation/hue)
      - Artifacts: GaussianBlur (simulates camera out-of-focus / motion blur)
      - Regularization: RandomErasing

    Args:
        cfg: Full Config object.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    aug  = cfg.augmentation
    size = getattr(cfg.data, "image_size", 224)

    jc = getattr(aug, "color_jitter", None)
    jitter = transforms.ColorJitter(
        brightness=getattr(jc, "brightness", 0.3),
        contrast=getattr(jc,   "contrast",   0.3),
        saturation=getattr(jc, "saturation", 0.2),
        hue=getattr(jc,        "hue",        0.05),
    ) if jc else transforms.ColorJitter(0.3, 0.3, 0.2, 0.05)

    bc = getattr(aug, "gaussian_blur", None)
    blur_p = getattr(bc, "prob", 0.2) if bc else 0.2
    raw_k  = getattr(bc, "kernel_size", [3, 7]) if bc else [3, 7]
    k      = raw_k[0] if isinstance(raw_k, (list, tuple)) else int(raw_k)
    k      = k if k % 2 == 1 else k + 1   # must be odd

    ec = getattr(aug, "random_erasing", None)
    ep = getattr(ec, "prob",  0.1)       if ec else 0.1
    es = getattr(ec, "scale", [0.02, 0.1]) if ec else [0.02, 0.1]

    h_flip  = getattr(aug, "horizontal_flip", 0.5)
    rot_deg = getattr(aug, "rotation_limit",  15)

    return transforms.Compose([
        transforms.Resize((size + 32, size + 32)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(p=h_flip),
        transforms.RandomRotation(degrees=rot_deg),
        transforms.RandomApply([jitter], p=0.8),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=k)], p=blur_p),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=ep, scale=tuple(es), value="random"),
    ])


def _build_val_transform(cfg: Config) -> transforms.Compose:
    """Deterministic validation / inference transform (resize + centre-crop)."""
    size = getattr(cfg.data, "image_size", 224)
    return transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

def validate_dataset(dataset_dir: str, logger: logging.Logger) -> None:
    """
    Verify the ImageFolder layout and log per-class image counts.

    Expected structure:
        <dataset_dir>/train/valid_cluster/  <images>
        <dataset_dir>/train/invalid/        <images>
        <dataset_dir>/val/valid_cluster/    <images>
        <dataset_dir>/val/invalid/          <images>

    Args:
        dataset_dir: Absolute path to the dataset root.
        logger:      Logger instance.

    Raises:
        FileNotFoundError: train/ or val/ split missing.
        RuntimeError:      Fewer than 2 class folders found.
    """
    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for split in ("train", "val"):
        split_path = os.path.join(dataset_dir, split)
        if not os.path.isdir(split_path):
            raise FileNotFoundError(
                f"Dataset split '{split}' not found: {split_path}\n"
                "  Create: train/valid_cluster/ and train/invalid/"
            )
        class_dirs = sorted(d for d in os.scandir(split_path) if d.is_dir())
        if len(class_dirs) < 2:
            raise RuntimeError(
                f"Expected >= 2 class folders in {split_path}, "
                f"found {len(class_dirs)}."
            )
        total = 0
        for cd in class_dirs:
            n = sum(1 for f in os.scandir(cd.path)
                    if f.name.lower().endswith(IMAGE_EXTS))
            logger.info("  [%s] %-18s  %4d images", split, cd.name, n)
            if n == 0:
                logger.warning("  Class folder %r is empty!", cd.name)
            total += n
        logger.info("  [%s] total: %d images", split, total)

# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    cfg: Config,
    logger: logging.Logger,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """
    Build train/val DataLoaders using torchvision.datasets.ImageFolder.

    Args:
        cfg:    Full config object.
        logger: Logger.

    Returns:
        (train_loader, val_loader, class_names) tuple.
    """
    ds_dir      = _abs(getattr(cfg.data, "dataset_dir", "data/instrument_cluster/gatekeeper"))
    batch_size  = getattr(cfg.training, "batch_size", getattr(cfg.data, "batch_size", 32))
    num_workers = getattr(cfg.data, "num_workers", 4)
    pin_memory  = getattr(cfg.data, "pin_memory",  True)

    train_ds = datasets.ImageFolder(
        os.path.join(ds_dir, "train"), transform=_build_train_transform(cfg))
    val_ds = datasets.ImageFolder(
        os.path.join(ds_dir, "val"),   transform=_build_val_transform(cfg))

    class_names = train_ds.classes
    logger.info("Class mapping (ImageFolder): %s", dict(enumerate(class_names)))

    kw = dict(num_workers=num_workers, pin_memory=pin_memory,
              persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw)

    logger.info(
        "DataLoaders | train: %d batches (%d imgs) | val: %d batches (%d imgs)",
        len(train_loader), len(train_ds), len(val_loader), len(val_ds),
    )
    return train_loader, val_loader, class_names

# ---------------------------------------------------------------------------
# Optimizer & Scheduler
# ---------------------------------------------------------------------------

def build_optimizer(model: nn.Module, cfg: Config) -> optim.Optimizer:
    """
    Build AdamW / Adam / SGD from config.

    Only parameters with requires_grad=True are passed to the optimizer,
    which makes backbone freezing / unfreezing transparent.

    Args:
        model: nn.Module (parameters with requires_grad will be optimised).
        cfg:   Full config.

    Returns:
        Configured torch.optim.Optimizer.
    """
    opt  = cfg.optimizer
    lr   = getattr(opt, "lr",           0.0005)
    wd   = getattr(opt, "weight_decay", 0.0001)
    name = getattr(opt, "name",         "AdamW").lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    elif name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    else:
        mom = getattr(opt, "momentum", 0.9)
        return optim.SGD(params, lr=lr, momentum=mom, weight_decay=wd, nesterov=True)


def build_scheduler(
    optimizer: optim.Optimizer,
    cfg: Config,
) -> optim.lr_scheduler._LRScheduler:
    """
    Build CosineAnnealingLR / StepLR / ReduceLROnPlateau from config.

    Args:
        optimizer: Configured optimizer.
        cfg:       Full config.

    Returns:
        LR scheduler.
    """
    sch  = getattr(cfg.optimizer, "scheduler", None)
    if sch is None:
        return optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    name = getattr(sch, "name", "CosineAnnealingLR").lower()
    if "cosine" in name:
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=getattr(sch, "T_max", 60),
            eta_min=getattr(sch, "eta_min", 5e-6),
        )
    elif "plateau" in name:
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            patience=getattr(sch, "patience", 5),
        )
    else:
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=getattr(sch, "step_size", 10),
            gamma=getattr(sch, "gamma", 0.1),
        )

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    labels: List[int],
    preds:  List[int],
    class_names: List[str],
    logger: logging.Logger,
) -> Dict[str, float]:
    """
    Compute classification metrics via scikit-learn.

    Logged metrics:
      - accuracy
      - precision_macro, recall_macro, f1_macro
      - f1_<class_name> per class

    Args:
        labels:      Ground-truth class indices.
        preds:       Predicted class indices.
        class_names: Human-readable class names.
        logger:      Logger.

    Returns:
        Flat dict of metric name -> float value.
    """
    y, yhat = np.array(labels), np.array(preds)
    acc  = float((y == yhat).mean())
    prec = float(precision_score(y, yhat, average="macro", zero_division=0))
    rec  = float(recall_score(y,    yhat, average="macro", zero_division=0))
    f1m  = float(f1_score(y,        yhat, average="macro", zero_division=0))
    pf1  = f1_score(y, yhat, average=None, zero_division=0)

    logger.info("\n%s",
        classification_report(y, yhat, target_names=class_names, zero_division=0))

    return {
        "accuracy":        acc,
        "precision_macro": prec,
        "recall_macro":    rec,
        "f1_macro":        f1m,
        **{f"f1_{cn}": float(v) for cn, v in zip(class_names, pf1)},
    }


def save_confusion_matrix(
    labels:      List[int],
    preds:       List[int],
    class_names: List[str],
    path:        str,
    epoch:       int,
) -> None:
    """
    Save a confusion matrix PNG.  Silent fail if matplotlib not installed.

    Args:
        labels:      Ground-truth labels.
        preds:       Predicted labels.
        class_names: Class names for axis labels.
        path:        Absolute save path.
        epoch:       Current epoch number for title.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im  = ax.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ticks = range(len(class_names))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (epoch {epoch})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model:     nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    scaler:    GradScaler,
    epoch:     int,
    metrics:   Dict[str, float],
    path:      str,
) -> None:
    """Persist a full training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state":    scaler.state_dict(),
        "metrics":         metrics,
    }, path)


def load_checkpoint(
    path:      str,
    model:     nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    scaler:    GradScaler,
    device:    torch.device,
    logger:    logging.Logger,
) -> int:
    """Load a training checkpoint and return the start epoch."""
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model_state"])
    optimizer.load_state_dict(ck["optimizer_state"])
    scheduler.load_state_dict(ck["scheduler_state"])
    scaler.load_state_dict(ck["scaler_state"])
    start = ck["epoch"] + 1
    logger.info("Resumed from checkpoint: %s (epoch %d)", path, start)
    return start

# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    optimizer:   optim.Optimizer,
    criterion:   nn.Module,
    scaler:      GradScaler,
    device:      torch.device,
    amp_enabled: bool,
    epoch:       int,
    num_epochs:  int,
    logger:      logging.Logger,
) -> Dict[str, float]:
    """
    Run one AMP-accelerated training epoch with gradient clipping.

    Args:
        model, loader, optimizer, criterion, scaler: training components.
        device:      Compute device.
        amp_enabled: Whether to use torch.cuda.amp.autocast.
        epoch:       Current epoch (0-indexed).
        num_epochs:  Total epochs.
        logger:      Logger.

    Returns:
        Dict with 'loss' and 'accuracy'.
    """
    model.train()
    running_loss = correct = total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds   = logits.detach().argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

        log_every = max(1, len(loader) // 5)
        if (batch_idx + 1) % log_every == 0:
            logger.debug(
                "  [Train] E%d/%d step %d/%d  loss=%.4f acc=%.4f",
                epoch + 1, num_epochs, batch_idx + 1, len(loader),
                running_loss / total, correct / total,
            )

    return {
        "loss":     running_loss / max(total, 1),
        "accuracy": correct      / max(total, 1),
    }


@torch.no_grad()
def validate(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   nn.Module,
    device:      torch.device,
    amp_enabled: bool,
    class_names: List[str],
    logger:      logging.Logger,
) -> Tuple[Dict[str, float], List[int], List[int]]:
    """
    Full validation pass: loss, accuracy, precision, recall, F1.

    Args:
        model, loader, criterion: evaluation components.
        device:      Compute device.
        amp_enabled: Whether to use autocast.
        class_names: Class names for metrics / reports.
        logger:      Logger.

    Returns:
        Tuple of (metrics_dict, all_labels, all_preds).
    """
    model.eval()
    running_loss = 0.0
    all_labels: List[int] = []
    all_preds:  List[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)
            loss   = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())

    val_loss = running_loss / max(len(all_labels), 1)
    metrics  = compute_metrics(all_labels, all_preds, class_names, logger)
    metrics["loss"] = val_loss
    return metrics, all_labels, all_preds

# ---------------------------------------------------------------------------
# Standalone predict() -- importable by downstream pipeline stages
# ---------------------------------------------------------------------------

def predict(
    image_path:  str,
    model:       nn.Module,
    transform:   Any,
    class_names: List[str],
    device:      torch.device,
) -> Dict[str, Any]:
    """
    Single-image inference.  Designed to be imported by other pipeline stages.

    Args:
        image_path:  Path to the image file.
        model:       Trained nn.Module in eval mode.
        transform:   Validation-style torchvision transform.
        class_names: Ordered class name list (index 0 = valid_cluster).
        device:      Torch device.

    Returns:
        Dict with keys:
            label        -- predicted class name (str)
            class_index  -- predicted class index (int)
            confidence   -- probability of predicted class (float 0-1)
            class_probs  -- {class_name: probability} for all classes
            is_valid     -- True if predicted class is 'valid_cluster'
    """
    model.eval()
    img    = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    idx   = int(np.argmax(probs))
    label = class_names[idx] if idx < len(class_names) else str(idx)

    return {
        "label":       label,
        "class_index": idx,
        "confidence":  float(probs[idx]),
        "class_probs": {cn: float(p) for cn, p in zip(class_names, probs)},
        "is_valid":    (label == "valid_cluster"),
    }

# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate the full gatekeeper training pipeline."""
    args   = parse_args()
    logger = setup_logger("cluster_gatekeeper_train")

    # ---- Config ----------------------------------------------------------
    config_path = _abs(args.config)
    logger.info("Loading config: %s", config_path)
    cfg = Config.from_file(config_path, overrides=args.set)

    seed = getattr(cfg, "seed", 42)
    configure_determinism(seed)
    set_seed(seed)
    logger.info("Seed: %d", seed)

    # ---- Device ----------------------------------------------------------
    dev_str = getattr(getattr(cfg, "project", cfg), "device", None)
    device  = (torch.device(dev_str)
               if (dev_str and dev_str not in ("null", "None", ""))
               else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    amp = getattr(cfg.training, "use_amp", True) and device.type == "cuda"
    logger.info("Device: %s | AMP: %s", device, amp)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # ---- Output directories ----------------------------------------------
    log_cfg     = cfg.logging
    weights_dir = _abs(getattr(log_cfg, "weights_dir", "runs/cluster_gatekeeper/weights"))
    plots_dir   = _abs(getattr(log_cfg, "plots_dir",   "runs/cluster_gatekeeper/plots"))
    log_dir     = _abs(getattr(log_cfg, "log_dir",     "runs/cluster_gatekeeper/logs"))
    best_path   = os.path.join(weights_dir, getattr(log_cfg, "best_model_filename", "best_model.pth"))
    last_path   = os.path.join(weights_dir, getattr(log_cfg, "last_model_filename", "last_model.pth"))
    for d in (weights_dir, plots_dir, log_dir):
        os.makedirs(d, exist_ok=True)
    # Re-configure logger with file handler
    setup_logger("cluster_gatekeeper_train", log_dir=log_dir)

    # ---- Infer-only ------------------------------------------------------
    if args.infer_only:
        if not (args.weights and args.image):
            logger.error("--infer-only requires both --weights and --image.")
            sys.exit(1)
        logger.info("=== INFERENCE ONLY ===")
        mw      = get_model("cluster_gatekeeper")()
        model   = mw.build(cfg.model).to(device)
        ck      = torch.load(args.weights, map_location=device)
        model.load_state_dict(ck.get("model_state", ck))
        classes = getattr(cfg.data, "class_names", CLASS_NAMES)
        result  = predict(args.image, model, _build_val_transform(cfg), classes, device)
        logger.info("Prediction: %s", result)
        print(result)
        return

    # ---- Dataset ---------------------------------------------------------
    ds_dir = _abs(getattr(cfg.data, "dataset_dir", "data/instrument_cluster/gatekeeper"))
    try:
        validate_dataset(ds_dir, logger)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.warning("Dataset check warning: %s", exc)

    # ---- DataLoaders -----------------------------------------------------
    train_loader, val_loader, class_names = build_dataloaders(cfg, logger)

    # ---- Model -----------------------------------------------------------
    logger.info("Building model: cluster_gatekeeper (%s)", cfg.model.architecture)
    mw    = get_model("cluster_gatekeeper")()
    model = mw.build(cfg.model).to(device)
    n_tot = sum(p.numel() for p in model.parameters())
    n_tr  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: total=%s  trainable=%s", f"{n_tot:,}", f"{n_tr:,}")

    # ---- Optimizer / Scheduler / AMP / Loss ------------------------------
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler    = GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # ---- Resume ----------------------------------------------------------
    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        start_epoch = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device, logger)

    # ---- Training hyper-params -------------------------------------------
    num_epochs    = getattr(cfg.training, "epochs",                  60)
    patience      = getattr(cfg.training, "early_stopping_patience", 15)
    freeze_epochs = getattr(cfg.training, "freeze_backbone_epochs",   5)

    best_val_f1, no_improve = -1.0, 0
    history: List[Dict[str, Any]] = []

    exp_name = getattr(cfg, "experiment_name", "cluster_gatekeeper")
    logger.info("=" * 65)
    logger.info("START: %s | epochs=%d | patience=%d | freeze=%d epochs",
                exp_name, num_epochs, patience, freeze_epochs)
    logger.info("=" * 65)
    t0 = time.perf_counter()

    # ---- Epoch loop ------------------------------------------------------
    for epoch in range(start_epoch, num_epochs):

        # Backbone freeze schedule
        if freeze_epochs:
            if epoch == 0:
                logger.info("Epoch 0: backbone frozen (head-only warm-up).")
            elif epoch == freeze_epochs:
                logger.info("Epoch %d: unfreezing backbone for full fine-tuning.", epoch)
                mw.unfreeze_backbone(model)
                # Rebuild optimizer so newly unfrozen params get LR state
                optimizer = build_optimizer(model, cfg)
                scheduler = build_scheduler(optimizer, cfg)

        # Train
        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, amp, epoch, num_epochs, logger,
        )

        # Validate
        val_m, v_labels, v_preds = validate(
            model, val_loader, criterion, device, amp, class_names, logger,
        )

        # LR step
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_m["f1_macro"])
        else:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        # History
        row = {
            "epoch": epoch + 1, "lr": lr,
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}":   v for k, v in val_m.items()},
        }
        history.append(row)

        logger.info(
            "E%3d/%d | lr=%.2e | "
            "tr_loss=%.4f tr_acc=%.4f | "
            "val_loss=%.4f val_acc=%.4f val_f1=%.4f",
            epoch + 1, num_epochs, lr,
            train_m["loss"], train_m["accuracy"],
            val_m["loss"],   val_m["accuracy"],   val_m["f1_macro"],
        )

        # Confusion matrix
        cm_path = os.path.join(plots_dir, f"cm_epoch_{epoch+1:03d}.png")
        save_confusion_matrix(v_labels, v_preds, class_names, cm_path, epoch + 1)

        # Best model checkpoint
        if val_m["f1_macro"] > best_val_f1:
            best_val_f1, no_improve = val_m["f1_macro"], 0
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_m, best_path)
            logger.info("  >> Best model saved | val_f1=%.4f -> %s", best_val_f1, best_path)
        else:
            no_improve += 1
            logger.info(
                "  No improvement (%d/%d) | best_f1=%.4f",
                no_improve, patience, best_val_f1,
            )

        # Last checkpoint (always saved for resume)
        save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_m, last_path)

        # Early stopping
        if no_improve >= patience:
            logger.info("Early stopping triggered at epoch %d.", epoch + 1)
            break

    # ---- Final summary ---------------------------------------------------
    elapsed = time.perf_counter() - t0
    logger.info("=" * 65)
    logger.info("DONE | %.1fs (%.1f min) | best_f1=%.4f", elapsed, elapsed / 60, best_val_f1)
    logger.info("Best weights: %s", best_path)
    logger.info("=" * 65)

    # Save CSV history
    if history:
        csv_path = os.path.join(log_dir, "metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        logger.info("Metrics CSV saved: %s", csv_path)


if __name__ == "__main__":
    main()
