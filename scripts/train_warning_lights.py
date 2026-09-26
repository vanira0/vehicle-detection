#!/usr/bin/env python
"""
Training pipeline for Instrument Cluster Warning Light Detection.

Dedicated entry-point that integrates into the existing vehicle-detection project
by reusing the shared Config, logger, and model registry infrastructure.

Steps performed
---------------
1. Load configs/instrument_cluster/yolo11_warning_lights.yaml (or --config path).
2. Validate environment: CUDA availability, data YAML / image directories.
3. Load InstrumentClusterDetector from the model registry.
4. Call model_wrapper.train_native(config) -- Ultralytics handles the full loop.
5. Run model_wrapper.validate() and log mAP50, mAP50-95, precision, recall.
6. Export colour-annotated visual predictions from the validation split.

Usage
-----
    # Basic run
    python scripts/train_warning_lights.py

    # Custom config / overrides
    python scripts/train_warning_lights.py \
        --config configs/instrument_cluster/yolo11_warning_lights.yaml \
        --set training.epochs=50 model.weights=yolov8s.pt

    # Skip visual export
    python scripts/train_warning_lights.py --no-visual-export
"""
#!/usr/bin/env python
import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import torch
import yaml

import models.instrument_cluster_detection  # noqa: F401
import models.yolo_segmentation             # noqa: F401
import models.gatekeeper                    # noqa: F401
import models.angle                         # noqa: F401
import models.damage                        # noqa: F401
import models.parts                         # noqa: F401
import models.vehicle                       # noqa: F401
import models.maskrcnn_segmentation         # noqa: F401

from models.registry import get_model
from utils.config import Config
from utils.logger import setup_logger
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO warning-light detector on instrument cluster images.")
    parser.add_argument("--config", type=str, default="configs/instrument_cluster/yolo11_warning_lights.yaml", help="Path to the YAML experiment configuration file.")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override config key-value pairs at runtime. Example: --set training.epochs=50 model.weights=yolov8s.pt")
    parser.add_argument("--no-visual-export", action="store_true", help="Skip exporting annotated visual prediction samples after training.")
    parser.add_argument("--visual-export-dir", type=str, default=None, help="Directory where annotated preview images are saved.")
    parser.add_argument("--visual-source-dir", type=str, default=None, help="Directory of source images to run inference on for visual export.")
    parser.add_argument("--max-visual-images", type=int, default=30, help="Maximum number of images to export as visual predictions.")
    return parser.parse_args()


def _abs_path(rel: str) -> str:
    return rel if os.path.isabs(rel) else str(_PROJECT_ROOT / rel)


def validate_environment(config: Config, logger: logging.Logger) -> None:
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        logger.info("CUDA available -- %d GPU(s): %s", gpu_count, gpu_names)
    else:
        logger.warning("CUDA not available. Training will run on CPU which may be very slow.")

    data_yaml_rel = getattr(config.data, "yaml_path", "")
    data_yaml = _abs_path(data_yaml_rel)
    if not os.path.isfile(data_yaml):
        raise FileNotFoundError(
            f"data.yaml not found at '{data_yaml}'.\n"
            "  Please create the file or update 'data.yaml_path' in your config.\n"
            "  A template is provided at: data/instrument_cluster/data.yaml"
        )
    logger.info("Data YAML found: %s", data_yaml)

    with open(data_yaml, "r", encoding="utf-8") as fh:
        data_meta = yaml.safe_load(fh)

    dataset_root = data_meta.get("path", "")
    if not os.path.isabs(dataset_root):
        dataset_root = str(_PROJECT_ROOT / dataset_root)

    for split in ("train", "val"):
        split_rel  = data_meta.get(split, f"images/{split}")
        split_path = os.path.join(dataset_root, split_rel)
        if not os.path.isdir(split_path):
            logger.warning("Image directory for split '%s' not found: %s -- Training will fail unless images are placed there.", split, split_path)
        else:
            n_imgs = sum(1 for f in os.listdir(split_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")))
            logger.info("Split '%-5s': %4d images  [%s]", split, n_imgs, split_path)

    nc_yaml   = data_meta.get("nc", 0)
    nc_config = len(getattr(config.data, "class_names", []))
    if nc_config and nc_yaml != nc_config:
        logger.warning("Class count mismatch: data.yaml says nc=%d, but config.data.class_names has %d entries.", nc_yaml, nc_config)


def run_validation(model_wrapper: Any, config: Config, logger: logging.Logger) -> Dict[str, float]:
    logger.info("=" * 60)
    logger.info("POST-TRAINING VALIDATION")
    logger.info("=" * 60)
    data_yaml = _abs_path(getattr(config.data, "yaml_path", ""))
    img_size  = getattr(config.data, "image_size", 640)
    try:
        metrics = model_wrapper.validate(data_yaml=data_yaml, img_size=img_size)
        logger.info("  %-22s  %.4f", "mAP@0.5",      metrics["mAP50"])
        logger.info("  %-22s  %.4f", "mAP@0.5:0.95", metrics["mAP50_95"])
        logger.info("  %-22s  %.4f", "Precision",     metrics["precision"])
        logger.info("  %-22s  %.4f", "Recall",        metrics["recall"])
        return metrics
    except Exception as exc:
        logger.error("Validation failed: %s", exc, exc_info=True)
        return {}


def export_previews(model_wrapper: Any, config: Config, args: argparse.Namespace, logger: logging.Logger) -> None:
    logger.info("=" * 60)
    logger.info("EXPORTING VISUAL PREDICTIONS")
    logger.info("=" * 60)
    if args.visual_source_dir:
        source_dir = args.visual_source_dir
    else:
        data_yaml = _abs_path(getattr(config.data, "yaml_path", ""))
        with open(data_yaml, "r", encoding="utf-8") as fh:
            data_meta = yaml.safe_load(fh)
        dataset_root = data_meta.get("path", "")
        if not os.path.isabs(dataset_root):
            dataset_root = str(_PROJECT_ROOT / dataset_root)
        val_rel    = data_meta.get("val", "images/val")
        source_dir = os.path.join(dataset_root, val_rel)

    if not os.path.isdir(source_dir):
        logger.warning("Visual export skipped: source directory not found: %s", source_dir)
        return

    if args.visual_export_dir:
        output_dir = args.visual_export_dir
    else:
        runs_dir   = _abs_path(getattr(config.output, "runs_dir", "runs"))
        output_dir = os.path.join(runs_dir, "visual_predictions")

    saved = model_wrapper.export_visual_predictions(image_dir=source_dir, output_dir=output_dir, max_images=args.max_visual_images)
    if saved:
        logger.info("Saved %d annotated images to: %s", len(saved), output_dir)
    else:
        logger.warning("No images were exported.")


def main() -> None:
    args   = parse_args()
    logger = setup_logger("train_warning_lights")

    config_path = _abs_path(args.config)
    logger.info("Loading config: %s", config_path)
    config = Config.from_file(config_path, overrides=args.set)

    exp_name = getattr(config, "experiment_name", "warning_lights")
    logger.info("=" * 60)
    logger.info("Experiment  : %s", exp_name)
    logger.info("Model name  : %s", config.model.name)
    logger.info("Stage       : %s", config.model.stage)
    logger.info("Weights     : %s", config.model.weights)
    logger.info("Epochs      : %s", config.training.epochs)
    logger.info("Batch size  : %s", config.training.batch_size)
    logger.info("Image size  : %s", config.data.image_size)
    logger.info("=" * 60)

    seed = getattr(config, "seed", 42)
    set_seed(seed)
    logger.info("Random seed set to %d", seed)

    logger.info("Running environment checks...")
    validate_environment(config, logger)

    model_name    = config.model.name
    logger.info("Fetching model from registry: '%s'", model_name)
    model_wrapper = get_model(model_name)()
    model_wrapper.build(config.model)

    if not hasattr(model_wrapper, "train_native"):
        raise RuntimeError(f"Model '{model_name}' does not implement train_native(). Only Ultralytics-native models are supported by this script.")

    logger.info("Launching native Ultralytics training...")
    t0 = time.perf_counter()

    try:
        train_results = model_wrapper.train_native(config)
        elapsed = time.perf_counter() - t0
        logger.info("Training complete in %.1f s (%.1f min).", elapsed, elapsed / 60)
        logger.info("Raw Ultralytics results: %s", train_results)
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        sys.exit(1)

    metrics = run_validation(model_wrapper, config, logger)

    if not args.no_visual_export:
        try:
            export_previews(model_wrapper, config, args, logger)
        except Exception as exc:
            logger.warning("Visual export failed (non-fatal): %s", exc, exc_info=True)
    else:
        logger.info("Visual export skipped (--no-visual-export).")

    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY -- %s", exp_name)
    logger.info("=" * 60)
    if metrics:
        logger.info("  mAP@0.5       : %.4f", metrics.get("mAP50",    0.0))
        logger.info("  mAP@0.5:0.95  : %.4f", metrics.get("mAP50_95", 0.0))
        logger.info("  Precision     : %.4f", metrics.get("precision", 0.0))
        logger.info("  Recall        : %.4f", metrics.get("recall",    0.0))
    runs_dir = _abs_path(getattr(config.output, "runs_dir", "runs"))
    logger.info("  Run directory : %s", runs_dir)
    logger.info("=" * 60)
    logger.info("Done.")


if __name__ == "__main__":
    main()
