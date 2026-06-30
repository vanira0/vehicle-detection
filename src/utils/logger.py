"""
Structured logging setup for training, evaluation, and inference.

Provides consistent formatting across all pipeline stages with
support for file and console output.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "vehicle_detection",
    log_dir: Optional[str] = None,
    level: str = "INFO",
    log_filename: Optional[str] = None,
) -> logging.Logger:
    """
    Configure and return a logger with console and optional file output.

    Args:
        name: Logger name (used to retrieve the same logger across modules).
        log_dir: Directory for log files. If None, only console output.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_filename: Custom log filename. Defaults to timestamped name.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File handler (if log_dir specified)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"train_{timestamp}.log"
        file_path = os.path.join(log_dir, log_filename)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    return logger


class MetricsLogger:
    """
    CSV-based metrics logger for tracking training/evaluation metrics
    across epochs and experiments.
    """

    def __init__(self, log_dir: str, filename: str = "metrics.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, filename)
        self._header_written = False

    def log(self, metrics: dict) -> None:
        """
        Append a row of metrics to the CSV file.
        The header is written automatically on the first call.
        """
        if not self._header_written:
            with open(self.filepath, "w") as f:
                f.write(",".join(metrics.keys()) + "\n")
            self._header_written = True

        with open(self.filepath, "a") as f:
            f.write(",".join(str(v) for v in metrics.values()) + "\n")

    def read(self):
        """Read the CSV file and return as a list of dicts."""
        import csv

        if not os.path.exists(self.filepath):
            return []
        with open(self.filepath, "r") as f:
            reader = csv.DictReader(f)
            return list(reader)
