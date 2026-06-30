"""
Reproducibility utilities.

Sets random seeds for Python, NumPy, PyTorch (CPU + CUDA),
and configures CuDNN for deterministic behavior.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
        deterministic: If True, enables CuDNN deterministic mode
                       (may slightly reduce performance).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Allow CuDNN to find optimal algorithms (faster but non-deterministic)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
