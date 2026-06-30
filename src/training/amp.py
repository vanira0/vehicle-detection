"""
Automatic Mixed Precision (AMP) utilities.

Provides a context manager and scaler wrapper for cleaner AMP usage
across the training pipeline.
"""

import torch


class AMPContext:
    """
    Context manager for Automatic Mixed Precision.

    Wraps torch.amp.autocast and GradScaler to simplify AMP usage.
    When disabled, acts as a no-op pass-through.

    Usage:
        amp_ctx = AMPContext(enabled=True, device_type="cuda")

        with amp_ctx.autocast():
            loss = model(inputs)

        amp_ctx.scale_and_step(loss, optimizer)
    """

    def __init__(self, enabled: bool = True, device_type: str = "cuda"):
        self.enabled = enabled and device_type == "cuda"
        self.device_type = device_type
        self.scaler = torch.amp.GradScaler(device_type, enabled=self.enabled)

    def autocast(self):
        """Return an autocast context manager."""
        return torch.amp.autocast(self.device_type, enabled=self.enabled)

    def scale_and_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        clip_grad_norm: float = None,
        model_parameters=None,
    ) -> None:
        """
        Scale loss, optionally clip gradients, and step the optimizer.

        Args:
            loss: The loss tensor to backpropagate.
            optimizer: The optimizer to step.
            clip_grad_norm: Optional max gradient norm for clipping.
            model_parameters: Model parameters for gradient clipping.
        """
        self.scaler.scale(loss).backward()

        if clip_grad_norm and model_parameters is not None:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_parameters, clip_grad_norm)

        self.scaler.step(optimizer)
        self.scaler.update()

    def state_dict(self) -> dict:
        """Return scaler state for checkpointing."""
        return self.scaler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """Load scaler state from checkpoint."""
        self.scaler.load_state_dict(state_dict)
