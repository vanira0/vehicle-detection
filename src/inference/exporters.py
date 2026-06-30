"""
Model exporters for deployment.

Export trained PyTorch models to:
    - ONNX format (for ONNX Runtime / OpenVINO inference)
    - TorchScript format (for C++ / mobile deployment)
"""

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

from utils.logger import setup_logger


logger = setup_logger("exporter")


class ONNXExporter:
    """
    Export a PyTorch model to ONNX format.

    Usage:
        exporter = ONNXExporter()
        exporter.export(model, output_path="model.onnx", input_size=(1, 3, 1024, 1024))
    """

    @staticmethod
    def export(
        model: nn.Module,
        output_path: str,
        input_size: Tuple[int, ...] = (1, 3, 1024, 1024),
        opset_version: int = 13,
        dynamic_axes: Optional[dict] = None,
        simplify: bool = True,
    ) -> str:
        """
        Export model to ONNX format.

        Args:
            model: PyTorch model in eval mode.
            output_path: Path to save the .onnx file.
            input_size: Input tensor shape (batch, channels, height, width).
            opset_version: ONNX opset version.
            dynamic_axes: Dynamic axes for variable input sizes.
            simplify: Whether to simplify the ONNX graph.

        Returns:
            Path to the exported ONNX file.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        model.eval()
        device = next(model.parameters()).device

        dummy_input = torch.randn(*input_size).to(device)

        if dynamic_axes is None:
            dynamic_axes = {
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "output": {0: "batch_size"},
            }

        logger.info(f"Exporting to ONNX: {output_path}")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

        # Optionally simplify
        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify

                onnx_model = onnx.load(output_path)
                simplified, check = onnx_simplify(onnx_model)
                if check:
                    onnx.save(simplified, output_path)
                    logger.info("ONNX model simplified successfully")
                else:
                    logger.warning("ONNX simplification check failed, keeping original")
            except ImportError:
                logger.info("onnx-simplifier not installed, skipping simplification")

        # Verify
        try:
            import onnx

            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            logger.info(f"ONNX model verified: {output_path}")
        except ImportError:
            logger.info("onnx not installed, skipping verification")

        return output_path


class TorchScriptExporter:
    """
    Export a PyTorch model to TorchScript format.

    Usage:
        exporter = TorchScriptExporter()
        exporter.export(model, output_path="model.pt", method="trace")
    """

    @staticmethod
    def export(
        model: nn.Module,
        output_path: str,
        method: str = "trace",
        input_size: Tuple[int, ...] = (1, 3, 1024, 1024),
    ) -> str:
        """
        Export model to TorchScript format.

        Args:
            model: PyTorch model in eval mode.
            output_path: Path to save the .pt file.
            method: "trace" or "script".
            input_size: Input tensor shape for tracing.

        Returns:
            Path to the exported TorchScript file.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        model.eval()
        device = next(model.parameters()).device

        logger.info(f"Exporting to TorchScript ({method}): {output_path}")

        if method == "trace":
            dummy_input = torch.randn(*input_size).to(device)
            scripted = torch.jit.trace(model, dummy_input)
        elif method == "script":
            scripted = torch.jit.script(model)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'.")

        scripted.save(output_path)
        logger.info(f"TorchScript model saved: {output_path}")

        return output_path
