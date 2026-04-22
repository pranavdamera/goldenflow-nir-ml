"""Export trained HoneyNIRNet checkpoint to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from app.ml.honey_nir_net import HoneyNIRNet


def export_onnx(checkpoint_path: str, output_path: str) -> None:
    """Export PyTorch model weights to ONNX and enforce size constraint."""
    model = HoneyNIRNet()
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy_input = torch.randn(1, 1, 256, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["spectral_input"],
        output_names=["class_logits", "adulteration_pct"],
        dynamic_axes={
            "spectral_input": {0: "batch_size"},
            "class_logits": {0: "batch_size"},
            "adulteration_pct": {0: "batch_size"},
        },
        opset_version=12,
    )

    output_file = Path(output_path)
    size_bytes = output_file.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    print(f"Exported ONNX model size: {size_mb:.4f} MB")
    if size_bytes >= 2 * 1024 * 1024:
        raise ValueError("ONNX model exceeds 2MB size limit.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for ONNX export."""
    parser = argparse.ArgumentParser(description="Export HoneyNIRNet to ONNX format.")
    parser.add_argument("--checkpoint", type=str, default="honey_nir_net.pt")
    parser.add_argument("--output", type=str, default="honey_nir_net.onnx")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_onnx(args.checkpoint, args.output)
