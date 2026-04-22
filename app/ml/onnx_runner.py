"""ONNX Runtime wrapper for HoneyNIRNet inference."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort

CLASS_LABELS = [
    "pure",
    "rice_syrup",
    "hfcs",
    "jaggery_syrup",
    "invert_sugar",
    "sugar_fed",
    "unknown",
]


class ONNXRunner:
    """Load and execute ONNX inference for single and batch requests."""

    def __init__(self, model_path: str) -> None:
        """Initialize an ONNX runtime session.

        Note:
            `honey_nir_net.onnx` is intentionally not committed to the repo.
            Generate it with `python training/export_onnx.py`.
        """
        self.model_path = model_path
        self.session: ort.InferenceSession | None = None
        path = Path(model_path)
        if path.exists():
            self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
        else:
            self.input_name = "spectral_input"
            warnings.warn(
                f"ONNX model not found at {model_path}. Falling back to deterministic dummy inference.",
                RuntimeWarning,
            )

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Compute numerically stable softmax probabilities."""
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    @staticmethod
    def _format_result(probabilities: np.ndarray, reg_value: float, inference_ms: float) -> Dict[str, object]:
        """Convert model outputs into structured response payload."""
        class_idx = int(np.argmax(probabilities))
        class_label = CLASS_LABELS[class_idx]
        return {
            "class_label": class_label,
            "class_probabilities": {label: float(probabilities[i]) for i, label in enumerate(CLASS_LABELS)},
            "adulteration_pct": float(np.clip(reg_value, 0.0, 1.0)),
            "inference_ms": float(inference_ms),
        }

    def _run_session(self, batch_input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run either real ONNX session or deterministic fallback outputs."""
        if self.session is None:
            pooled = np.mean(batch_input, axis=2).reshape(-1, 1)
            logits = np.concatenate([pooled + i * 0.05 for i in range(7)], axis=1).astype(np.float32)
            reg = 1.0 / (1.0 + np.exp(-pooled))
            return logits, reg.astype(np.float32)

        outputs = self.session.run(None, {self.input_name: batch_input.astype(np.float32)})
        if len(outputs) != 2:
            raise ValueError("ONNX model must output classification logits and regression value.")
        class_logits = np.asarray(outputs[0], dtype=np.float32)
        reg_values = np.asarray(outputs[1], dtype=np.float32)
        return class_logits, reg_values

    def run_single(self, processed_array: np.ndarray) -> Dict[str, object]:
        """Infer one preprocessed spectrum and return prediction metadata."""
        batched = np.asarray(processed_array, dtype=np.float32).reshape(1, 1, 256)
        start = time.perf_counter()
        class_logits, reg_values = self._run_session(batched)
        inference_ms = (time.perf_counter() - start) * 1000.0
        if inference_ms > 300.0:
            warnings.warn(f"Inference time {inference_ms:.2f}ms exceeds 300ms target.", RuntimeWarning)

        probs = self._softmax(class_logits)[0]
        reg = float(np.asarray(reg_values).reshape(-1)[0])
        return self._format_result(probs, reg, inference_ms)

    def run_batch(self, list_of_arrays: List[np.ndarray]) -> List[Dict[str, object]]:
        """Infer a batch of preprocessed spectra and return per-sample outputs."""
        if not list_of_arrays:
            return []
        stacked = np.asarray(list_of_arrays, dtype=np.float32).reshape(len(list_of_arrays), 1, 256)
        start = time.perf_counter()
        class_logits, reg_values = self._run_session(stacked)
        total_ms = (time.perf_counter() - start) * 1000.0
        per_sample_ms = total_ms / len(list_of_arrays)
        if per_sample_ms > 300.0:
            warnings.warn(
                f"Average per-sample inference time {per_sample_ms:.2f}ms exceeds 300ms target.",
                RuntimeWarning,
            )

        probs = self._softmax(class_logits)
        reg_values = np.asarray(reg_values).reshape(-1)
        return [self._format_result(probs[i], float(reg_values[i]), per_sample_ms) for i in range(len(list_of_arrays))]
