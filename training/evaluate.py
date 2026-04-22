"""LOBO evaluation script for HoneyNIRNet classification and regression.

LOBO (Leave-One-Batch-Out) is used to mimic real deployment conditions:
holding out entire apiary batches avoids leakage from correlated samples that
would occur under random splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error

from app.ml.honey_nir_net import HoneyNIRNet
from training.train import LABEL_TO_INDEX

INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}


def _load_frame(path: str) -> pd.DataFrame:
    """Load evaluation dataframe and verify required columns."""
    df = pd.read_csv(path)
    required = [f"wavelength_{i}" for i in range(1, 257)] + ["label", "adulteration_pct", "apiary_batch_id"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def evaluate_lobo(data_csv: str, checkpoint_path: str, output_path: str) -> Dict[str, Any]:
    """Run LOBO folds and save an evaluation report as JSON."""
    df = _load_frame(data_csv)
    features = [f"wavelength_{i}" for i in range(1, 257)]

    model = HoneyNIRNet()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    class_truth: List[int] = []
    class_pred: List[int] = []
    reg_truth: List[float] = []
    reg_pred: List[float] = []
    fold_scores: Dict[str, float] = {}

    for batch_id in df["apiary_batch_id"].unique():
        test_df = df[df["apiary_batch_id"] == batch_id]
        if test_df.empty:
            continue

        x_test = torch.tensor(test_df[features].to_numpy(dtype=np.float32)).unsqueeze(1)
        y_test_cls = test_df["label"].map(LABEL_TO_INDEX).to_numpy(dtype=np.int64)
        y_test_reg = test_df["adulteration_pct"].to_numpy(dtype=np.float32)

        with torch.no_grad():
            logits, reg = model(x_test)
            preds_cls = torch.argmax(logits, dim=1).cpu().numpy()
            preds_reg = reg.squeeze(1).cpu().numpy()

        class_truth.extend(y_test_cls.tolist())
        class_pred.extend(preds_cls.tolist())
        reg_truth.extend(y_test_reg.tolist())
        reg_pred.extend(preds_reg.tolist())
        fold_scores[str(batch_id)] = float(accuracy_score(y_test_cls, preds_cls))

    class_labels = list(range(len(LABEL_TO_INDEX)))
    cm = confusion_matrix(class_truth, class_pred, labels=class_labels)
    class_accuracy: Dict[str, float] = {}
    for idx in class_labels:
        mask = np.array(class_truth) == idx
        if np.sum(mask) == 0:
            class_accuracy[INDEX_TO_LABEL[idx]] = 0.0
        else:
            class_accuracy[INDEX_TO_LABEL[idx]] = float(np.mean(np.array(class_pred)[mask] == idx))

    report: Dict[str, Any] = {
        "accuracy_per_class": class_accuracy,
        "macro_f1": float(f1_score(class_truth, class_pred, average="macro")),
        "regression_mae": float(mean_absolute_error(reg_truth, reg_pred)),
        "confusion_matrix": cm.tolist(),
        "fold_accuracy": fold_scores,
    }

    out = Path(output_path)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for LOBO evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate HoneyNIRNet using LOBO cross-validation.")
    parser.add_argument("--data-csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="honey_nir_net.pt")
    parser.add_argument("--output", type=str, default="evaluation_report.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = evaluate_lobo(args.data_csv, args.checkpoint, args.output)
    print(json.dumps(result, indent=2))
