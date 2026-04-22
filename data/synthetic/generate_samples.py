"""Generate synthetic spectral CSV samples for demo and testing."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from app.utils.synthetic_data import generate_synthetic_spectrum

LABELS: List[str] = ["pure", "rice_syrup", "hfcs", "jaggery_syrup", "invert_sugar", "sugar_fed", "unknown"]


def label_to_adulteration_pct(label: str) -> float:
    """Map synthetic class label to a plausible adulteration percentage."""
    if label == "pure":
        return float(np.random.uniform(0.0, 0.05))
    if label == "unknown":
        return float(np.random.uniform(0.1, 0.6))
    return float(np.random.uniform(0.2, 0.95))


def generate_dataset(num_samples: int, output_csv: str) -> None:
    """Create a synthetic dataset with model-compatible columns."""
    rows = []
    for _ in range(num_samples):
        label = str(np.random.choice(LABELS))
        spectrum = generate_synthetic_spectrum(label=label)
        row = {f"wavelength_{i + 1}": float(v) for i, v in enumerate(spectrum)}
        row["label"] = label
        row["adulteration_pct"] = label_to_adulteration_pct(label)
        row["apiary_batch_id"] = f"batch_{np.random.randint(1, 11)}"
        row["scan_id"] = str(uuid.uuid4())
        rows.append(row)

    frame = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    print(f"Saved {num_samples} samples to {output_csv}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for synthetic dataset generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic NIR spectra.")
    parser.add_argument("--num-samples", type=int, default=1200)
    parser.add_argument("--output", type=str, default="data/synthetic/synthetic_samples.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(args.num_samples, args.output)
