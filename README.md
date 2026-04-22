# GoldenFlow Labs — NIR Spectral Classification Engine

Open-source ML inference stack for portable honey adulteration detection via near-infrared spectroscopy.

## Overview

Honey adulteration is a global food integrity issue that contributes to major economic loss, including an estimated $750M+ annual impact in the US market. GoldenFlow Labs builds portable NIR + AI tooling to detect adulteration risk at point-of-test.

Near-infrared (NIR) spectroscopy works by exciting molecular bond vibrations — primarily C–H and O–H stretches that are characteristic of sugar composition — in the 900–1700nm window. Adulterant syrups (rice syrup, HFCS, invert sugar, jaggery) each produce distinguishable absorption profiles because their sugar ratios and water content differ from authentic honey. This repository provides the open-source inference and MLOps framework used to process those spectra, run classification/regression inference, and produce tamper-evident audit records.

This public repository includes:
- FastAPI inference API
- Preprocessing + ONNX runtime inference pipeline
- Training, evaluation, and ONNX export scripts
- Synthetic data generators for safe demonstration

This repository does not include:
- HoneyPrint proprietary spectral database
- Proprietary trained production weights

## Architecture

```text
Raw Spectral Input (256-point array, 900–1700nm)
        |
        v
SpectralPreprocessor
  ├── validate_range()       — reject out-of-range or malformed arrays
  ├── smooth()               — Savitzky-Golay (window=11, poly=2) noise removal
  └── normalize()            — SNV: center + scale by per-spectrum std
        |
        v
HoneyNIRNet (1D-CNN + SE Attention)
  ├── ConvSEBlock ×3         — 1→32→64→128 channels, kernel=5, SE reduction=16
  ├── AdaptiveAvgPool1d      — collapse spectral dimension to scalar per channel
  ├── classifier head        — Linear(128, 7) → 7-class logits
  └── regressor head         — Linear(128, 1) → Sigmoid → adulteration % ∈ [0, 1]
        |
        v
ONNX Runtime (honey_nir_net.onnx, target ≤2MB)
        |
        v
{class_label, adulteration_pct}
        |
        v
AuditChain
  └── SHA-256 hash of (input + output + timestamp) → tamper-evident record
```

## Model

`HoneyNIRNet` is a 1D-CNN with Squeeze-Excitation channel attention trained to classify honey adulteration profiles and estimate adulteration percentage.

SE attention was chosen over multi-head self-attention because NIR spectra are smooth, band-correlated signals: the important structure is which frequency bands co-activate, not positional order along the axis. Transformers would add positional relationships that don't exist here and would push the ONNX export beyond the 2MB deployment target.

Output classes:
- `pure`: likely unadulterated honey profile
- `rice_syrup`: rice syrup adulteration signature
- `hfcs`: high-fructose corn syrup signature
- `jaggery_syrup`: jaggery syrup signature
- `invert_sugar`: invert sugar signature
- `sugar_fed`: likely sugar-fed source profile
- `unknown`: non-conforming or ambiguous profile

Training pipeline:
- **Focal Loss** down-weights easy negatives (abundant `pure` samples) so the model learns the minority adulteration classes rather than collapsing to predict `pure`
- **LOBO (Leave-One-Batch-Out)** cross-validation holds out entire acquisition batches, not random samples — this tests generalization to new sensor sessions and operator conditions, which random splits would leak through shared measurement variance
- AdamW optimizer with cosine LR scheduling

Inference targets:
- ONNX export size under 2MB
- Sub-300ms end-to-end latency per sample

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/predict` | Single-sample inference with preprocessing and audit record creation |
| POST | `/batch` | Batch inference for up to 2000 samples with summary metrics and batch audit ID |

## Performance

- Sub-300ms end-to-end latency target for single-sample inference
- Consistent behavior validated across 20+ independent verification runs
- ONNX model export size constrained below 2MB

## Data & Proprietary IP

The HoneyPrint database is proprietary and not distributed in this repository. See `data/README.md` for details and synthetic data usage.

## Quickstart

### Local

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t goldenflow-nir-ml .
docker run --rm -p 8000:8000 goldenflow-nir-ml
```

### Run Tests

```bash
pytest tests/
```

## Patent

Core device architecture and HoneyPrint database methodology protected under provisional patent filing, September 2025.

## License

MIT for code. HoneyPrint database and trained weights are proprietary and not included.
