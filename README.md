# GoldenFlow Labs — NIR Spectral Classification Engine

Open-source ML inference stack for portable honey adulteration detection via near-infrared spectroscopy.

## Overview

Honey adulteration is a global food integrity issue that contributes to major economic loss, including an estimated $750M+ annual impact in the US market. GoldenFlow Labs builds portable NIR + AI tooling to detect adulteration risk at point-of-test.

Near-infrared (NIR) spectroscopy captures molecular absorption signatures from 900-1700nm, enabling fast, non-destructive fingerprinting of honey samples. This repository provides the open-source inference and MLOps framework used to process spectra, run classification/regression inference, and produce tamper-evident audit records.

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
Raw Spectral Input
        |
        v
Preprocessor (SNV + SG filter)
        |
        v
HoneyNIRNet (1D-CNN + SE Attention)
        |
        v
ONNX Runtime
        |
        v
{Class Label + Adulteration %}
        |
        v
Audit Chain (SHA-256)
```

## Model

`HoneyNIRNet` combines 1D convolutional feature extraction with Squeeze-Excitation channel attention to capture adulteration-sensitive spectral patterns.

Output classes:
- `pure`: likely unadulterated honey profile
- `rice_syrup`: rice syrup adulteration signature
- `hfcs`: high-fructose corn syrup signature
- `jaggery_syrup`: jaggery syrup signature
- `invert_sugar`: invert sugar signature
- `sugar_fed`: likely sugar-fed source profile
- `unknown`: non-conforming or ambiguous profile

Training pipeline:
- Focal Loss for class imbalance mitigation
- LOBO (Leave-One-Batch-Out) cross-validation
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
architecture / pipelines / code for the open source
