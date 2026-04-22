"""Tests for single-sample prediction endpoint."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from app.main import app
from app.utils.synthetic_data import generate_synthetic_spectrum


def test_predict_returns_expected_payload() -> None:
    """POST /predict should return 200 and expected response keys."""
    with TestClient(app) as client:
        payload = {
            "scan_id": str(uuid.uuid4()),
            "spectral_array": generate_synthetic_spectrum("pure").tolist(),
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        expected = {
            "scan_id",
            "class_label",
            "class_probabilities",
            "adulteration_pct",
            "inference_ms",
            "audit_hash",
            "timestamp",
        }
        assert expected.issubset(set(data.keys()))
        assert "inference_ms" in data
