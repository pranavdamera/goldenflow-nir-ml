"""Tests for batch prediction endpoint."""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from app.main import app
from app.utils.synthetic_data import generate_synthetic_spectrum


def test_batch_returns_summary_counts() -> None:
    """POST /batch should process samples and report total count."""
    with TestClient(app) as client:
        samples = [
            {"scan_id": str(uuid.uuid4()), "spectral_array": generate_synthetic_spectrum("hfcs").tolist()}
            for _ in range(50)
        ]
        response = client.post("/batch", json={"samples": samples})
        assert response.status_code == 200
        payload = response.json()
        assert payload["batch_summary"]["total_samples"] == 50
