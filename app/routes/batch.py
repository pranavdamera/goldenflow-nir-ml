"""Batch prediction endpoint implementation."""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import List

import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from app.audit.chain import AuditChain
from app.ml.onnx_runner import ONNXRunner
from app.ml.preprocessor import SpectralPreprocessor
from app.models.schemas import BatchRequest, BatchResponse, BatchSummary, PredictResponse
from app.routes.predict import get_audit_chain, get_preprocessor, get_runner

router = APIRouter(tags=["inference"])


@router.post("/batch", response_model=BatchResponse)
def predict_batch(
    payload: BatchRequest,
    runner: ONNXRunner = Depends(get_runner),
    preprocessor: SpectralPreprocessor = Depends(get_preprocessor),
    audit_chain: AuditChain = Depends(get_audit_chain),
) -> BatchResponse:
    """Run vectorized preprocessing/inference for up to 2000 spectra."""
    if len(payload.samples) > 2000:
        raise HTTPException(status_code=422, detail="Maximum 2000 samples per batch request.")

    start = time.perf_counter()
    processed_arrays: List[np.ndarray] = []
    for sample in payload.samples:
        raw = np.asarray(sample.spectral_array, dtype=np.float32)
        try:
            processed_arrays.append(preprocessor.full_pipeline(raw))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid spectrum for scan_id={sample.scan_id}: {exc}") from exc

    outputs = runner.run_batch(processed_arrays)
    predictions: List[PredictResponse] = []
    adulteration_values: List[float] = []

    for sample, output in zip(payload.samples, outputs, strict=True):
        raw_bytes = np.asarray(sample.spectral_array, dtype=np.float32).tobytes()
        input_hash = hashlib.sha256(raw_bytes).hexdigest()
        record = audit_chain.create_record(sample.scan_id, input_hash, output)
        audit_chain.append_to_log(record)
        adulteration_pct = float(output["adulteration_pct"])
        adulteration_values.append(adulteration_pct)

        predictions.append(
            PredictResponse(
                scan_id=sample.scan_id,
                class_label=str(output["class_label"]),
                class_probabilities=dict(output["class_probabilities"]),  # type: ignore[arg-type]
                adulteration_pct=adulteration_pct,
                inference_ms=float(output["inference_ms"]),
                audit_hash=record["chain_hash"],
                timestamp=record["timestamp"],
            )
        )

    processing_time_ms = (time.perf_counter() - start) * 1000.0
    adulterated_count = sum(value >= 0.2 for value in adulteration_values)
    batch_summary = BatchSummary(
        total_samples=len(payload.samples),
        adulterated_count=adulterated_count,
        mean_adulteration_pct=float(np.mean(adulteration_values) if adulteration_values else 0.0),
        processing_time_ms=processing_time_ms,
        audit_batch_id=str(uuid.uuid4()),
    )
    return BatchResponse(predictions=predictions, batch_summary=batch_summary)
