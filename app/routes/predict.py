"""Single-sample prediction endpoint implementation."""

from __future__ import annotations

import hashlib

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request

from app.audit.chain import AuditChain
from app.ml.onnx_runner import ONNXRunner
from app.ml.preprocessor import SpectralPreprocessor
from app.models.schemas import PredictRequest, PredictResponse

router = APIRouter(tags=["inference"])


def get_runner(request: Request) -> ONNXRunner:
    """Return shared ONNXRunner instance from FastAPI app state."""
    return request.app.state.onnx_runner


def get_preprocessor(request: Request) -> SpectralPreprocessor:
    """Return shared SpectralPreprocessor instance from app state."""
    return request.app.state.preprocessor


def get_audit_chain(request: Request) -> AuditChain:
    """Return shared AuditChain instance from app state."""
    return request.app.state.audit_chain


@router.post("/predict", response_model=PredictResponse)
def predict_sample(
    payload: PredictRequest,
    runner: ONNXRunner = Depends(get_runner),
    preprocessor: SpectralPreprocessor = Depends(get_preprocessor),
    audit_chain: AuditChain = Depends(get_audit_chain),
) -> PredictResponse:
    """Run preprocessing, ONNX inference, and audit chain logging for one sample."""
    try:
        raw = np.asarray(payload.spectral_array, dtype=np.float32)
        processed = preprocessor.full_pipeline(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    output = runner.run_single(processed)
    raw_bytes = np.asarray(payload.spectral_array, dtype=np.float32).tobytes()
    input_hash = hashlib.sha256(raw_bytes).hexdigest()
    record = audit_chain.create_record(payload.scan_id, input_hash, output)
    audit_chain.append_to_log(record)

    return PredictResponse(
        scan_id=payload.scan_id,
        class_label=str(output["class_label"]),
        class_probabilities=dict(output["class_probabilities"]),  # type: ignore[arg-type]
        adulteration_pct=float(output["adulteration_pct"]),
        inference_ms=float(output["inference_ms"]),
        audit_hash=record["chain_hash"],
        timestamp=record["timestamp"],
    )
