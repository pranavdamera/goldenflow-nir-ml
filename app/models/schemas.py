"""Pydantic schemas used by the inference API."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Request payload for single-sample prediction."""

    scan_id: str = Field(..., description="External UUID or unique scan identifier.")
    spectral_array: List[float] = Field(
        ...,
        description="Raw 256-point spectral array spanning 900-1700nm.",
        min_length=256,
        max_length=256,
    )


class PredictResponse(BaseModel):
    """Response payload for single-sample prediction."""

    scan_id: str
    class_label: str
    class_probabilities: Dict[str, float]
    adulteration_pct: float
    inference_ms: float
    audit_hash: str
    timestamp: str


class BatchSample(BaseModel):
    """A single sample item in batch prediction."""

    scan_id: str
    spectral_array: List[float] = Field(..., min_length=256, max_length=256)

    @field_validator("spectral_array")
    @classmethod
    def validate_length(cls, value: List[float]) -> List[float]:
        """Ensure each sample carries exactly 256 spectral points."""
        if len(value) != 256:
            raise ValueError("Each spectral_array must contain exactly 256 values.")
        return value


class BatchRequest(BaseModel):
    """Request payload for batch inference."""

    samples: List[BatchSample] = Field(..., min_length=1, max_length=2000)


class BatchSummary(BaseModel):
    """Aggregated metadata returned for a completed batch."""

    total_samples: int
    adulterated_count: int
    mean_adulteration_pct: float
    processing_time_ms: float
    audit_batch_id: str


class BatchResponse(BaseModel):
    """Response payload for batch inference output + summary."""

    predictions: List[PredictResponse]
    batch_summary: BatchSummary
