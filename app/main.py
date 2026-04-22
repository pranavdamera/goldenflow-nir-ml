"""FastAPI application entrypoint for GoldenFlow NIR inference."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.audit.chain import AuditChain
from app.ml.onnx_runner import ONNXRunner
from app.ml.preprocessor import SpectralPreprocessor
from app.routes.batch import router as batch_router
from app.routes.predict import router as predict_router


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load shared ML dependencies once at startup to eliminate per-request cold starts."""
    application.state.onnx_runner = ONNXRunner("honey_nir_net.onnx")
    application.state.preprocessor = SpectralPreprocessor()
    application.state.audit_chain = AuditChain()
    yield


app = FastAPI(title="GoldenFlow NIR ML API", version="1.0.0", lifespan=lifespan)

app.include_router(predict_router)
app.include_router(batch_router)


@app.get("/health")
def health() -> dict[str, str]:
    """Return a simple health status for service checks."""
    return {"status": "ok"}
