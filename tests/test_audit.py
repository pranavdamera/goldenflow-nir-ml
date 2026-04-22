"""Tests for cryptographic audit chain integrity checks."""

from __future__ import annotations

from app.audit.chain import AuditChain


def test_verify_record_detects_tampering() -> None:
    """Tampering with output payload should invalidate record verification."""
    chain = AuditChain()
    output = {
        "class_label": "pure",
        "class_probabilities": {"pure": 0.9},
        "adulteration_pct": 0.1,
        "inference_ms": 42.0,
    }
    previous = chain.previous_chain_hash
    record = chain.create_record("scan-1", "inputhash", output)

    tampered = dict(record)
    tampered_output = dict(record["output"])
    tampered_output["adulteration_pct"] = 0.8
    tampered["output"] = tampered_output

    assert chain.verify_record(tampered, previous) is False
