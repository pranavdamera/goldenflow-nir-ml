"""Cryptographic audit chain for tamper-resistant inference traceability.

Tamper detection works by chaining hashes over sequential inference records:
any modification to historical output changes `output_hash`, which then breaks
all downstream `chain_hash` values and is therefore detectable.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class AuditChain:
    """Append-only SHA-256 audit chain for inference records."""

    def __init__(self) -> None:
        """Initialize an in-memory pointer to previous chain hash."""
        self.previous_chain_hash = "GENESIS"

    @staticmethod
    def _sha256(text: str) -> str:
        """Compute SHA-256 hash over UTF-8 encoded text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def create_record(self, scan_id: str, input_hash: str, output_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Create a signed audit record from model output payload."""
        output_payload = json.dumps(output_dict, sort_keys=True)
        output_hash = self._sha256(output_payload)
        chain_hash = self._sha256(f"{self.previous_chain_hash}{output_hash}")
        timestamp = datetime.now(timezone.utc).isoformat()

        record = {
            "scan_id": scan_id,
            "timestamp": timestamp,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "chain_hash": chain_hash,
            "output": output_dict,
        }
        self.previous_chain_hash = chain_hash
        return record

    def verify_record(self, record: Dict[str, Any], previous_chain_hash: str) -> bool:
        """Verify a record against the expected previous chain hash."""
        required_keys = {"scan_id", "timestamp", "input_hash", "output_hash", "chain_hash", "output"}
        if not required_keys.issubset(set(record.keys())):
            return False

        recomputed_output_hash = self._sha256(json.dumps(record["output"], sort_keys=True))
        if recomputed_output_hash != record["output_hash"]:
            return False

        recomputed_chain_hash = self._sha256(f"{previous_chain_hash}{record['output_hash']}")
        return recomputed_chain_hash == record["chain_hash"]

    def append_to_log(self, record: Dict[str, Any], filepath: str = "audit_log.jsonl") -> None:
        """Append one JSON record to a newline-delimited audit log file."""
        path = Path(filepath)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
