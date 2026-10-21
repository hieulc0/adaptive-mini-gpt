"""Provenance-aware verified memory bank (M29 Phase 4).

Enforces strict provenance, verification auditing, and source validation.
Only stores facts verified by independent oracles or approved documents.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from adaptive_mini_gpt.evaluation.multi_domain import DomainType


@dataclass
class VerifiedMemoryEntry:
    """A single verified fact with explicit provenance and validation audit."""

    entry_id: str
    domain: DomainType
    topic_or_key: str
    verified_fact: str
    evidence_source: str
    verifier_identity: str
    model_version: str
    timestamp: str
    status: str = "verified"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["domain"] = self.domain.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VerifiedMemoryEntry:
        data_copy = dict(data)
        data_copy["domain"] = DomainType(data_copy["domain"])
        return cls(**data_copy)


class VerifiedMemoryBank:
    """Stores and retrieves verified knowledge with source provenance auditing."""

    APPROVED_SOURCE_PREFIXES = (
        "approved_local_doc:",
        "arithmetic_oracle",
        "logic_truth_table",
        "verified_exercise_generator",
    )

    def __init__(self, storage_path: Path | None = None) -> None:
        self.storage_path = storage_path
        self.entries: dict[str, VerifiedMemoryEntry] = {}
        if storage_path and storage_path.exists():
            self.load(storage_path)

    def store_verified_fact(
        self,
        domain: DomainType,
        topic_or_key: str,
        verified_fact: str,
        evidence_source: str,
        verifier_identity: str,
        model_version: str = "base_1m",
    ) -> VerifiedMemoryEntry:
        """Store a fact only if it originates from an approved, auditable source."""
        # Security invariant: prevent prompt-injected or unchecked assertions
        is_approved = any(evidence_source.startswith(p) for p in self.APPROVED_SOURCE_PREFIXES)
        if not is_approved:
            raise ValueError(
                f"Security violation: evidence_source '{evidence_source}' is not on the approved "
                f"provenance whitelist: {self.APPROVED_SOURCE_PREFIXES}"
            )

        if not verified_fact.strip():
            raise ValueError("Cannot store empty fact in verified memory")

        entry_key = f"{domain.value}::{topic_or_key.strip().lower()}"
        entry_id = hashlib.sha256(entry_key.encode("utf-8")).hexdigest()[:16]
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        entry = VerifiedMemoryEntry(
            entry_id=entry_id,
            domain=domain,
            topic_or_key=topic_or_key.strip(),
            verified_fact=verified_fact.strip(),
            evidence_source=evidence_source,
            verifier_identity=verifier_identity,
            model_version=model_version,
            timestamp=now,
            status="verified",
        )
        self.entries[entry_id] = entry

        if self.storage_path:
            self.save(self.storage_path)

        return entry

    def retrieve(self, domain: DomainType, query: str) -> VerifiedMemoryEntry | None:
        """Search verified memory for the most relevant fact for the query."""
        q_tokens = set(query.lower().split())
        best_entry: VerifiedMemoryEntry | None = None
        max_overlap = 0

        for entry in self.entries.values():
            if entry.domain != domain:
                continue
            key_tokens = set(entry.topic_or_key.lower().split())
            overlap = len(q_tokens.intersection(key_tokens))
            if overlap > max_overlap:
                max_overlap = overlap
                best_entry = entry

        return best_entry

    def get_all(self, domain: DomainType | None = None) -> list[VerifiedMemoryEntry]:
        """Return all stored entries, optionally filtered by domain."""
        if domain is None:
            return list(self.entries.values())
        return [e for e in self.entries.values() if e.domain == domain]

    def save(self, path: Path) -> None:
        """Persist memory to a JSON file."""
        data = {k: v.to_dict() for k, v in self.entries.items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path) -> None:
        """Load memory from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.entries = {k: VerifiedMemoryEntry.from_dict(v) for k, v in data.items()}
