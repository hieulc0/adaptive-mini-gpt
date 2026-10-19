"""Structured Working Memory with Provenance and Span Tracking (M15).

Implements bounded, in-process working memory representing resolved information:
- Operation name
- Extracted argument values
- Exact source-span provenance (source text, start/end character offsets, type)
- Deterministic verification state and result
- Explicit rejection / abstention reasons

Safety:
- In-process, bounded dictionary/dataclass representation.
- No vector database, persistent external storage, or arbitrary code execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArgumentProvenance:
    """Provenance tracking for an individual argument slot."""

    slot_name: str
    source_text: str
    start_idx: int
    end_idx: int
    parsed_value: Any
    type_name: str
    extraction_method: str = "span_copy"  # "span_copy" | "regex" | "autoregressive" | "derived"

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_name": self.slot_name,
            "source_text": self.source_text,
            "span": [self.start_idx, self.end_idx],
            "parsed_value": self.parsed_value,
            "type_name": self.type_name,
            "extraction_method": self.extraction_method,
        }


@dataclass
class WorkingMemory:
    """Structured in-process working memory for rule reasoning and execution."""

    operation: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, ArgumentProvenance] = field(default_factory=dict)
    verified: bool = False
    result: Any | None = None
    status: str = "empty"  # "empty" | "grounded" | "verified" | "rejected"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def bind_operation(self, operation: str) -> None:
        """Set active operation."""
        self.operation = operation.strip().lower()
        self.status = "operation_selected"

    def bind_argument(
        self,
        slot_name: str,
        value: Any,
        source_text: str,
        span: tuple[int, int],
        type_name: str,
        extraction_method: str = "span_copy",
    ) -> None:
        """Bind an argument value with full source-span provenance."""
        self.arguments[slot_name] = value
        self.provenance[slot_name] = ArgumentProvenance(
            slot_name=slot_name,
            source_text=source_text,
            start_idx=span[0],
            end_idx=span[1],
            parsed_value=value,
            type_name=type_name,
            extraction_method=extraction_method,
        )
        self.status = "grounded"

    def bind_result(self, result: Any, verified: bool = True) -> None:
        """Record verified execution result."""
        self.result = result
        self.verified = verified
        self.status = "verified" if verified else "unverified"

    def reject(self, reason: str) -> None:
        """Explicitly reject reasoning or execution with justification."""
        self.verified = False
        self.status = "rejected"
        self.error = reason

    def is_complete_for_rule(self, required_slots: list[str]) -> bool:
        """Check if all required slots are bound in working memory."""
        return all(slot in self.arguments for slot in required_slots)

    def to_dict(self) -> dict[str, Any]:
        """Convert working memory to serializable dictionary representation."""
        return {
            "operation": self.operation,
            "arguments": copy_serializable(self.arguments),
            "provenance": {k: v.to_dict() for k, v in self.provenance.items()},
            "verified": self.verified,
            "result": copy_serializable(self.result),
            "status": self.status,
            "error": self.error,
            "metadata": copy_serializable(self.metadata),
        }


def copy_serializable(val: Any) -> Any:
    """Helper to ensure JSON-serializable representation of memory attributes."""
    if isinstance(val, (int, float, str, bool)) or val is None:
        return val
    if isinstance(val, list):
        return [copy_serializable(x) for x in val]
    if isinstance(val, dict):
        return {str(k): copy_serializable(v) for k, v in val.items()}
    return str(val)
