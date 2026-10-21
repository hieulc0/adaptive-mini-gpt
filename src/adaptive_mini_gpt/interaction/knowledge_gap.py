"""Knowledge-gap detection and structured error diagnostics (M29 Phase 2).

Defines structured records for tracking incorrect, unsupported, and abstained
neural model interactions without conflating generation probability with factual correctness.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from adaptive_mini_gpt.evaluation.multi_domain import DomainType


class ErrorCategory(str, Enum):
    """Fine-grained error categories distinguishing failure modalities."""

    INCORRECT = "incorrect"
    UNSUPPORTED = "unsupported"
    EXPLICIT_ABSTENTION = "explicit_abstention"
    MISSING_INFORMATION = "missing_information"
    REPEATED_PROCEDURAL_FAILURE = "repeated_procedural_failure"


@dataclass
class KnowledgeGapRecord:
    """A compact structured record of an interaction failure or knowledge deficit."""

    gap_id: str
    timestamp: str
    domain: DomainType
    question: str
    context: str | None
    model_checkpoint: str
    predicted_answer: str
    ground_truth: str | None
    confidence_score: float
    verification_status: str  # "PASS", "FAIL", "ABSTAIN", "UNSUPPORTED"
    error_category: ErrorCategory
    supporting_evidence: str | None = None
    immediate_correction_succeeded: bool = False
    candidate_learning_objective: str = ""
    occurrence_count: int = 1

    @classmethod
    def create(
        cls,
        domain: DomainType,
        question: str,
        predicted_answer: str,
        ground_truth: str | None,
        confidence_score: float,
        verification_status: str,
        error_category: ErrorCategory,
        model_checkpoint: str = "base_1m",
        context: str | None = None,
        supporting_evidence: str | None = None,
        immediate_correction_succeeded: bool = False,
        candidate_learning_objective: str = "",
    ) -> KnowledgeGapRecord:
        """Construct a record with a deterministic gap_id fingerprint."""
        raw_key = f"{domain.value}::{question.strip().lower()}::{candidate_learning_objective.strip().lower()}"
        gap_id = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if not candidate_learning_objective:
            candidate_learning_objective = f"Master {domain.value} question: {question[:40]}..."

        return cls(
            gap_id=gap_id,
            timestamp=now,
            domain=domain,
            question=question,
            context=context,
            model_checkpoint=model_checkpoint,
            predicted_answer=predicted_answer,
            ground_truth=ground_truth,
            confidence_score=round(confidence_score, 4),
            verification_status=verification_status,
            error_category=error_category,
            supporting_evidence=supporting_evidence,
            immediate_correction_succeeded=immediate_correction_succeeded,
            candidate_learning_objective=candidate_learning_objective,
            occurrence_count=1,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        data = asdict(self)
        data["domain"] = self.domain.value
        data["error_category"] = self.error_category.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeGapRecord:
        """Deserialize from dictionary."""
        data_copy = dict(data)
        data_copy["domain"] = DomainType(data_copy["domain"])
        data_copy["error_category"] = ErrorCategory(data_copy["error_category"])
        return cls(**data_copy)


class KnowledgeGapTracker:
    """Aggregates and tracks detected knowledge gaps across interactions."""

    def __init__(self) -> None:
        self.records: dict[str, KnowledgeGapRecord] = {}
        self.interaction_log: list[dict[str, Any]] = []

    def record_interaction(
        self,
        domain: DomainType,
        question: str,
        predicted_answer: str,
        ground_truth: str | None,
        confidence_score: float,
        is_verified: bool,
        verification_status: str,
        context: str | None = None,
        supporting_evidence: str | None = None,
        model_checkpoint: str = "base_1m",
        candidate_learning_objective: str = "",
    ) -> KnowledgeGapRecord | None:
        """Process an interaction outcome and log knowledge gaps when verification fails."""
        self.interaction_log.append({
            "domain": domain.value,
            "question": question,
            "predicted_answer": predicted_answer,
            "ground_truth": ground_truth,
            "confidence_score": confidence_score,
            "is_verified": is_verified,
            "verification_status": verification_status,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })

        if is_verified and verification_status == "PASS":
            return None

        # Categorize the error
        is_abstain = (
            predicted_answer.strip().lower() == "insufficient information."
            or verification_status == "ABSTAIN"
        )
        if is_abstain:
            if ground_truth and ground_truth.strip().lower() == "insufficient information.":
                # Appropriate abstention, not an error
                return None
            elif context and ground_truth:
                # Model abstained when answer was present in context
                category = ErrorCategory.EXPLICIT_ABSTENTION
            else:
                category = ErrorCategory.MISSING_INFORMATION
        elif verification_status == "UNSUPPORTED":
            category = ErrorCategory.UNSUPPORTED
        else:
            category = ErrorCategory.INCORRECT

        record = KnowledgeGapRecord.create(
            domain=domain,
            question=question,
            context=context,
            model_checkpoint=model_checkpoint,
            predicted_answer=predicted_answer,
            ground_truth=ground_truth,
            confidence_score=confidence_score,
            verification_status=verification_status,
            error_category=category,
            supporting_evidence=supporting_evidence,
            candidate_learning_objective=candidate_learning_objective,
        )

        if record.gap_id in self.records:
            self.records[record.gap_id].occurrence_count += 1
            if self.records[record.gap_id].occurrence_count >= 2:
                self.records[record.gap_id].error_category = ErrorCategory.REPEATED_PROCEDURAL_FAILURE
        else:
            self.records[record.gap_id] = record

        return self.records[record.gap_id]

    def get_repeated_gaps(self, min_occurrences: int = 2) -> list[KnowledgeGapRecord]:
        """Return gaps that have failed repeatedly."""
        return [r for r in self.records.values() if r.occurrence_count >= min_occurrences]

    def get_gaps_by_domain(self, domain: DomainType) -> list[KnowledgeGapRecord]:
        """Filter gaps by domain."""
        return [r for r in self.records.values() if r.domain == domain]

    def compute_diagnostics(self) -> dict[str, Any]:
        """Compute precision and rate metrics for knowledge-gap diagnostics."""
        total = len(self.interaction_log)
        if total == 0:
            return {
                "total_interactions": 0,
                "total_gaps": 0,
                "gap_rate": 0.0,
                "false_confidence_rate": 0.0,
                "unnecessary_abstention_rate": 0.0,
                "gap_detection_precision": 1.0,
            }

        false_confidence = sum(
            1 for item in self.interaction_log
            if not item["is_verified"] and item["confidence_score"] >= 0.70
        )
        unnecessary_abstention = sum(
            1 for item in self.interaction_log
            if item["predicted_answer"].strip().lower() == "insufficient information."
            and item["ground_truth"]
            and item["ground_truth"].strip().lower() != "insufficient information."
        )
        total_gaps = len(self.records)

        return {
            "total_interactions": total,
            "total_gaps": total_gaps,
            "gap_rate": round(total_gaps / total, 4),
            "false_confidence_rate": round(false_confidence / total, 4),
            "unnecessary_abstention_rate": round(unnecessary_abstention / total, 4),
            "gap_detection_precision": round((total_gaps - false_confidence) / max(1, total_gaps), 4),
        }
