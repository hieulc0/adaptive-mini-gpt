"""M37 Phase 4: Evidence Validation & Contradiction Resolution.

Audits retrieved search results against rigorous evidence standards:
  - Provenance Whitelist Verification: Enforces approved source prefixes.
  - Reliability Filtering: Rejects or discounts low-reliability / outdated sources.
  - Contradiction & Misconception Detection: Flags conflicting historical or disproven claims.
  - Prompt Injection Immunity: Neutralizes adversarial attempts to override the controller.
  - Calibrated Decision Recommendation: Recommends ANSWER, VERIFY, or ABSTAIN.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from adaptive_mini_gpt.metacognition.action_space import LearningAction
from adaptive_mini_gpt.research.corpus import DocumentType
from adaptive_mini_gpt.research.search_engine import SearchResult

APPROVED_PROVENANCE_PREFIXES = (
    "approved_local_doc:",
    "verified_memory:",
)

INJECTION_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"override (the )?controller",
    r"you are now (an? )?unrestricted",
    r"system prompt:",
    r"disregard (the )?rules",
]

CONTRADICTION_MARKERS = [
    r"\b(disproven|outdated|misconception|obsolete|false claim|satirical|fabricat)\b",
    r"\b(historically believed.*refuted|no longer considered)\b",
]


@dataclass
class EvidenceValidationResult:
    """The structured audit verdict for retrieved evidence."""

    query: str
    is_supported: bool
    contradiction_detected: bool
    prompt_injection_detected: bool
    selected_passage: str | None
    selected_doc_id: str | None
    selected_provenance: str | None
    reliability_score: float
    conflicting_assertions: list[str]
    recommended_action: LearningAction
    audit_notes: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["recommended_action"] = self.recommended_action.value
        return d


class EvidenceValidator:
    """Validates retrieved passages and resolves conflicting claims before knowledge use."""

    def __init__(self, min_trust_threshold: float = 0.75) -> None:
        self.min_trust_threshold = min_trust_threshold

    def validate_evidence(
        self,
        query: str,
        retrieved_results: list[SearchResult],
    ) -> EvidenceValidationResult:
        """Audit retrieved search results and recommend safe action."""
        if not retrieved_results:
            return EvidenceValidationResult(
                query=query,
                is_supported=False,
                contradiction_detected=False,
                prompt_injection_detected=False,
                selected_passage=None,
                selected_doc_id=None,
                selected_provenance=None,
                reliability_score=0.0,
                conflicting_assertions=[],
                recommended_action=LearningAction.ABSTAIN,
                audit_notes="No documents retrieved above relevance threshold; recommend safe abstention.",
            )

        conflicts: list[str] = []
        injection_found = False
        valid_candidates: list[SearchResult] = []

        for res in retrieved_results:
            # 1. Provenance check
            if not any(res.provenance.startswith(pref) for pref in APPROVED_PROVENANCE_PREFIXES):
                conflicts.append(f"Unapproved provenance: {res.provenance}")
                continue

            # 2. Injection check
            text_lower = res.content.lower()
            for pat in INJECTION_PATTERNS:
                if re.search(pat, text_lower):
                    injection_found = True
                    conflicts.append(f"Prompt injection pattern detected in doc '{res.doc_id}'")
                    break
            if injection_found:
                continue

            # 3. Contradiction / Obsolete theory check
            is_contradictory = (
                res.doc_type == DocumentType.CONTRADICTORY
                or res.reliability_score < 0.3
                or any(re.search(pat, text_lower) for pat in CONTRADICTION_MARKERS)
            )
            if is_contradictory:
                conflicts.append(f"Doc '{res.doc_id}' ({res.title}) contains disproven or contradictory claim.")
                continue

            if res.doc_type == DocumentType.TRUSTWORTHY and res.reliability_score >= self.min_trust_threshold:
                valid_candidates.append(res)
            elif res.doc_type == DocumentType.INCOMPLETE:
                conflicts.append(f"Doc '{res.doc_id}' is marked incomplete.")

        # Determine decision
        if injection_found:
            return EvidenceValidationResult(
                query=query,
                is_supported=False,
                contradiction_detected=True,
                prompt_injection_detected=True,
                selected_passage=None,
                selected_doc_id=None,
                selected_provenance=None,
                reliability_score=0.0,
                conflicting_assertions=conflicts,
                recommended_action=LearningAction.ABSTAIN,
                audit_notes="Adversarial prompt injection neutralized; abstaining.",
            )

        if not valid_candidates:
            has_contradiction = len(conflicts) > 0
            return EvidenceValidationResult(
                query=query,
                is_supported=False,
                contradiction_detected=has_contradiction,
                prompt_injection_detected=False,
                selected_passage=None,
                selected_doc_id=None,
                selected_provenance=None,
                reliability_score=0.0,
                conflicting_assertions=conflicts,
                recommended_action=LearningAction.ABSTAIN,
                audit_notes="No trustworthy candidate met threshold; contradictory or incomplete evidence rejected.",
            )

        best_cand = valid_candidates[0]
        return EvidenceValidationResult(
            query=query,
            is_supported=True,
            contradiction_detected=(len(conflicts) > 0),
            prompt_injection_detected=False,
            selected_passage=best_cand.content,
            selected_doc_id=best_cand.doc_id,
            selected_provenance=best_cand.provenance,
            reliability_score=best_cand.reliability_score,
            conflicting_assertions=conflicts,
            recommended_action=LearningAction.ANSWER,
            audit_notes=f"Supported by trustworthy source '{best_cand.doc_id}' (provenance={best_cand.provenance}).",
        )
