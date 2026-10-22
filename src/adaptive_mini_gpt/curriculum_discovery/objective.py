"""M36: Structured Learning Objective Specification.

Defines the core data structures for autonomous capability-gap discovery:
  - HypothesizedCause: Categorizes why a failure occurred.
  - LearningObjective: A formal, testable learning proposal containing:
      * Skill or concept
      * Evidence of repeated failure
      * Hypothesized cause
      * Prerequisite knowledge
      * Proposed intervention (ANSWER, RETRIEVE, PRACTICE, TRAIN_CANDIDATE, ABSTAIN)
      * Expected benefit
      * Estimated resource cost
      * Independent success measurement
"""

from __future__ import annotations

import enum
import hashlib
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from adaptive_mini_gpt.metacognition.action_space import LearningAction


class HypothesizedCause(str, enum.Enum):
    """Categorizes the underlying root cause of an interaction failure."""

    PROCEDURAL_GAP = "procedural_gap"
    FACTUAL_GAP = "factual_gap"
    REPRESENTATION_WEAKNESS = "representation_weakness"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass
class LearningObjective:
    """A structured, testable learning proposal for targeted improvement."""

    objective_id: str
    skill_or_concept: str
    evidence_of_repeated_failure: list[str]
    hypothesized_cause: HypothesizedCause
    prerequisite_knowledge: list[str]
    proposed_intervention: LearningAction
    expected_benefit: float
    estimated_resource_cost: float
    independent_success_measurement: str
    prerequisites_satisfied: bool = True
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    @classmethod
    def create(
        cls,
        skill_or_concept: str,
        evidence_of_repeated_failure: list[str],
        hypothesized_cause: HypothesizedCause,
        prerequisite_knowledge: list[str],
        proposed_intervention: LearningAction,
        expected_benefit: float,
        estimated_resource_cost: float,
        independent_success_measurement: str,
        prerequisites_satisfied: bool = True,
    ) -> LearningObjective:
        """Construct a LearningObjective with deterministic fingerprint ID."""
        fingerprint_raw = f"{skill_or_concept}::{hypothesized_cause.value}::{proposed_intervention.value}"
        obj_id = hashlib.sha256(fingerprint_raw.encode("utf-8")).hexdigest()[:16]
        return cls(
            objective_id=obj_id,
            skill_or_concept=skill_or_concept,
            evidence_of_repeated_failure=list(evidence_of_repeated_failure),
            hypothesized_cause=hypothesized_cause,
            prerequisite_knowledge=list(prerequisite_knowledge),
            proposed_intervention=proposed_intervention,
            expected_benefit=round(expected_benefit, 4),
            estimated_resource_cost=round(estimated_resource_cost, 4),
            independent_success_measurement=independent_success_measurement,
            prerequisites_satisfied=prerequisites_satisfied,
        )

    def is_valid(self) -> bool:
        """Verify structural validity and feasibility of the objective."""
        if not self.skill_or_concept or not self.skill_or_concept.strip():
            return False
        if not self.evidence_of_repeated_failure:
            return False
        if not isinstance(self.hypothesized_cause, HypothesizedCause):
            return False
        if not isinstance(self.proposed_intervention, LearningAction):
            return False
        if self.estimated_resource_cost <= 0.0:
            return False
        if not self.independent_success_measurement or not self.independent_success_measurement.strip():
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        d = asdict(self)
        d["hypothesized_cause"] = self.hypothesized_cause.value
        d["proposed_intervention"] = self.proposed_intervention.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearningObjective:
        """Deserialize from dictionary."""
        cp = dict(data)
        cp["hypothesized_cause"] = HypothesizedCause(cp["hypothesized_cause"])
        cp["proposed_intervention"] = LearningAction(cp["proposed_intervention"])
        return cls(**cp)
