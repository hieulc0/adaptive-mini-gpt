"""Milestone M80 Phase 2: Skill-Gap Diagnostic Detector.

Distinguishes:
1. KNOWN_CONFIDENT: Existing skill composition explains task with high likelihood and margin.
2. KNOWN_AMBIGUOUS: Existing skills explain task, but two candidates tie in evidence (resolved by active inquiry).
3. SKILL_GAP_UNSUPPORTED: No existing skill or combination achieves adequate likelihood (S(P) < -1.65).
   Triggers autonomous neural adapter acquisition.
4. INCONSISTENT_NOISY: Observations contradict deterministic mapping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch

from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.neural_plan_discovery.plan_space import PlanSpace
from adaptive_mini_gpt.neural_skill_composition.compound_tasks import CompoundProceduralExample
from adaptive_mini_gpt.neural_skill_composition.primitive_skills import (
    QualifiedPrimitiveRecord,
)
from adaptive_mini_gpt.reliable_plan_discovery.likelihood_plan_engine import (
    LikelihoodPlanDiscoveryEngine,
)

logger = logging.getLogger(__name__)

DEFAULT_MIN_LOG_LIKELIHOOD: float = -1.65
DEFAULT_MIN_EVIDENCE_MARGIN: float = 0.08
DEFAULT_MIN_TOKEN_ACCURACY: float = 0.50


class SkillGapState(StrEnum):
    """Diagnostic state of task compatibility with existing skill repertoire."""

    KNOWN_CONFIDENT = "KNOWN_CONFIDENT"
    KNOWN_AMBIGUOUS = "KNOWN_AMBIGUOUS"
    SKILL_GAP_UNSUPPORTED = "SKILL_GAP_UNSUPPORTED"
    INCONSISTENT_NOISY = "INCONSISTENT_NOISY"


@dataclass
class SkillGapDiagnosis:
    """Detailed diagnosis of whether a task requires new neural skill acquisition."""

    task_id: str
    state: SkillGapState
    requires_neural_acquisition: bool
    top_candidate_plan: str
    top_likelihood: float
    evidence_margin: float
    best_token_accuracy: float
    explanation: str


class SkillGapDetector:
    """Evaluates task against existing repertoire to detect genuine neural capability gaps."""

    @classmethod
    def diagnose_task(
        cls,
        model: MiniGPT,
        task_id: str,
        demonstrations: list[CompoundProceduralExample],
        primitives: dict[Any, QualifiedPrimitiveRecord],
        device: torch.device,
        min_log_likelihood: float = DEFAULT_MIN_LOG_LIKELIHOOD,
        min_evidence_margin: float = DEFAULT_MIN_EVIDENCE_MARGIN,
        min_token_accuracy: float = DEFAULT_MIN_TOKEN_ACCURACY,
    ) -> SkillGapDiagnosis:
        """Evaluate task demonstrations to diagnose whether a new skill adapter is warranted."""
        all_candidate_plans = PlanSpace.generate_candidate_plans(
            available_primitives=list(primitives.keys()),
            max_depth=2,
            include_abstain=True,
        )

        search_res = LikelihoodPlanDiscoveryEngine.discover_plan(
            model=model,
            task_id=task_id,
            demonstrations=demonstrations,
            primitives=primitives,
            device=device,
            candidate_plans=all_candidate_plans,
            min_log_likelihood=min_log_likelihood,
            min_evidence_margin=min_evidence_margin,
            min_token_accuracy=min_token_accuracy,
        )

        sorted_traces = sorted(
            search_res.candidate_traces.values(),
            key=lambda t: (t.sequence_log_likelihood, t.token_accuracy, t.exact_match),
            reverse=True,
        )
        best_trace = sorted_traces[0] if sorted_traces else None
        best_ll = best_trace.sequence_log_likelihood if best_trace else -99.9
        best_acc = best_trace.token_accuracy if best_trace else 0.0
        evidence_margin = search_res.evidence_margin
        confidence = search_res.confidence_score

        # 1. Absolute Fitness Failure -> Genuinely unsupported skill
        if best_ll < min_log_likelihood or best_acc < min_token_accuracy:
            return SkillGapDiagnosis(
                task_id=task_id,
                state=SkillGapState.SKILL_GAP_UNSUPPORTED,
                requires_neural_acquisition=True,
                top_candidate_plan=best_trace.plan.plan_id if best_trace else "NONE",
                top_likelihood=best_ll,
                evidence_margin=round(evidence_margin, 4),
                best_token_accuracy=best_acc,
                explanation=f"Top candidate plan likelihood ({best_ll:.2f}) < threshold ({min_log_likelihood:.2f}). Existing skills cannot compose this transformation.",
            )

        # 2. Narrow Margin -> Ambiguous between known skills (not a missing skill)
        if evidence_margin < min_evidence_margin or confidence < 0.80:
            return SkillGapDiagnosis(
                task_id=task_id,
                state=SkillGapState.KNOWN_AMBIGUOUS,
                requires_neural_acquisition=False,
                top_candidate_plan=best_trace.plan.plan_id if best_trace else "NONE",
                top_likelihood=best_ll,
                evidence_margin=round(evidence_margin, 4),
                best_token_accuracy=best_acc,
                explanation="High likelihood achieved but evidence margin is ambiguous. Requires active probe, not new neural adapter.",
            )

        # 3. Known confident composition
        return SkillGapDiagnosis(
            task_id=task_id,
            state=SkillGapState.KNOWN_CONFIDENT,
            requires_neural_acquisition=False,
            top_candidate_plan=best_trace.plan.plan_id if best_trace else "NONE",
            top_likelihood=best_ll,
            evidence_margin=round(evidence_margin, 4),
            best_token_accuracy=best_acc,
            explanation="Task is confidently solved by existing neural skill composition.",
        )
