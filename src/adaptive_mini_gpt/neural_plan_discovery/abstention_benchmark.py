"""Phase 7: Unknown Procedures and Bounded Abstention Benchmark.

Evaluates whether the autonomous plan discovery controller correctly identifies
when a task is uncomposable / outside its existing neural primitive repertoire,
measuring:
- True Positive: Plan chosen when task is genuinely composable
- False Positive: Plan chosen when task is uncomposable (erroneous forced composition)
- True Negative: Abstains when task is uncomposable (correct fallback request)
- False Negative: Abstains when task was composable (missed reuse opportunity)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.neural_plan_discovery.identifiability_tasks import (
    PlanIdentifiabilityGenerator,
    TaskFamilyClass,
)
from adaptive_mini_gpt.neural_plan_discovery.plan_discovery_engine import (
    AutonomousPlanDiscoveryEngine,
    PlanSearchResult,
)
from adaptive_mini_gpt.neural_skill_composition.compound_tasks import CompoundTaskType
from adaptive_mini_gpt.neural_skill_composition.primitive_skills import (
    PrimitiveSkillType,
    QualifiedPrimitiveRecord,
)

logger = logging.getLogger(__name__)


@dataclass
class AbstentionEvaluationReport:
    """Report detailing controller discrimination on composable vs uncomposable tasks."""

    total_tasks_evaluated: int
    composable_tasks: int
    uncomposable_tasks: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    abstention_accuracy: float
    true_negative_rate: float  # Correct abstention rate on uncomposable
    true_positive_rate: float  # Correct composition rate on composable
    task_details: list[dict[str, Any]]


class AbstentionEvaluator:
    """Evaluates plan discovery behavior across a mixed stream of composable and uncomposable tasks."""

    @classmethod
    def evaluate_mixed_stream(
        cls,
        model: MiniGPT,
        primitives: dict[PrimitiveSkillType, QualifiedPrimitiveRecord],
        device: torch.device,
        seed: int = 42,
        n_demo: int = 4,
    ) -> AbstentionEvaluationReport:
        """Run plan discovery across 4 composable tasks and 2 uncomposable tasks."""
        generator = PlanIdentifiabilityGenerator(vocab_size=16)

        # 4 composable tasks
        composable_specs = [
            ("comp_reflect_shift", CompoundTaskType.REFLECT_THEN_SHIFT3),
            ("comp_shift_reflect", CompoundTaskType.SHIFT3_THEN_REFLECT),
            ("comp_reflect_affine", CompoundTaskType.REFLECT_THEN_AFFINE),
            ("comp_affine_shift", CompoundTaskType.AFFINE_THEN_SHIFT3),
        ]
        # 2 uncomposable tasks
        uncomposable_specs = [
            ("uncomp_bitwise_xor", "BITWISE_XOR_MASK"),
            ("uncomp_deceptive_inv", "DECEPTIVE_INVERSION"),
        ]

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        details: list[dict[str, Any]] = []

        # 1. Evaluate composable tasks
        for idx, (t_id, c_type) in enumerate(composable_specs):
            task = generator.generate_task(
                task_id=f"{t_id}_s{seed}",
                compound_type=c_type,
                seed=seed * 10 + idx,
                n_demo=n_demo,
            )
            res = AutonomousPlanDiscoveryEngine.discover_plan(
                model=model,
                task_id=task.task_id,
                demonstrations=task.demonstrations,
                primitives=primitives,
                device=device,
            )
            if not res.is_abstained:
                tp += 1
                decision = "CORRECT_COMPOSITION_SELECTED"
            else:
                fn += 1
                decision = "FALSE_ABSTENTION_MISSED_REUSE"

            details.append({
                "task_id": task.task_id,
                "is_composable": True,
                "selected_stages": res.selected_plan.stages,
                "is_abstained": res.is_abstained,
                "confidence_score": res.confidence_score,
                "decision": decision,
            })

        # 2. Evaluate uncomposable tasks
        for idx, (t_id, u_rule) in enumerate(uncomposable_specs):
            task = generator.generate_task(
                task_id=f"{t_id}_s{seed}",
                unfamiliar_rule=u_rule,
                seed=seed * 10 + 5 + idx,
                n_demo=n_demo,
            )
            res = AutonomousPlanDiscoveryEngine.discover_plan(
                model=model,
                task_id=task.task_id,
                demonstrations=task.demonstrations,
                primitives=primitives,
                device=device,
            )
            if res.is_abstained:
                tn += 1
                decision = "CORRECT_ABSTENTION_FALLBACK_REQUESTED"
            else:
                fp += 1
                decision = "FALSE_POSITIVE_FORCED_INVALID_PLAN"

            details.append({
                "task_id": task.task_id,
                "is_composable": False,
                "selected_stages": res.selected_plan.stages,
                "is_abstained": res.is_abstained,
                "confidence_score": res.confidence_score,
                "decision": decision,
            })

        total = tp + fp + tn + fn
        acc = (tp + tn) / total if total > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return AbstentionEvaluationReport(
            total_tasks_evaluated=total,
            composable_tasks=len(composable_specs),
            uncomposable_tasks=len(uncomposable_specs),
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            abstention_accuracy=round(acc, 4),
            true_negative_rate=round(tnr, 4),
            true_positive_rate=round(tpr, 4),
            task_details=details,
        )
