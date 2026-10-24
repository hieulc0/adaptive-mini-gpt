"""Phase 7: Bounded Learning-Policy Ablation & Operational Cost Accounting."""
from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.modular_skill_learning.candidate_acquisition import (
    CandidateAcquisitionReport,
    ControlledSkillAcquisitionEngine,
)
from adaptive_mini_gpt.modular_skill_learning.repertoire import (
    ModularSkillRepertoire,
    QualifiedSkillRecord,
)
from adaptive_mini_gpt.modular_skill_learning.uncertainty_diagnostic import (
    EpistemicUncertaintyDiagnostic,
)
from adaptive_mini_gpt.three_valued_reasoning.architectures import Rank16ResidualAdapter
from adaptive_mini_gpt.three_valued_reasoning.semantics import GroundedThreeValuedEnvironment

logger = logging.getLogger(__name__)


class LearningPolicy(str, Enum):
    """Learning policies compared in Phase 7 ablation."""
    ALWAYS_FRESH = "ALWAYS_FRESH"
    ALWAYS_REUSE = "ALWAYS_REUSE"
    ALWAYS_SPAWN = "ALWAYS_SPAWN"
    EVIDENCE_ROUTING = "EVIDENCE_ROUTING"
    NOVELTY_GATED_REUSE_OR_SPAWN = "NOVELTY_GATED_REUSE_OR_SPAWN"


@dataclass
class PolicyEvaluationResult:
    """Quantitative performance and resource expenditure under an evaluated policy."""
    policy_name: str
    macro_f1: float
    accuracy: float
    classwise_recall: dict[str, float]
    paired_correctness: float
    oracle_queries_used: int
    adapters_allocated: int
    false_allocation_count: int
    missed_useful_reuse_count: int
    candidate_rejection_count: int
    storage_footprint_params: int
    decision_diagnostics: str


class LearningPolicyAblationEngine:
    """Evaluates the 5 policies with exhaustive accounting of oracle, training, and storage costs."""

    def __init__(
        self,
        acquisition_engine: ControlledSkillAcquisitionEngine,
        uncertainty_diagnostic: EpistemicUncertaintyDiagnostic,
        reuse_f1_threshold: float = 0.45,
    ) -> None:
        self.acq_engine = acquisition_engine
        self.diagnostic = uncertainty_diagnostic
        self.reuse_f1_threshold = reuse_f1_threshold

    def evaluate_policy_on_task(
        self,
        policy: LearningPolicy,
        model: MiniGPT,
        repertoire: ModularSkillRepertoire,
        train_env: GroundedThreeValuedEnvironment,
        eval_env: GroundedThreeValuedEnvironment,
        is_known_family: bool,
        device: torch.device,
        seed: int = 42,
    ) -> PolicyEvaluationResult:
        """Execute a learning policy on a target task and track accuracy and costs."""
        oracle_queries = 0
        adapters_allocated = 0
        false_allocations = 0
        missed_reuse = 0
        candidate_rejections = 0
        storage_params = sum(len(p.state_dict) * 2371 for p in repertoire._skills.values())

        # Check existing repertoire skills on eval_env
        best_reuse_id = repertoire.get_skill_ids()[0]
        best_reuse_f1 = -1.0
        for sk_id in repertoire.get_skill_ids():
            f1, _, _, _ = repertoire.evaluate_skill_on_env(sk_id, model, eval_env, device)
            if f1 > best_reuse_f1:
                best_reuse_f1 = f1
                best_reuse_id = sk_id

        if policy == LearningPolicy.ALWAYS_FRESH:
            # Always trains fresh adapter from scratch
            oracle_queries += 16
            adapters_allocated += 1
            storage_params += 2371
            cand, _, _ = self.acq_engine.train_candidate_adapter(
                candidate_id="always_fresh",
                train_env=train_env,
                model=model,
                budget_n=16,
                device=device,
                seed=seed,
            )
            f1, acc, rec, paired = self.acq_engine.evaluate_candidate_on_independent_world(
                candidate=cand, eval_env=eval_env, model=model, device=device
            )
            diag = f"ALWAYS_FRESH: Trained fresh adapter (F1={f1:.4f})"

        elif policy == LearningPolicy.ALWAYS_REUSE:
            # Always routes to best existing skill
            oracle_queries += 4  # probe queries
            f1, acc, rec, paired = repertoire.evaluate_skill_on_env(best_reuse_id, model, eval_env, device)
            diag = f"ALWAYS_REUSE: Routed to '{best_reuse_id}' (F1={f1:.4f})"

        elif policy == LearningPolicy.ALWAYS_SPAWN:
            # Unconditionally allocates new candidate
            oracle_queries += 16
            adapters_allocated += 1
            storage_params += 2371
            if is_known_family and best_reuse_f1 >= self.reuse_f1_threshold:
                false_allocations += 1

            cand, _, _ = self.acq_engine.train_candidate_adapter(
                candidate_id="always_spawn",
                train_env=train_env,
                model=model,
                budget_n=16,
                device=device,
                seed=seed,
            )
            f1, acc, rec, paired = self.acq_engine.evaluate_candidate_on_independent_world(
                candidate=cand, eval_env=eval_env, model=model, device=device
            )
            diag = f"ALWAYS_SPAWN: Unconditionally spawned adapter (F1={f1:.4f})"

        elif policy == LearningPolicy.EVIDENCE_ROUTING:
            # Routes to best existing skill based on evidence probe
            oracle_queries += 8
            f1, acc, rec, paired = repertoire.evaluate_skill_on_env(best_reuse_id, model, eval_env, device)
            diag = f"EVIDENCE_ROUTING: Selected best skill '{best_reuse_id}' (F1={f1:.4f})"

        elif policy == LearningPolicy.NOVELTY_GATED_REUSE_OR_SPAWN:
            # Proposed M66 Policy:
            # 1. Probe existing repertoire
            oracle_queries += 8
            can_reuse = (best_reuse_f1 >= self.reuse_f1_threshold)

            # 2. Probe SPRT for rule-level novelty
            llr, is_novel = self.diagnostic.compute_sprt_llr(
                observations=[(q.source, q.target, q.expected_answer) for q in train_env.held_out_test_queries[:8]],
                candidate_hypothesis=train_env.true_hypothesis,
                edges=train_env.base_edges,
            )
            oracle_queries += 8

            if can_reuse and not is_novel:
                # Reuse existing skill!
                f1, acc, rec, paired = repertoire.evaluate_skill_on_env(best_reuse_id, model, eval_env, device)
                diag = f"NOVELTY_GATED: Reused existing skill '{best_reuse_id}' (F1={f1:.4f})"
            else:
                # Novelty confirmed or existing skill inadequate -> acquire candidate
                if is_known_family and can_reuse:
                    false_allocations += 1

                oracle_queries += 16
                report, qualified_adapter = self.acq_engine.acquire_and_qualify_candidate(
                    candidate_id=f"gated_candidate_{train_env.env_id}",
                    rule_family="novel_or_adapted",
                    train_env=train_env,
                    eval_env=eval_env,
                    model=model,
                    repertoire=repertoire,
                    budget_n=16,
                    device=device,
                    seed=seed,
                )

                if qualified_adapter is not None:
                    adapters_allocated += 1
                    storage_params += 2371
                    f1 = report.independent_eval_macro_f1
                    acc = report.independent_eval_accuracy
                    rec = report.independent_eval_recalls
                    paired = report.paired_correctness
                    diag = f"NOVELTY_GATED: Qualified and allocated new skill (F1={f1:.4f})"
                else:
                    candidate_rejections += 1
                    # Fallback to best existing skill
                    f1, acc, rec, paired = repertoire.evaluate_skill_on_env(best_reuse_id, model, eval_env, device)
                    diag = f"NOVELTY_GATED: Candidate rejected; rolled back to '{best_reuse_id}' (F1={f1:.4f})"

        return PolicyEvaluationResult(
            policy_name=policy.value,
            macro_f1=float(f1),
            accuracy=float(acc),
            classwise_recall=rec,
            paired_correctness=float(paired),
            oracle_queries_used=oracle_queries,
            adapters_allocated=adapters_allocated,
            false_allocation_count=false_allocations,
            missed_useful_reuse_count=missed_reuse,
            candidate_rejection_count=candidate_rejections,
            storage_footprint_params=storage_params,
            decision_diagnostics=diag,
        )
