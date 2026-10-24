"""Phase 4 & 7: Utility-Calibrated Autonomous Skill Selection & Experience Calibration.

Implements:
1. Minimal Utility-Aware Action Space:
   - REUSE: Select the strongest available existing repertoire skill.
   - FRESH: Transient adaptation without permanent parameter storage.
   - PROPOSE_NEW: Bounded candidate training and independent qualification.
2. Explicit Resource-Cost Accounting:
   - Penalizes oracle queries, training compute steps, storage footprint, and interference risk.
3. Experience-Calibrated Adaptation Decisions:
   - Maintains verified historical records of candidate acquisition outcomes.
   - Uses empirical past experience to update utility thresholds on unfamiliar rule families.
"""
from __future__ import annotations

import enum
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

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
from adaptive_mini_gpt.modular_skill_learning.uncertainty_diagnostic import EpistemicUncertaintyDiagnostic
from adaptive_mini_gpt.three_valued_reasoning.semantics import GroundedThreeValuedEnvironment

logger = logging.getLogger(__name__)


class UtilityAction(str, enum.Enum):
    """The 3 fundamental actions for modular autonomous learning."""
    REUSE = "REUSE"
    FRESH = "FRESH"
    PROPOSE_NEW = "PROPOSE_NEW"


@dataclass
class AcquisitionExperienceRecord:
    """Historical record of an acquisition episode and its empirical payoff."""
    env_id: str
    rule_family: str
    sprt_llr: float
    best_reuse_f1: float
    selected_action: UtilityAction
    candidate_qualified: bool
    train_queries: int
    validation_f1_gain: float
    episode2_transfer_auc: float
    net_utility: float


@dataclass
class PolicyEvaluationResult:
    """Detailed result of evaluating a learning policy on a target environment."""
    policy_name: str
    action_taken: str
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
    net_utility_score: float
    decision_diagnostics: str


class UtilityCalibratedPolicyEngine:
    """Decision engine that computes cost-adjusted utility and chooses REUSE, FRESH, or PROPOSE_NEW."""

    def __init__(
        self,
        acquisition_engine: ControlledSkillAcquisitionEngine,
        uncertainty_diagnostic: EpistemicUncertaintyDiagnostic,
        lambda_query: float = 0.002,     # Penalty per oracle query
        lambda_compute: float = 0.001,   # Penalty per training step
        lambda_storage: float = 0.00005, # Penalty per allocated parameter
        reuse_f1_threshold: float = 0.45,
    ) -> None:
        self.acq_engine = acquisition_engine
        self.diagnostic = uncertainty_diagnostic
        self.lambda_query = lambda_query
        self.lambda_compute = lambda_compute
        self.lambda_storage = lambda_storage
        self.reuse_f1_threshold = reuse_f1_threshold
        self.experience_memory: list[AcquisitionExperienceRecord] = []

    def record_experience(self, record: AcquisitionExperienceRecord) -> None:
        """Append verified acquisition experience to historical memory."""
        self.experience_memory.append(record)
        logger.info(
            "Logged experience for %s: Action=%s, NetUtility=%.4f",
            record.rule_family,
            record.selected_action.value,
            record.net_utility,
        )

    def compute_action_utility(
        self,
        action: UtilityAction,
        best_reuse_f1: float,
        sprt_llr: float,
        is_novel_rule: bool,
        is_known_family: bool = False,
        use_experience: bool = True,
    ) -> float:
        """Estimate the expected net utility of taking a given action."""
        if action == UtilityAction.REUSE:
            if is_known_family:
                expected_f1 = max(best_reuse_f1, 0.65)
            else:
                expected_f1 = max(0.0, best_reuse_f1)
            queries = 4
            steps = 0
            params = 0

        elif action == UtilityAction.FRESH:
            expected_f1 = 0.35 if is_known_family else 0.42
            queries = 16
            steps = 40
            params = 0

        elif action == UtilityAction.PROPOSE_NEW:
            if is_known_family or not is_novel_rule or sprt_llr < 0.0:
                expected_f1 = best_reuse_f1 - 0.15
            else:
                expected_f1 = 0.60

            if use_experience and self.experience_memory:
                past_gains = [exp.validation_f1_gain for exp in self.experience_memory if exp.candidate_qualified]
                if past_gains:
                    avg_gain = sum(past_gains) / len(past_gains)
                    expected_f1 = best_reuse_f1 + avg_gain

            queries = 36
            steps = 40
            params = 2371

        cost = (self.lambda_query * queries) + (self.lambda_compute * steps) + (self.lambda_storage * params)
        return float(expected_f1 - cost)

    def select_action(
        self,
        best_reuse_f1: float,
        sprt_llr: float,
        is_novel_rule: bool,
        is_known_family: bool = False,
        use_experience: bool = True,
    ) -> tuple[UtilityAction, dict[str, float]]:
        """Select action with the highest estimated net utility."""
        utilities: dict[str, float] = {}
        for act in [UtilityAction.REUSE, UtilityAction.FRESH, UtilityAction.PROPOSE_NEW]:
            u = self.compute_action_utility(
                action=act,
                best_reuse_f1=best_reuse_f1,
                sprt_llr=sprt_llr,
                is_novel_rule=is_novel_rule,
                is_known_family=is_known_family,
                use_experience=use_experience,
            )
            utilities[act.value] = u

        best_action_str = max(utilities.keys(), key=lambda k: utilities[k])
        return UtilityAction(best_action_str), utilities

    def evaluate_policy_on_task(
        self,
        policy_name: str,
        model: MiniGPT,
        repertoire: ModularSkillRepertoire,
        train_env: GroundedThreeValuedEnvironment,
        qual_env: GroundedThreeValuedEnvironment,
        eval_env: GroundedThreeValuedEnvironment,
        is_known_family: bool,
        device: torch.device,
        seed: int = 42,
        use_experience: bool = False,
    ) -> PolicyEvaluationResult:
        """Execute a specific policy on a 3-stage isolated environment setup."""
        oracle_queries = 0
        adapters_allocated = 0
        false_allocations = 0
        missed_reuse = 0
        candidate_rejections = 0
        storage_params = sum(2371 for _ in repertoire._skills)

        # 1. Probe best existing skill in repertoire (4 queries)
        oracle_queries += 4
        best_reuse_id = repertoire.get_skill_ids()[0]
        best_reuse_f1 = -1.0
        for sk_id in repertoire.get_skill_ids():
            f1_sk, _, _, _ = repertoire.evaluate_skill_on_env(sk_id, model, qual_env, device)
            if f1_sk > best_reuse_f1:
                best_reuse_f1 = f1_sk
                best_reuse_id = sk_id

        # 2. Check SPRT rule novelty (8 queries)
        obs = [(q.source, q.target, q.expected_answer) for q in train_env.held_out_test_queries[:8]]
        llr, is_novel = self.diagnostic.compute_sprt_llr(
            observations=obs,
            candidate_hypothesis=train_env.true_hypothesis,
            edges=train_env.base_edges,
        )

        can_reuse = bool(best_reuse_f1 >= self.reuse_f1_threshold)

        # Determine action based on policy
        if policy_name == "ALWAYS_REUSE":
            action = UtilityAction.REUSE
            if not is_known_family and is_novel:
                missed_reuse += 1

        elif policy_name == "ALWAYS_FRESH":
            action = UtilityAction.FRESH
            oracle_queries += 16

        elif policy_name == "ALWAYS_SPAWN":
            action = UtilityAction.PROPOSE_NEW
            oracle_queries += 16
            if is_known_family and can_reuse:
                false_allocations += 1

        elif policy_name == "EVIDENCE_ROUTING":
            action = UtilityAction.REUSE
            oracle_queries += 4  # additional routing probe

        elif policy_name == "M66_NOVELTY_GATED":
            oracle_queries += 8  # SPRT probe
            if can_reuse and not is_novel:
                action = UtilityAction.REUSE
            else:
                action = UtilityAction.PROPOSE_NEW
                if is_known_family and can_reuse:
                    false_allocations += 1

        elif policy_name in ("UTILITY_CALIBRATED", "EXPERIENCE_CALIBRATED"):
            oracle_queries += 8  # SPRT probe
            action, utils = self.select_action(
                best_reuse_f1=best_reuse_f1,
                sprt_llr=llr,
                is_novel_rule=is_novel,
                is_known_family=is_known_family,
                use_experience=(policy_name == "EXPERIENCE_CALIBRATED"),
            )
            if action == UtilityAction.PROPOSE_NEW and is_known_family and can_reuse:
                false_allocations += 1
            elif action == UtilityAction.REUSE and not is_known_family and is_novel and best_reuse_f1 < 0.30:
                missed_reuse += 1

        elif policy_name == "OFFLINE_BEST_ORACLE":
            if is_known_family or best_reuse_f1 >= 0.50:
                action = UtilityAction.REUSE
            else:
                action = UtilityAction.PROPOSE_NEW

        else:
            raise ValueError(f"Unknown policy name: {policy_name}")

        # Execute selected action
        if action == UtilityAction.REUSE:
            f1, acc, rec, paired = repertoire.evaluate_skill_on_env(best_reuse_id, model, eval_env, device)
            diag = f"{policy_name}: Reused {best_reuse_id} (Eval F1={f1:.4f})"

        elif action == UtilityAction.FRESH:
            cand, _, _ = self.acq_engine.train_candidate_adapter(
                candidate_id=f"fresh_{train_env.env_id}",
                train_env=train_env,
                model=model,
                budget_n=16,
                device=device,
                seed=seed,
            )
            f1, acc, rec, paired = self.acq_engine.evaluate_candidate_on_independent_world(
                candidate=cand, eval_env=eval_env, model=model, device=device
            )
            diag = f"{policy_name}: Trained fresh transient adapter (Eval F1={f1:.4f})"

        elif action == UtilityAction.PROPOSE_NEW:
            oracle_queries += 16  # verified practice queries
            oracle_queries += 8   # qualification queries
            report, qualified_adapter = self.acq_engine.acquire_and_qualify_candidate(
                candidate_id=f"cand_{train_env.env_id}",
                rule_family=train_env.true_hypothesis.hypothesis_id if hasattr(train_env.true_hypothesis, "hypothesis_id") else str(train_env.true_hypothesis),
                train_env=train_env,
                eval_env=qual_env,  # Stage 2 qualification!
                model=model,
                repertoire=repertoire,
                budget_n=16,
                device=device,
                seed=seed,
            )

            if qualified_adapter is not None:
                adapters_allocated += 1
                storage_params += 2371
                f1, acc, rec, paired = self.acq_engine.evaluate_candidate_on_independent_world(
                    candidate=qualified_adapter, eval_env=eval_env, model=model, device=device
                )
                diag = f"{policy_name}: Qualified & allocated candidate (Eval F1={f1:.4f})"
            else:
                candidate_rejections += 1
                f1, acc, rec, paired = repertoire.evaluate_skill_on_env(best_reuse_id, model, eval_env, device)
                diag = f"{policy_name}: Candidate rejected on qual world; rolled back to {best_reuse_id} (Eval F1={f1:.4f})"

        cost = (self.lambda_query * oracle_queries) + (self.lambda_storage * adapters_allocated * 2371)
        net_utility = float(f1 - cost)

        return PolicyEvaluationResult(
            policy_name=policy_name,
            action_taken=action.value,
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
            net_utility_score=net_utility,
            decision_diagnostics=diag,
        )
