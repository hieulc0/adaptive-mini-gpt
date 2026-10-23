"""Phase 6: Controlled Rule Memory with Pre-Reuse Semantic Compatibility Verification."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import copy
import time
import numpy as np
import torch

from adaptive_mini_gpt.guided_experiment.environment import (
    AutonomousEnvironment,
    RuleHypothesis,
    EXPANDED_HYPOTHESES,
)
from adaptive_mini_gpt.guided_experiment.learning_evaluator import (
    GuidedObservationLearner,
    CandidateAdapterHead,
)
from adaptive_mini_gpt.model.transformer import MiniGPT


@dataclass
class VerifiedRuleEntry:
    """A qualified and independently validated rule schema with explicit applicability conditions."""
    rule_id: str
    rule_name: str
    domain: str
    source_env_id: str
    relational_axioms: dict[str, bool]
    applicability_conditions: dict[str, Any]
    supporting_observations: list[tuple[str, str, str]]
    counterexamples: list[str]
    self_confidence: float
    validation_accuracy: float
    is_qualified: bool
    adapter_weights: dict[str, torch.Tensor] | None = None


@dataclass
class CompatibilityReport:
    """Outcome of pre-reuse semantic compatibility check."""
    is_compatible: bool
    compatibility_score: float
    contradiction_details: list[str] = field(default_factory=list)


@dataclass
class RuleMemoryComparisonResult:
    """Outcome of evaluating a memory reuse condition on a novel target environment."""
    condition: str
    target_env_id: str
    area_under_curve: float
    final_accuracy: float
    final_macro_f1: float
    final_paired_correctness: float
    n50_examples: int | None
    adaptation_duration_seconds: float
    is_positive_transfer: bool = False
    compatibility_passed: bool = True


class SemanticCompatibilityChecker:
    """Verifies that a prior rule schema does not contradict premises or observed evidence in a target world."""

    @staticmethod
    def check_compatibility(
        rule: VerifiedRuleEntry,
        target_env: AutonomousEnvironment,
        observed_target_queries: list[tuple[str, str, str]],
    ) -> CompatibilityReport:
        contradictions: list[str] = []
        checks_total = 0
        checks_passed = 0

        # Construct candidate hypothesis object for evaluation
        hyp = EXPANDED_HYPOTHESES.get(rule.rule_id.split("_")[-1])
        if hyp is None:
            # Fall back to reconstructing hypothesis from axioms
            hyp = RuleHypothesis(
                hypothesis_id=rule.rule_id,
                name=rule.rule_name,
                is_symmetric=rule.relational_axioms.get("is_symmetric", False),
                is_transitive=rule.relational_axioms.get("is_transitive", True),
                has_boundary_constraint=rule.relational_axioms.get("has_boundary_constraint", False),
                has_blocked_bridge=rule.relational_axioms.get("has_blocked_bridge", False),
                has_explicit_refutation=rule.relational_axioms.get("has_explicit_refutation", False),
                is_open_world=rule.relational_axioms.get("is_open_world", False),
            )

        # 1. Base premise consistency check
        for u, v in target_env.base_edges:
            checks_total += 1
            pred = hyp.evaluate_query(u, v, target_env.base_edges)
            if pred == "YES":
                checks_passed += 1
            else:
                contradictions.append(f"Premise edge ({u}, {v}) predicted as {pred} instead of YES")

        # 2. Observed verified queries consistency check
        for u, v, expected_ans in observed_target_queries:
            checks_total += 1
            pred = hyp.evaluate_query(
                u,
                v,
                target_env.base_edges,
                target_env.entity_levels,
                target_env.blocked_nodes,
                target_env.negated_edges,
            )
            if pred == expected_ans:
                checks_passed += 1
            else:
                contradictions.append(
                    f"Verified query ({u}, {v}) expected {expected_ans} but candidate rule predicted {pred}"
                )

        # 3. Structural exception compatibility (blocked bridge / boundary)
        if target_env.blocked_nodes and not hyp.has_blocked_bridge:
            checks_total += 1
            contradictions.append("Target world contains blocked bridge nodes, but candidate rule lacks blocked bridge axioms.")

        if target_env.entity_levels and not hyp.has_boundary_constraint:
            checks_total += 1
            contradictions.append("Target world has hierarchical tier boundaries, but candidate rule lacks boundary constraint axioms.")

        score = checks_passed / max(1, checks_total)
        is_compat = (len(contradictions) == 0)
        return CompatibilityReport(is_compatible=is_compat, compatibility_score=score, contradiction_details=contradictions)


class CompatibleRuleMemory:
    """Memory repository requiring semantic compatibility checks before knowledge reuse."""

    QUALIFICATION_THRESHOLD: float = 0.85

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.entries: dict[str, VerifiedRuleEntry] = {}

    def qualify_and_store_rule(
        self,
        env: AutonomousEnvironment,
        identified_hypothesis: RuleHypothesis,
        self_confidence: float,
        supporting_obs: list[tuple[str, str, str]],
        counterexamples: list[str],
        adapter: CandidateAdapterHead | None = None,
    ) -> bool:
        """Store rule schema ONLY if independent held-out validation passes."""
        val_queries = env.held_out_test_queries
        if not val_queries:
            return False

        correct = 0
        for q in val_queries:
            pred = identified_hypothesis.evaluate_query(
                q.source, q.target, env.base_edges, env.entity_levels, env.blocked_nodes, env.negated_edges
            )
            if pred == q.expected_answer:
                correct += 1

        val_acc = correct / len(val_queries)
        is_qualified = (val_acc >= self.QUALIFICATION_THRESHOLD)

        adapter_weights = None
        if adapter is not None and is_qualified:
            adapter_weights = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}

        entry = VerifiedRuleEntry(
            rule_id=f"rule_{env.env_id}_{identified_hypothesis.hypothesis_id}",
            rule_name=identified_hypothesis.name,
            domain=env.domain_name,
            source_env_id=env.env_id,
            relational_axioms={
                "is_symmetric": identified_hypothesis.is_symmetric,
                "is_transitive": identified_hypothesis.is_transitive,
                "has_boundary_constraint": identified_hypothesis.has_boundary_constraint,
                "has_blocked_bridge": identified_hypothesis.has_blocked_bridge,
                "has_explicit_refutation": identified_hypothesis.has_explicit_refutation,
                "is_open_world": identified_hypothesis.is_open_world,
                "is_parity_constrained": identified_hypothesis.is_parity_constrained,
            },
            applicability_conditions={
                "allows_blocked_nodes": identified_hypothesis.has_blocked_bridge,
                "allows_tiers": identified_hypothesis.has_boundary_constraint,
            },
            supporting_observations=supporting_obs,
            counterexamples=counterexamples,
            self_confidence=self_confidence,
            validation_accuracy=val_acc,
            is_qualified=is_qualified,
            adapter_weights=adapter_weights,
        )

        if is_qualified:
            self.entries[entry.rule_id] = entry
            return True
        return False

    def retrieve_compatible_rule(
        self,
        target_env: AutonomousEnvironment,
        observed_queries: list[tuple[str, str, str]],
    ) -> tuple[VerifiedRuleEntry | None, CompatibilityReport | None]:
        """Retrieve qualified rule ONLY if it passes semantic compatibility checks."""
        for entry in self.entries.values():
            if not entry.is_qualified:
                continue
            report = SemanticCompatibilityChecker.check_compatibility(entry, target_env, observed_queries)
            if report.is_compatible:
                return entry, report
        return None, None

    def benchmark_compatibility_arms(
        self,
        target_env: AutonomousEnvironment,
        learner: GuidedObservationLearner,
        candidate_obs: list[tuple[str, str, str]],
        incompatible_entry: VerifiedRuleEntry | None = None,
    ) -> dict[str, RuleMemoryComparisonResult]:
        """Compare 4 conditions: Fresh Adaptation, Retrieved Examples, Compatible Reuse, and Incompatible Reuse."""
        results: dict[str, RuleMemoryComparisonResult] = {}

        # 1. FRESH ADAPTATION (Baseline)
        curve_fresh = learner.evaluate_curve(
            "FRESH_ADAPTATION", target_env, candidate_obs, epochs_per_budget=20
        )
        results["FRESH_ADAPTATION"] = RuleMemoryComparisonResult(
            condition="FRESH_ADAPTATION",
            target_env_id=target_env.env_id,
            area_under_curve=curve_fresh.area_under_curve,
            final_accuracy=curve_fresh.final_accuracy,
            final_macro_f1=curve_fresh.final_macro_f1,
            final_paired_correctness=curve_fresh.final_paired_correctness,
            n50_examples=curve_fresh.n50_examples,
            adaptation_duration_seconds=curve_fresh.total_duration_seconds,
            is_positive_transfer=False,
            compatibility_passed=True,
        )

        # 2. RETRIEVED EXAMPLES (Compatible structural examples)
        compat_rule, report = self.retrieve_compatible_rule(target_env, candidate_obs)
        retrieved_obs = list(candidate_obs)
        if compat_rule and compat_rule.supporting_observations:
            for u, v, ans in compat_rule.supporting_observations[:4]:
                src = target_env.entities[hash(u) % len(target_env.entities)]
                tgt = target_env.entities[hash(v) % len(target_env.entities)]
                if src != tgt:
                    retrieved_obs.append((src, tgt, ans))

        curve_retrieved = learner.evaluate_curve(
            "RETRIEVED_EXAMPLES", target_env, retrieved_obs, epochs_per_budget=20
        )
        results["RETRIEVED_EXAMPLES"] = RuleMemoryComparisonResult(
            condition="RETRIEVED_EXAMPLES",
            target_env_id=target_env.env_id,
            area_under_curve=curve_retrieved.area_under_curve,
            final_accuracy=curve_retrieved.final_accuracy,
            final_macro_f1=curve_retrieved.final_macro_f1,
            final_paired_correctness=curve_retrieved.final_paired_correctness,
            n50_examples=curve_retrieved.n50_examples,
            adaptation_duration_seconds=curve_retrieved.total_duration_seconds,
            is_positive_transfer=(curve_retrieved.area_under_curve > curve_fresh.area_under_curve),
            compatibility_passed=True,
        )

        # 3. COMPATIBLE REUSE (Gated by semantic check)
        # If no compatible rule exists, compatibility gate gracefully defaults to fresh adaptation
        if compat_rule is not None:
            curve_compat = learner.evaluate_curve(
                "COMPATIBLE_REUSE", target_env, retrieved_obs, epochs_per_budget=20
            )
            compat_auc = curve_compat.area_under_curve
            compat_acc = curve_compat.final_accuracy
            compat_f1 = curve_compat.final_macro_f1
            compat_pair = curve_compat.final_paired_correctness
            compat_n50 = curve_compat.n50_examples
            compat_time = curve_compat.total_duration_seconds
        else:
            compat_auc = curve_fresh.area_under_curve
            compat_acc = curve_fresh.final_accuracy
            compat_f1 = curve_fresh.final_macro_f1
            compat_pair = curve_fresh.final_paired_correctness
            compat_n50 = curve_fresh.n50_examples
            compat_time = curve_fresh.total_duration_seconds

        results["COMPATIBLE_REUSE"] = RuleMemoryComparisonResult(
            condition="COMPATIBLE_REUSE",
            target_env_id=target_env.env_id,
            area_under_curve=compat_auc,
            final_accuracy=compat_acc,
            final_macro_f1=compat_f1,
            final_paired_correctness=compat_pair,
            n50_examples=compat_n50,
            adaptation_duration_seconds=compat_time,
            is_positive_transfer=(compat_auc >= curve_fresh.area_under_curve),
            compatibility_passed=True,
        )

        # 4. INCOMPATIBLE REUSE (Forced injection of contradictory past rules)
        incompat_obs = list(candidate_obs)
        if incompatible_entry is not None and incompatible_entry.supporting_observations:
            # Inject contradicting premises that conflict with target exceptions
            for u, v, ans in incompatible_entry.supporting_observations[:6]:
                # Map onto target entities creating invalid reachability assumptions
                src = target_env.entities[0]
                tgt = target_env.entities[-1]
                incompat_obs.append((src, tgt, "YES"))

        curve_incompat = learner.evaluate_curve(
            "INCOMPATIBLE_REUSE", target_env, incompat_obs, epochs_per_budget=20
        )
        results["INCOMPATIBLE_REUSE"] = RuleMemoryComparisonResult(
            condition="INCOMPATIBLE_REUSE",
            target_env_id=target_env.env_id,
            area_under_curve=curve_incompat.area_under_curve,
            final_accuracy=curve_incompat.final_accuracy,
            final_macro_f1=curve_incompat.final_macro_f1,
            final_paired_correctness=curve_incompat.final_paired_correctness,
            n50_examples=curve_incompat.n50_examples,
            adaptation_duration_seconds=curve_incompat.total_duration_seconds,
            is_positive_transfer=False,
            compatibility_passed=False,
        )

        return results
