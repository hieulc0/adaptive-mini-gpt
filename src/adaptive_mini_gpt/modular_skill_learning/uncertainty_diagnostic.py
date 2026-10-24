"""Phase 3: Formal Distinction Between Missing Facts (RELATION_UNKNOWN) and Unfamiliar Rules (RULE_UNKNOWN)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from adaptive_mini_gpt.guided_experiment.environment import AutonomousEnvironment, RuleHypothesis
from adaptive_mini_gpt.three_valued_reasoning.semantics import GroundedThreeValuedEnvironment

logger = logging.getLogger(__name__)


class UncertaintyCategory(str, Enum):
    """Rigorous categorization of epistemic and structural uncertainty."""
    FACTUAL_CERTAINTY = "FACTUAL_CERTAINTY"            # Evidence proves YES or NO
    RELATION_UNKNOWN = "RELATION_UNKNOWN"              # Missing/unobserved factual evidence (relation-level)
    RULE_KNOWN = "RULE_KNOWN"                          # Observations conform to an existing hypothesis
    RULE_UNKNOWN = "RULE_UNKNOWN"                      # Observations contradict all known hypotheses (rule-level)


@dataclass
class EpistemicSeparationTrace:
    """Detailed audit trace confirming separation of factual uncertainty from structural novelty."""
    env_id: str
    condition: str
    total_queries: int
    relation_unknown_count: int
    rule_unknown_detected: bool
    sprt_log_likelihood_ratio: float
    adapter_allocation_proposed: bool
    allocation_decision_valid: bool
    diagnostics: str


class EpistemicUncertaintyDiagnostic:
    """Evaluates and enforces separation between relation-level missing evidence and rule-level novelty."""

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.05,
        p0: float = 0.10,  # Expected error rate under true hypothesis
        p1: float = 0.50,  # Expected error rate under alien/unknown rule
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.p0 = p0
        self.p1 = p1

        # Wald SPRT decision boundaries
        self.lower_bound = np.log(beta / (1.0 - alpha))
        self.upper_bound = np.log((1.0 - beta) / alpha)

    def compute_sprt_llr(
        self,
        observations: list[tuple[str, str, str]],  # (source, target, answer)
        candidate_hypothesis: RuleHypothesis,
        edges: list[tuple[str, str]],
        entity_levels: dict[str, int] | None = None,
        negated_edges: set[tuple[str, str]] | None = None,
    ) -> tuple[float, bool]:
        """Compute SPRT log-likelihood ratio comparing hypothesis fit against novelty."""
        if not observations:
            return 0.0, False

        llr = 0.0
        for src, tgt, obs_ans in observations:
            pred_ans = candidate_hypothesis.evaluate_query(
                source=src,
                target=tgt,
                edges=edges,
                entity_levels=entity_levels,
                negated_edges=negated_edges,
            )
            # Match is consistent with H0; mismatch supports H1 (novel rule)
            if obs_ans == pred_ans:
                llr += np.log((1.0 - self.p1) / (1.0 - self.p0))
            else:
                llr += np.log(self.p1 / self.p0)

        is_novel = (llr >= self.upper_bound)
        return float(llr), bool(is_novel)

    def evaluate_environment_condition(
        self,
        env: GroundedThreeValuedEnvironment,
        known_hypotheses: list[RuleHypothesis],
        condition_name: str,
    ) -> EpistemicSeparationTrace:
        """Run epistemic diagnostic ensuring RELATION_UNKNOWN never triggers adapter allocation."""
        queries = env.held_out_test_queries
        rel_unk_count = sum(1 for q in queries if q.expected_answer == "UNKNOWN")

        # Build sample observations from known edge facts and explicit queries
        obs = []
        for q in queries:
            obs.append((q.source, q.target, q.expected_answer))

        # Check SPRT across all known hypotheses (find closest fitting hypothesis)
        best_llr = float("inf")
        novelty_votes = []
        for hyp in known_hypotheses:
            llr, is_nov = self.compute_sprt_llr(
                observations=obs,
                candidate_hypothesis=hyp,
                edges=env.base_edges,
                entity_levels=getattr(env.base_env, "entity_levels", None),
                negated_edges=getattr(env.base_env, "negated_edges", None),
            )
            if llr < best_llr:
                best_llr = llr
            novelty_votes.append(is_nov)

        # Rule is unknown iff ALL known hypotheses reject the observations
        rule_unknown = all(novelty_votes) if novelty_votes else True

        # STRICT INVARIANT:
        # Adapter allocation is proposed ONLY when rule_unknown is True!
        # RELATION_UNKNOWN count must NEVER trigger adapter allocation on its own!
        allocation_proposed = rule_unknown

        # Validation check:
        # In known-incomplete worlds, rel_unk_count may be high, but rule_unknown MUST be False!
        if condition_name == "known_incomplete" and rule_unknown:
            decision_valid = False
            diag = "VIOLATION: Incomplete observations falsely triggered RULE_UNKNOWN."
        elif condition_name == "unknown_rule" and not rule_unknown:
            decision_valid = False
            diag = "FAILURE: Novel rule family was not detected by SPRT."
        else:
            decision_valid = True
            diag = (
                f"PASSED: Condition='{condition_name}'. UNK_queries={rel_unk_count}. "
                f"RuleNovelty={rule_unknown} (LLR={best_llr:.2f}). Allocation={allocation_proposed}."
            )

        return EpistemicSeparationTrace(
            env_id=env.env_id,
            condition=condition_name,
            total_queries=len(queries),
            relation_unknown_count=rel_unk_count,
            rule_unknown_detected=rule_unknown,
            sprt_log_likelihood_ratio=best_llr,
            adapter_allocation_proposed=allocation_proposed,
            allocation_decision_valid=decision_valid,
            diagnostics=diag,
        )
