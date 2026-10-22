"""Phase 4: Hypothesis Formation, Revision, and Unknown-Rule Detection for M50."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math
import numpy as np

from adaptive_mini_gpt.experiment_design.expanded_hypothesis import (
    EXPANDED_HYPOTHESES,
    RuleHypothesis,
    UNKNOWN_RULE,
)


@dataclass
class RevisionStep:
    """Record of a single observation update step."""
    query_id: str
    source: str
    target: str
    observed_answer: str
    prior_entropy: float
    posterior_entropy: float
    information_gain: float
    top_hypothesis_before: str
    top_hypothesis_after: str
    hypothesis_swapped: bool
    contradictions_detected: list[str] = field(default_factory=list)
    top_confidence: float = 0.0


@dataclass
class HypothesisRevisionReport:
    """Summary of hypothesis tracking, revision, and contradiction metrics."""
    initial_entropy: float = 3.0
    final_entropy: float = 0.0
    cumulative_information_gain: float = 0.0
    final_top_hypothesis: str = ""
    final_top_confidence: float = 0.0
    total_revisions_swaps: int = 0
    total_contradictions_detected: int = 0
    disproven_rules_abandoned: list[str] = field(default_factory=list)
    is_unknown_rule_triggered: bool = False
    false_confidence_count: int = 0
    history: list[RevisionStep] = field(default_factory=list)


class HypothesisRevisionTracker:
    """Maintains, revises, and evaluates candidate rule hypotheses under sequential evidence."""

    def __init__(
        self,
        base_edges: list[tuple[str, str]],
        entity_levels: dict[str, int] | None = None,
        blocked_nodes: set[str] | None = None,
        negated_edges: set[tuple[str, str]] | None = None,
        hypothesis_set: dict[str, RuleHypothesis] | None = None,
    ) -> None:
        self.base_edges = base_edges
        self.entity_levels = entity_levels or {}
        self.blocked_nodes = blocked_nodes or set()
        self.negated_edges = negated_edges or set()
        self.hypotheses = hypothesis_set or copy_hypotheses(EXPANDED_HYPOTHESES)

        # Uniform prior over all hypotheses in the candidate set
        n_hyp = len(self.hypotheses)
        self.prior_prob: float = 1.0 / n_hyp
        self.beliefs: dict[str, float] = {h_id: self.prior_prob for h_id in self.hypotheses}
        self.unknown_rule_belief: float = 0.0

        self.history: list[RevisionStep] = []
        self.disproven_rules: set[str] = set()
        self.false_confidence_count: int = 0
        self.revision_swap_count: int = 0

    def get_entropy(self) -> float:
        """Calculate current Shannon entropy of hypothesis belief distribution in bits."""
        probs = [p for p in self.beliefs.values() if p > 1e-9]
        if self.unknown_rule_belief > 1e-9:
            probs.append(self.unknown_rule_belief)
        if not probs:
            return 0.0
        return float(-sum(p * math.log2(p) for p in probs))

    def get_top_hypothesis(self) -> tuple[RuleHypothesis, float]:
        """Return the highest-confidence hypothesis and its posterior probability."""
        if self.unknown_rule_belief > 0.95:
            return UNKNOWN_RULE, self.unknown_rule_belief

        best_h_id = max(self.beliefs, key=lambda k: self.beliefs[k])
        best_conf = self.beliefs[best_h_id]

        if best_conf < self.unknown_rule_belief:
            return UNKNOWN_RULE, self.unknown_rule_belief

        return self.hypotheses.get(best_h_id, UNKNOWN_RULE), best_conf

    def predict_outcome_prob(self, source: str, target: str) -> dict[str, float]:
        """Compute marginal probability P(outcome | source, target) under current beliefs."""
        probs = {"YES": 0.0, "NO": 0.0, "UNKNOWN": 0.0}
        for h_id, h in self.hypotheses.items():
            b = self.beliefs[h_id]
            if b <= 0.0:
                continue
            pred = h.evaluate_query(
                source, target, self.base_edges, self.entity_levels, self.blocked_nodes, self.negated_edges
            )
            probs[pred] += b

        # Normalize
        total = sum(probs.values())
        if total > 0.0:
            for k in probs:
                probs[k] /= total
        else:
            probs = {"YES": 1/3, "NO": 1/3, "UNKNOWN": 1/3}
        return probs

    def compute_expected_information_gain(self, source: str, target: str) -> float:
        """Compute Expected Information Gain (EIG) for asking (source, relation, target)."""
        current_entropy = self.get_entropy()
        if current_entropy < 1e-6:
            return 0.0

        p_outcomes = self.predict_outcome_prob(source, target)
        expected_post_entropy = 0.0

        for outcome, p_o in p_outcomes.items():
            if p_o < 1e-6:
                continue

            # Simulate posterior
            sim_beliefs = {}
            for h_id, h in self.hypotheses.items():
                b = self.beliefs[h_id]
                if b <= 0.0:
                    sim_beliefs[h_id] = 0.0
                    continue
                pred = h.evaluate_query(
                    source, target, self.base_edges, self.entity_levels, self.blocked_nodes, self.negated_edges
                )
                lik = 0.98 if pred == outcome else 0.01
                sim_beliefs[h_id] = b * lik

            sim_total = sum(sim_beliefs.values())
            if sim_total > 0.0:
                post_probs = [val / sim_total for val in sim_beliefs.values() if val > 1e-9]
                h_post = float(-sum(p * math.log2(p) for p in post_probs))
            else:
                h_post = 0.0

            expected_post_entropy += p_o * h_post

        eig = max(0.0, current_entropy - expected_post_entropy)
        return float(eig)

    def update_belief(
        self,
        source: str,
        target: str,
        observed_answer: str,
        true_hypothesis_id: str | None = None,
        query_id: str = "q_active",
    ) -> float:
        """Bayesian belief update with strict contradiction and unknown-rule detection."""
        prior_entropy = self.get_entropy()
        top_before, conf_before = self.get_top_hypothesis()

        contradictions: list[str] = []
        new_unnorm: dict[str, float] = {}

        for h_id, h in self.hypotheses.items():
            curr_b = self.beliefs[h_id]
            if curr_b <= 0.0:
                new_unnorm[h_id] = 0.0
                continue

            pred = h.evaluate_query(
                source, target, self.base_edges, self.entity_levels, self.blocked_nodes, self.negated_edges
            )

            if pred != observed_answer:
                # Contradiction: strict rule falsification
                contradictions.append(h_id)
                self.disproven_rules.add(h_id)
                new_unnorm[h_id] = 0.0
            else:
                # Consistent with observation
                new_unnorm[h_id] = curr_b * 1.0

        total_weight = sum(new_unnorm.values())

        if total_weight < 1e-9:
            # ALL candidate hypotheses disproven! Trigger UNKNOWN_RULE
            self.beliefs = {h_id: 0.0 for h_id in self.hypotheses}
            self.unknown_rule_belief = 1.0
        else:
            self.unknown_rule_belief = 0.0
            for h_id in self.hypotheses:
                self.beliefs[h_id] = new_unnorm[h_id] / total_weight

        post_entropy = self.get_entropy()
        info_gain = max(0.0, prior_entropy - post_entropy)

        top_after, conf_after = self.get_top_hypothesis()
        hypothesis_swapped = (top_before.hypothesis_id != top_after.hypothesis_id)
        if hypothesis_swapped:
            self.revision_swap_count += 1

        # Check false confidence: high confidence (>90%) in a wrong hypothesis
        if true_hypothesis_id and top_after.hypothesis_id != true_hypothesis_id and conf_after > 0.90:
            self.false_confidence_count += 1

        step = RevisionStep(
            query_id=query_id,
            source=source,
            target=target,
            observed_answer=observed_answer,
            prior_entropy=prior_entropy,
            posterior_entropy=post_entropy,
            information_gain=info_gain,
            top_hypothesis_before=top_before.hypothesis_id,
            top_hypothesis_after=top_after.hypothesis_id,
            hypothesis_swapped=hypothesis_swapped,
            contradictions_detected=contradictions,
            top_confidence=conf_after,
        )
        self.history.append(step)
        return info_gain

    def generate_report(self) -> HypothesisRevisionReport:
        """Summarize hypothesis revision trajectory."""
        top_h, conf = self.get_top_hypothesis()
        cum_gain = sum(s.information_gain for s in self.history)
        total_contra = sum(len(s.contradictions_detected) for s in self.history)

        return HypothesisRevisionReport(
            initial_entropy=self.history[0].prior_entropy if self.history else self.get_entropy(),
            final_entropy=self.get_entropy(),
            cumulative_information_gain=cum_gain,
            final_top_hypothesis=top_h.hypothesis_id,
            final_top_confidence=conf,
            total_revisions_swaps=self.revision_swap_count,
            total_contradictions_detected=total_contra,
            disproven_rules_abandoned=list(self.disproven_rules),
            is_unknown_rule_triggered=(self.unknown_rule_belief > 0.95),
            false_confidence_count=self.false_confidence_count,
            history=self.history,
        )


def copy_hypotheses(h_dict: dict[str, RuleHypothesis]) -> dict[str, RuleHypothesis]:
    return {k: v for k, v in h_dict.items()}
