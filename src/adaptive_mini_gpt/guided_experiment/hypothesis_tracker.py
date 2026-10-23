"""Guided Hypothesis Tracker and Shannon Entropy / Expected Information Gain Engine."""
from __future__ import annotations

import math
from typing import Any
from adaptive_mini_gpt.guided_experiment.environment import (
    EXPANDED_HYPOTHESES,
    RuleHypothesis,
    UNKNOWN_RULE,
)
from adaptive_mini_gpt.guided_experiment.structural_probe_generator import StructuralProbe


class GuidedHypothesisTracker:
    """Maintains Bayesian belief distribution over candidate hypotheses and calculates EIG."""

    def __init__(
        self,
        base_edges: list[tuple[str, str]],
        candidate_hypotheses: dict[str, RuleHypothesis] | None = None,
        entity_levels: dict[str, int] | None = None,
        blocked_nodes: set[str] | None = None,
        negated_edges: set[tuple[str, str]] | None = None,
        allow_unknown_rule: bool = True,
        observation_noise: float = 0.02,
    ) -> None:
        self.base_edges = list(base_edges)
        self.entity_levels = entity_levels or {}
        self.blocked_nodes = blocked_nodes or set()
        self.negated_edges = negated_edges or set()
        self.observation_noise = observation_noise
        self.allow_unknown_rule = allow_unknown_rule

        # Known candidate hypotheses (H8)
        self.hypotheses: dict[str, RuleHypothesis] = dict(candidate_hypotheses or EXPANDED_HYPOTHESES)
        self.h_ids: list[str] = list(self.hypotheses.keys())

        # Initialize prior
        num_hyp = len(self.h_ids)
        if self.allow_unknown_rule:
            unk_prior = 0.04
            rem = (1.0 - unk_prior) / num_hyp
            self.beliefs: dict[str, float] = {h_id: rem for h_id in self.h_ids}
            self.beliefs[UNKNOWN_RULE.hypothesis_id] = unk_prior
        else:
            uniform = 1.0 / num_hyp
            self.beliefs = {h_id: uniform for h_id in self.h_ids}

        self.history: list[dict[str, Any]] = []

    def compute_entropy(self) -> float:
        """Calculate Shannon entropy H(B) in bits."""
        ent = 0.0
        for p in self.beliefs.values():
            if p > 1e-12:
                ent -= p * math.log2(p)
        return max(0.0, ent)

    def evaluate_hypothesis_prediction(self, h_id: str, source: str, target: str) -> str:
        """Evaluate predicted answer under hypothesis."""
        if h_id == UNKNOWN_RULE.hypothesis_id:
            return "UNKNOWN"
        h = self.hypotheses[h_id]
        return h.evaluate_query(
            source=source,
            target=target,
            edges=self.base_edges,
            entity_levels=self.entity_levels,
            blocked_nodes=self.blocked_nodes,
            negated_edges=self.negated_edges,
        )

    def get_outcome_likelihood(self, h_id: str, source: str, target: str, outcome: str) -> float:
        """Likelihood P(outcome | h, query) incorporating observation noise."""
        if h_id == UNKNOWN_RULE.hypothesis_id:
            # UNKNOWN_RULE assumes all 3 outcomes are plausible under open distribution
            return 1.0 / 3.0

        pred = self.evaluate_hypothesis_prediction(h_id, source, target)
        if outcome == pred:
            return 1.0 - self.observation_noise
        else:
            return self.observation_noise / 2.0

    def compute_expected_information_gain(self, probe: StructuralProbe | tuple[str, str]) -> float:
        """Calculate expected information gain (EIG) in bits for a candidate probe."""
        if isinstance(probe, StructuralProbe):
            source, target = probe.source, probe.target
        else:
            source, target = probe

        current_entropy = self.compute_entropy()
        if current_entropy < 1e-6:
            return 0.0

        outcomes = ["YES", "NO", "UNKNOWN"]
        p_outcomes: dict[str, float] = {}

        # Marginal P(y | q, B) = sum_h P(h) P(y | h, q)
        for y in outcomes:
            p_y = sum(
                p_h * self.get_outcome_likelihood(h_id, source, target, y)
                for h_id, p_h in self.beliefs.items()
            )
            p_outcomes[y] = p_y

        expected_posterior_entropy = 0.0
        for y, p_y in p_outcomes.items():
            if p_y > 1e-9:
                # Compute normalized posterior B_{q, y}(h)
                post_beliefs: dict[str, float] = {}
                for h_id, p_h in self.beliefs.items():
                    lik = self.get_outcome_likelihood(h_id, source, target, y)
                    post_beliefs[h_id] = (p_h * lik) / p_y

                # Compute posterior entropy
                post_ent = 0.0
                for p_post in post_beliefs.values():
                    if p_post > 1e-12:
                        post_ent -= p_post * math.log2(p_post)
                expected_posterior_entropy += p_y * max(0.0, post_ent)

        eig = current_entropy - expected_posterior_entropy
        return max(0.0, eig)

    def update(self, source: str, target: str, answer: str) -> float:
        """Update posterior beliefs given observed answer, returning actual information gain in bits."""
        prior_entropy = self.compute_entropy()

        # Compute unnormalized posterior
        unnorm: dict[str, float] = {}
        for h_id, p_h in self.beliefs.items():
            lik = self.get_outcome_likelihood(h_id, source, target, answer)
            unnorm[h_id] = p_h * lik

        total = sum(unnorm.values())
        if total > 1e-12:
            self.beliefs = {h_id: v / total for h_id, v in unnorm.items()}
        else:
            # Degenerate: revert to uniform or all unknown
            if self.allow_unknown_rule:
                self.beliefs = {h_id: 0.0 for h_id in self.h_ids}
                self.beliefs[UNKNOWN_RULE.hypothesis_id] = 1.0
            else:
                self.beliefs = {h_id: 1.0 / len(self.h_ids) for h_id in self.h_ids}

        posterior_entropy = self.compute_entropy()
        actual_ig = max(0.0, prior_entropy - posterior_entropy)

        self.history.append({
            "source": source,
            "target": target,
            "answer": answer,
            "prior_entropy": prior_entropy,
            "posterior_entropy": posterior_entropy,
            "information_gain": actual_ig,
            "top_hypothesis": self.get_top_hypothesis(),
            "unknown_rule_belief": self.beliefs.get(UNKNOWN_RULE.hypothesis_id, 0.0),
        })

        return actual_ig

    def get_top_hypothesis(self) -> tuple[str, float]:
        """Return (hypothesis_id, posterior_probability) for highest-belief hypothesis."""
        best_h = max(self.beliefs.items(), key=lambda kv: kv[1])
        return best_h[0], float(best_h[1])

    def is_unknown_rule_detected(self, threshold: float = 0.50) -> bool:
        """Returns True if the posterior belief that the rule is outside known hypotheses exceeds threshold."""
        return self.beliefs.get(UNKNOWN_RULE.hypothesis_id, 0.0) >= threshold
