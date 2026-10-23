"""Phase 6 & Epistemic Tracking: Normalized Bayesian Hypothesis Tracker and Out-of-Hypothesis Detector."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from adaptive_mini_gpt.guided_experiment.environment import (
    EXPANDED_HYPOTHESES,
    RuleHypothesis,
    UNKNOWN_RULE,
)
from adaptive_mini_gpt.guided_experiment.structural_probe_generator import StructuralProbe


@dataclass
class EpistemicState:
    """Snapshot of epistemic belief distribution at step t."""
    step: int
    beliefs: dict[str, float]
    entropy_bits: float
    top_hypothesis_id: str
    top_hypothesis_prob: float
    unknown_rule_prob: float
    cumulative_contradictions: dict[str, int]


class NormalizedEpistemicTracker:
    """Strictly normalized Bayesian belief tracker over candidate hypotheses and UNKNOWN_RULE.

    Key Scientific Properties:
    1. Proper Normalization: Sum of all beliefs (known hypotheses + UNKNOWN_RULE) = 1.0 at every step.
    2. Bidirectional Evidence Updating: Belief in UNKNOWN_RULE can freely increase or decrease
       depending on whether subsequent evidence contradicts or confirms known hypotheses.
       No hardcoded monotonic growth.
    3. Noise Robustness: Differentiates noisy observations on known rules from genuinely unmodeled rule structures.
    """

    CLASS_LABELS = ["YES", "NO", "UNKNOWN"]

    def __init__(
        self,
        base_edges: list[tuple[str, str]],
        candidate_hypotheses: dict[str, RuleHypothesis] | None = None,
        entity_levels: dict[str, int] | None = None,
        blocked_nodes: set[str] | None = None,
        negated_edges: set[tuple[str, str]] | None = None,
        unknown_rule_prior: float = 0.15,
        observation_noise: float = 0.03,
    ) -> None:
        self.base_edges = list(base_edges)
        self.entity_levels = entity_levels or {}
        self.blocked_nodes = blocked_nodes or set()
        self.negated_edges = negated_edges or set()
        self.observation_noise = observation_noise
        self.unknown_rule_prior = unknown_rule_prior

        # Known candidate hypotheses (H8)
        self.hypotheses: dict[str, RuleHypothesis] = dict(candidate_hypotheses or EXPANDED_HYPOTHESES)
        self.h_ids: list[str] = list(self.hypotheses.keys())

        # Initialize normalized prior distribution
        num_known = len(self.h_ids)
        rem_mass = 1.0 - self.unknown_rule_prior
        p_known = rem_mass / float(num_known)

        self.beliefs: dict[str, float] = {h_id: p_known for h_id in self.h_ids}
        self.beliefs[UNKNOWN_RULE.hypothesis_id] = self.unknown_rule_prior

        # Track empirical outcome counts for Dirichlet-multinomial predictive distribution of UNKNOWN_RULE
        # Prior pseudocounts: alpha_0 = 1.0 for each class (Laplace smoothing)
        self.outcome_counts: dict[str, float] = {label: 1.0 for label in self.CLASS_LABELS}

        # Track cumulative contradictions per hypothesis
        self.contradictions: dict[str, int] = {h_id: 0 for h_id in self.h_ids}
        self.contradictions[UNKNOWN_RULE.hypothesis_id] = 0

        self.history: list[EpistemicState] = []
        self._record_state(step=0)

    def _record_state(self, step: int) -> None:
        top_h, top_p = self.get_top_hypothesis()
        self.history.append(
            EpistemicState(
                step=step,
                beliefs=dict(self.beliefs),
                entropy_bits=self.compute_entropy(),
                top_hypothesis_id=top_h,
                top_hypothesis_prob=top_p,
                unknown_rule_prob=float(self.beliefs.get(UNKNOWN_RULE.hypothesis_id, 0.0)),
                cumulative_contradictions=dict(self.contradictions),
            )
        )

    def compute_entropy(self) -> float:
        """Calculate Shannon entropy H(B) in bits."""
        ent = 0.0
        for p in self.beliefs.values():
            if p > 1e-12:
                ent -= p * math.log2(p)
        return max(0.0, float(ent))

    def evaluate_hypothesis_prediction(self, h_id: str, source: str, target: str) -> str:
        """Evaluate deterministic prediction under hypothesis."""
        if h_id == UNKNOWN_RULE.hypothesis_id:
            # UNKNOWN_RULE does not make a rigid deterministic claim; open predictive distribution used in likelihood
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
        """Normalized outcome likelihood P(outcome | h, query)."""
        if h_id == UNKNOWN_RULE.hypothesis_id:
            # Dirichlet-multinomial predictive distribution over observed classes
            total_counts = sum(self.outcome_counts.values())
            return float(self.outcome_counts[outcome] / max(1e-6, total_counts))

        pred = self.evaluate_hypothesis_prediction(h_id, source, target)
        if outcome == pred:
            return 1.0 - self.observation_noise
        else:
            return self.observation_noise / 2.0

    def compute_expected_information_gain(self, probe: StructuralProbe | tuple[str, str]) -> float:
        """Compute expected Shannon information gain (EIG) in bits for a candidate probe."""
        if isinstance(probe, StructuralProbe):
            source, target = probe.source, probe.target
        else:
            source, target = probe

        current_entropy = self.compute_entropy()
        if current_entropy < 1e-6:
            return 0.0

        p_outcomes: dict[str, float] = {}
        for y in self.CLASS_LABELS:
            p_y = sum(
                p_h * self.get_outcome_likelihood(h_id, source, target, y)
                for h_id, p_h in self.beliefs.items()
            )
            p_outcomes[y] = p_y

        expected_post_entropy = 0.0
        for y, p_y in p_outcomes.items():
            if p_y > 1e-9:
                post_beliefs = {}
                for h_id, p_h in self.beliefs.items():
                    lik = self.get_outcome_likelihood(h_id, source, target, y)
                    post_beliefs[h_id] = (p_h * lik) / p_y

                post_ent = 0.0
                for p_post in post_beliefs.values():
                    if p_post > 1e-12:
                        post_ent -= p_post * math.log2(p_post)
                expected_post_entropy += p_y * max(0.0, post_ent)

        return max(0.0, float(current_entropy - expected_post_entropy))

    def update(self, source: str, target: str, answer: str) -> float:
        """Update Bayesian posterior beliefs given observed verified answer.

        Returns:
            Actual information gain in bits: Delta H = H_prior - H_post.
        """
        prior_entropy = self.compute_entropy()

        # Update empirical outcome distribution for UNKNOWN_RULE
        if answer in self.outcome_counts:
            self.outcome_counts[answer] += 1.0

        # Track contradictions
        for h_id in self.h_ids:
            pred = self.evaluate_hypothesis_prediction(h_id, source, target)
            if pred != answer:
                self.contradictions[h_id] += 1

        # Bayes update: P(h | D) proportional to P(h) * P(y | h, q)
        unnorm = {}
        for h_id, p_h in self.beliefs.items():
            lik = self.get_outcome_likelihood(h_id, source, target, answer)
            unnorm[h_id] = p_h * lik

        total = sum(unnorm.values())
        if total > 1e-12:
            self.beliefs = {h_id: float(v / total) for h_id, v in unnorm.items()}
        else:
            # Fallback to pure UNKNOWN_RULE if all mass vanishes
            self.beliefs = {h_id: 0.0 for h_id in self.h_ids}
            self.beliefs[UNKNOWN_RULE.hypothesis_id] = 1.0

        # Update history
        step = len(self.history)
        self._record_state(step=step)

        post_entropy = self.compute_entropy()
        actual_ig = max(0.0, float(prior_entropy - post_entropy))
        return actual_ig

    def get_top_hypothesis(self) -> tuple[str, float]:
        """Return (hypothesis_id, posterior_probability) for highest-belief hypothesis."""
        best_h = max(self.beliefs.items(), key=lambda kv: kv[1])
        return best_h[0], float(best_h[1])

    def is_unknown_rule_detected(self, threshold: float = 0.50) -> bool:
        """Returns True if normalized posterior belief in UNKNOWN_RULE exceeds threshold."""
        return float(self.beliefs.get(UNKNOWN_RULE.hypothesis_id, 0.0)) >= threshold
