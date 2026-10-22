"""Phase 3: Bayesian Hypothesis Tracking and Expected Information Gain (EIG)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.active_learning.environment import (
    CandidateQuery,
    HYPOTHESES,
    RuleHypothesis,
)


@dataclass
class TrackerStep:
    query_id: str
    query_text: str
    observed_answer: str
    prior_entropy: float
    posterior_entropy: float
    information_gain: float
    top_hypothesis: str
    top_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "observed_answer": self.observed_answer,
            "prior_entropy": round(self.prior_entropy, 4),
            "posterior_entropy": round(self.posterior_entropy, 4),
            "information_gain": round(self.information_gain, 4),
            "top_hypothesis": self.top_hypothesis,
            "top_confidence": round(self.top_confidence, 4),
        }


class BayesianHypothesisTracker:
    """Maintains and updates probabilistic beliefs over candidate rule hypotheses."""

    def __init__(self, hypotheses: dict[str, RuleHypothesis] | None = None) -> None:
        self.hypotheses = hypotheses or HYPOTHESES
        self.k = len(self.hypotheses)
        self.beliefs: dict[str, float] = {h_name: 1.0 / self.k for h_name in self.hypotheses}
        self.history: list[TrackerStep] = []

    def get_entropy(self) -> float:
        """Computes Shannon entropy in bits: H(H) = - sum(p * log2(p))."""
        ent = 0.0
        for p in self.beliefs.values():
            if p > 1e-12:
                ent -= p * math.log2(p)
        return ent

    def predict_for_hypothesis(self, q: CandidateQuery, h: RuleHypothesis) -> str:
        """Determines the expected answer for query q under hypothesis h."""
        if q.query_type == "REVERSE_TEST":
            return "YES" if h.is_symmetric else "NO"
        elif q.query_type in ("TWO_HOP_BRIDGE", "THREE_HOP_CHAIN"):
            return "YES" if h.is_transitive else "UNKNOWN"
        elif q.query_type == "DIRECT_EDGE":
            return "YES"
        elif q.query_type == "REFUTATION_CHECK":
            return "NO"
        elif q.query_type == "DISCONNECTED_CHECK":
            return "UNKNOWN"
        else:
            return q.expected_answer

    def compute_expected_information_gain(self, q: CandidateQuery) -> float:
        """Computes Expected Information Gain (EIG) in bits for query q:

        EIG(q) = H(H) - sum_y P(y|q) * H(H|q, y)
        """
        prior_ent = self.get_entropy()
        ans_marginal: dict[str, float] = {"YES": 0.0, "NO": 0.0, "UNKNOWN": 0.0}

        for h_name, h in self.hypotheses.items():
            prior_p = self.beliefs[h_name]
            pred_y = self.predict_for_hypothesis(q, h)
            ans_marginal[pred_y] += prior_p

        expected_post_ent = 0.0
        for y, p_y in ans_marginal.items():
            if p_y <= 1e-12:
                continue
            post_ent = 0.0
            for h_name, h in self.hypotheses.items():
                prior_p = self.beliefs[h_name]
                pred_y = self.predict_for_hypothesis(q, h)
                likelihood = 1.0 if pred_y == y else 0.0
                post_p = (likelihood * prior_p) / p_y
                if post_p > 1e-12:
                    post_ent -= post_p * math.log2(post_p)
            expected_post_ent += p_y * post_ent

        return max(0.0, prior_ent - expected_post_ent)

    def update_belief(self, q: CandidateQuery, observed_answer: str) -> float:
        """Performs Bayesian update after observing oracle outcome, returning information gain."""
        prior_ent = self.get_entropy()
        unnorm: dict[str, float] = {}

        for h_name, h in self.hypotheses.items():
            pred = self.predict_for_hypothesis(q, h)
            # Soft likelihood with small smoothing epsilon (0.02) to prevent hard zeroing
            likelihood = 0.98 if pred == observed_answer else 0.02
            unnorm[h_name] = self.beliefs[h_name] * likelihood

        total = sum(unnorm.values())
        if total > 1e-12:
            self.beliefs = {k: v / total for k, v in unnorm.items()}

        post_ent = self.get_entropy()
        info_gain = max(0.0, prior_ent - post_ent)

        top_h, top_conf = self.get_top_hypothesis()
        step = TrackerStep(
            query_id=q.query_id,
            query_text=q.question_text,
            observed_answer=observed_answer,
            prior_entropy=prior_ent,
            posterior_entropy=post_ent,
            information_gain=info_gain,
            top_hypothesis=top_h.name,
            top_confidence=top_conf,
        )
        self.history.append(step)
        return info_gain

    def get_top_hypothesis(self) -> tuple[RuleHypothesis, float]:
        """Returns the most probable hypothesis and its posterior confidence."""
        best_name = max(self.beliefs.keys(), key=lambda k: self.beliefs[k])
        return self.hypotheses[best_name], self.beliefs[best_name]

    def is_correctly_identified(self, true_h: RuleHypothesis, threshold: float = 0.70) -> bool:
        """Checks if the true hypothesis has posterior above threshold."""
        return (self.beliefs.get(true_h.name, 0.0) >= threshold) and (
            self.get_top_hypothesis()[0].name == true_h.name
        )
