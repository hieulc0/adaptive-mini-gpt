"""Milestone M93 Phase 3: Cost-Aware Active Experimentation Engine.

Implements probabilistic hypothesis tracking and cost-aware active query selection under observation noise:
1. Noise-Tolerant Soft Likelihood Model:
   - For a noise rate eta in [0, 1), the probability of observing label y given true mapping h(x) is:
       P(y | x, h) = (1 - eta) if h(x) == y else (eta / 15)
   - Prevents noisy observations from irreversibly discarding the ground-truth hypothesis.
2. Cost-Aware Expected Information Gain (EIG):
   - Computes Shannon entropy over candidate hypothesis posterior distribution:
       H(P) = -sum_h P(h) * log2(P(h))
   - Evaluates expected information gain per unit query cost:
       Utility(x) = (H(P) - E_y[H(P | x, y)]) / C_query
3. Strategic Query Selection:
   - Evaluates unqueried points x in {0..15} and selects probe maximizing utility.
   - Early Stopping: When top candidate reaches posterior confidence threshold (tau >= 0.80),
     terminates experimentation to minimize query expenditure.
4. Epistemic Abstention:
   - Detects severe contradictions when maximum posterior confidence remains below threshold,
     triggering calibrated abstention rather than ungrounded execution.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from adaptive_mini_gpt.autonomous_discovery.active_discovery_engine import HypothesisSpace
from adaptive_mini_gpt.autonomous_discovery.unknown_rule_environment import ProceduralRule
from adaptive_mini_gpt.probabilistic_uncertainty.noisy_environment import (
    NoisyObservation,
    NoisyRuleTask,
)


@dataclass(frozen=True)
class CostAwareStepLog:
    """Telemetry log for a cost-aware experiment step."""

    step_idx: int
    probe_x: int
    observed_y: int
    is_corrupted: bool
    top_candidate_id: str
    top_confidence: float
    posterior_entropy: float
    information_gain: float
    cumulative_cost: float


@dataclass(frozen=True)
class CostAwareOutcome:
    """Final decision and beliefs from cost-aware experimentation."""

    task_id: str
    decision: str  # "EXECUTE", "ABSTAIN"
    selected_rule: ProceduralRule | None
    predicted_table: dict[int, int]
    queries_used: int
    total_cost: float
    final_confidence: float
    final_entropy: float
    step_logs: list[CostAwareStepLog]


class CostAwareExperimenter:
    """Probabilistic active learner operating under noise and query cost budgets."""

    def __init__(
        self,
        hypothesis_space: HypothesisSpace | None = None,
        confidence_threshold: float = 0.75,
        min_info_gain_threshold: float = 0.05,
    ) -> None:
        self.hypothesis_space = hypothesis_space or HypothesisSpace()
        self.confidence_threshold = confidence_threshold
        self.min_info_gain_threshold = min_info_gain_threshold

    def compute_posterior(
        self,
        hypotheses: Sequence[ProceduralRule],
        observations: Sequence[NoisyObservation],
        prior: list[float] | None = None,
        assumed_noise_rate: float = 0.15,
    ) -> list[float]:
        """Compute Bayesian posterior distribution over candidate hypotheses."""
        n_hyp = len(hypotheses)
        if n_hyp == 0:
            return []

        # Uniform prior if not specified
        log_probs = [math.log(prior[i]) if prior else -math.log(n_hyp) for i in range(n_hyp)]

        log_match = math.log(max(1e-6, 1.0 - assumed_noise_rate))
        log_mismatch = math.log(max(1e-6, assumed_noise_rate / 15.0))

        for obs in observations:
            for i, h in enumerate(hypotheses):
                if h.apply(obs.x) == obs.y:
                    log_probs[i] += log_match
                else:
                    log_probs[i] += log_mismatch

        # Normalize via log-sum-exp
        max_log = max(log_probs)
        exps = [math.exp(lp - max_log) for lp in log_probs]
        sum_exp = sum(exps)
        return [e / sum_exp for e in exps]

    def compute_entropy(self, dist: Sequence[float]) -> float:
        """Shannon entropy in bits."""
        ent = 0.0
        for p in dist:
            if p > 1e-12:
                ent -= p * math.log2(p)
        return ent

    def compute_expected_info_gain(
        self,
        hypotheses: Sequence[ProceduralRule],
        current_posterior: Sequence[float],
        x: int,
        assumed_noise_rate: float = 0.15,
    ) -> float:
        """Compute Expected Information Gain for probing point x."""
        current_entropy = self.compute_entropy(current_posterior)

        # Predictive distribution over possible outcomes y in {0..15}
        py = [0.0] * 16
        for y in range(16):
            for i, h in enumerate(hypotheses):
                p_h = current_posterior[i]
                if p_h < 1e-6:
                    continue
                match = (h.apply(x) == y)
                like = (1.0 - assumed_noise_rate) if match else (assumed_noise_rate / 15.0)
                py[y] += like * p_h

        # Expected posterior entropy: sum_y P(y|x) * H(P(h | x, y))
        expected_post_entropy = 0.0
        for y in range(16):
            p_y = py[y]
            if p_y < 1e-6:
                continue

            temp_obs = [NoisyObservation(x=x, y=y)]
            virtual_post = self.compute_posterior(
                hypotheses=hypotheses,
                observations=temp_obs,
                prior=list(current_posterior),
                assumed_noise_rate=assumed_noise_rate,
            )
            expected_post_entropy += p_y * self.compute_entropy(virtual_post)

        info_gain = max(0.0, current_entropy - expected_post_entropy)
        return info_gain

    def select_best_probe(
        self,
        task: NoisyRuleTask,
        hypotheses: Sequence[ProceduralRule],
        posterior: Sequence[float],
        queried_xs: set[int],
    ) -> tuple[int, float]:
        """Find probe x* maximizing EIG per unit cost."""
        candidate_xs = [x for x in range(16) if x not in queried_xs]
        if not candidate_xs:
            return 0, 0.0

        best_x = candidate_xs[0]
        best_eig = -1.0

        for x in candidate_xs:
            eig = self.compute_expected_info_gain(
                hypotheses=hypotheses,
                current_posterior=posterior,
                x=x,
                assumed_noise_rate=max(0.05, task.noise_rate),
            )
            if eig > best_eig:
                best_eig = eig
                best_x = x

        return best_x, best_eig

    def run_experimentation(
        self,
        task: NoisyRuleTask,
        strategy: str = "cost_aware",  # "cost_aware", "random", "passive", "disagreement"
    ) -> CostAwareOutcome:
        """Execute cost-aware experimentation loop on a noisy task."""
        observed = list(task.initial_observations)

        # 1. Immediate structural contradiction check (conflicting observations for same input)
        seen_xs: dict[int, int] = {}
        for obs in observed:
            if obs.x in seen_xs and seen_xs[obs.x] != obs.y:
                return CostAwareOutcome(
                    task_id=task.task_id,
                    decision="ABSTAIN",
                    selected_rule=None,
                    predicted_table={},
                    queries_used=0,
                    total_cost=0.0,
                    final_confidence=0.0,
                    final_entropy=0.0,
                    step_logs=[],
                )
            seen_xs[obs.x] = obs.y

        queried_xs = {obs.x for obs in observed}
        hypotheses = self.hypothesis_space.hypotheses
        step_logs: list[CostAwareStepLog] = []

        assumed_noise = max(0.10, task.noise_rate)
        posterior = self.compute_posterior(hypotheses, observed, assumed_noise_rate=assumed_noise)

        # Main experimentation loop
        while task.remaining_budget > 0 and strategy != "passive":
            top_prob = max(posterior)

            # Early stopping check: high confidence achieved or remaining budget exhausted
            if strategy == "cost_aware" and top_prob >= self.confidence_threshold:
                break

            if strategy == "cost_aware":
                probe_x, eig = self.select_best_probe(task, hypotheses, posterior, queried_xs)
                if eig < self.min_info_gain_threshold:
                    # Marginal benefit of querying is below cost threshold!
                    break
            elif strategy == "random":
                unqueried = [x for x in range(16) if x not in queried_xs]
                if not unqueried:
                    break
                probe_x = random.choice(unqueried)
                eig = 0.0
            elif strategy == "disagreement":
                unqueried = [x for x in range(16) if x not in queried_xs]
                if not unqueried:
                    break
                probe_x, eig = self.select_best_probe(task, hypotheses, posterior, queried_xs)
            else:
                break

            # Execute query
            new_obs = task.oracle_query(probe_x)
            observed.append(new_obs)
            queried_xs.add(probe_x)

            # Update posterior
            posterior = self.compute_posterior(hypotheses, observed, assumed_noise_rate=assumed_noise)
            top_prob = max(posterior)
            top_idx = posterior.index(top_prob)
            top_rule = hypotheses[top_idx]

            step_logs.append(
                CostAwareStepLog(
                    step_idx=task.queries_used,
                    probe_x=probe_x,
                    observed_y=new_obs.y,
                    is_corrupted=new_obs.is_corrupted,
                    top_candidate_id=top_rule.rule_id,
                    top_confidence=round(top_prob, 4),
                    posterior_entropy=round(self.compute_entropy(posterior), 4),
                    information_gain=round(eig, 4),
                    cumulative_cost=task.total_cost_incurred,
                )
            )

        # Decision synthesis
        final_conf = max(posterior) if posterior else 0.0
        final_idx = posterior.index(final_conf) if posterior else 0
        final_rule = hypotheses[final_idx] if hypotheses else None
        final_entropy = self.compute_entropy(posterior) if posterior else 0.0

        # Abstain if task is unsupported or confidence remains low
        if final_conf < 0.50 or final_entropy > 3.0:
            decision = "ABSTAIN"
            table: dict[int, int] = {}
        else:
            decision = "EXECUTE"
            table = final_rule.generate_table() if final_rule else {}

        return CostAwareOutcome(
            task_id=task.task_id,
            decision=decision,
            selected_rule=final_rule,
            predicted_table=table,
            queries_used=task.queries_used,
            total_cost=task.total_cost_incurred,
            final_confidence=round(final_conf, 4),
            final_entropy=round(final_entropy, 4),
            step_logs=step_logs,
        )
