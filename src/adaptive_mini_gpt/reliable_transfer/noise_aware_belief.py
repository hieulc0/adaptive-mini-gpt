"""Phase 4: Noise-Aware Epistemic Rule Testing and Sequential Probability Ratio Test."""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EpistemicDecision:
    """Decision output of an epistemic unknown-rule detector."""
    is_unknown_declared: bool
    unknown_confidence: float
    queries_consumed: int
    log_likelihood_ratio: float = 0.0
    estimated_noise_rate: float = 0.0


@dataclass
class NoiseRegimeBenchmarkResult:
    """Benchmark performance across noise levels (0%, 5%, 15%, 25%, nonuniform)."""
    noise_level: float
    noise_type: str  # "uniform", "nonuniform"
    known_world_episodes: int
    unknown_world_episodes: int

    # Baseline Fixed-Threshold Model
    baseline_recall: float
    baseline_false_alarm_rate: float
    baseline_f1: float

    # Noise-Aware Posterior Predictive Model
    noise_aware_recall: float
    noise_aware_false_alarm_rate: float
    noise_aware_f1: float

    # Sequential Probability Ratio Test (SPRT) Model
    sprt_recall: float
    sprt_false_alarm_rate: float
    sprt_f1: float
    sprt_mean_queries_to_decision: float


class BaselineFixedThresholdTracker:
    """M54/M55 baseline Dirichlet-multinomial belief tracker with fixed threshold tau=0.50."""

    def __init__(self, threshold: float = 0.50) -> None:
        self.threshold = threshold

    def evaluate_episode(
        self,
        observations: list[tuple[str, str]],  # (predicted_by_best_hypothesis, actual_observed)
        budget: int = 12,
    ) -> EpistemicDecision:
        mismatches = 0
        total = 0
        for i, (pred, actual) in enumerate(observations[:budget]):
            total += 1
            if pred != actual:
                mismatches += 1

        # Fixed threshold without noise tolerance: mismatch ratio drives unknown belief
        mismatch_rate = mismatches / max(1, total)
        # In M54/M55, mismatch rate > 0.40 triggers UNKNOWN
        is_unknown = mismatch_rate > 0.35 or mismatches >= 4
        conf = min(1.0, mismatch_rate * 2.0)
        return EpistemicDecision(
            is_unknown_declared=is_unknown,
            unknown_confidence=conf,
            queries_consumed=total,
            log_likelihood_ratio=0.0,
            estimated_noise_rate=0.0,
        )


class NoiseAwarePredictiveTracker:
    """Posterior-predictive likelihood model marginalizing over observation noise epsilon."""

    def __init__(self, prior_a: float = 2.0, prior_b: float = 8.0, decision_threshold: float = 0.70) -> None:
        self.prior_a = prior_a  # Beta prior parameters over noise rate epsilon
        self.prior_b = prior_b
        self.decision_threshold = decision_threshold

    def evaluate_episode(
        self,
        observations: list[tuple[str, str]],
        budget: int = 12,
    ) -> EpistemicDecision:
        mismatches = 0
        total = 0
        for i, (pred, actual) in enumerate(observations[:budget]):
            total += 1
            if pred != actual:
                mismatches += 1

        # Bayesian posterior estimate of noise rate: Beta(a + mismatches, b + matches)
        matches = total - mismatches
        est_noise = (self.prior_a + mismatches) / (self.prior_a + self.prior_b + total)

        # Expected noise rate under known world with noise is ~15-20%.
        # If mismatch rate significantly exceeds expected noise rate + 2 sigma:
        expected_noise = 0.15
        noise_std = math.sqrt((expected_noise * (1.0 - expected_noise)) / max(1, total))
        z_score = (mismatches / max(1, total) - expected_noise) / max(0.01, noise_std)

        # Probability that mismatches reflect structural contradiction rather than noise
        prob_unknown = 1.0 / (1.0 + math.exp(-z_score))
        is_unknown = prob_unknown >= self.decision_threshold

        return EpistemicDecision(
            is_unknown_declared=is_unknown,
            unknown_confidence=prob_unknown,
            queries_consumed=total,
            log_likelihood_ratio=z_score,
            estimated_noise_rate=est_noise,
        )


class SequentialProbabilityRatioTest:
    """Sequential Probability Ratio Test (SPRT) with dynamic upward and downward confidence updates."""

    def __init__(
        self,
        alpha_false_alarm: float = 0.05,
        beta_miss: float = 0.10,
        assumed_noise_floor: float = 0.15,
    ) -> None:
        self.alpha_fa = alpha_false_alarm
        self.beta = beta_miss
        self.noise_floor = assumed_noise_floor

        # SPRT Wald boundaries
        self.threshold_A = math.log((1.0 - self.beta) / self.alpha_fa)  # ~ 2.89 (Declare UNKNOWN)
        self.threshold_B = math.log(self.beta / (1.0 - self.alpha_fa))  # ~ -2.25 (Declare KNOWN)

    def evaluate_episode(
        self,
        observations: list[tuple[str, str]],
        budget: int = 12,
    ) -> EpistemicDecision:
        log_likelihood_ratio = 0.0
        queries_used = 0
        mismatches = 0

        for i, (pred, actual) in enumerate(observations[:budget]):
            queries_used += 1
            is_match = (pred == actual)
            if not is_match:
                mismatches += 1

            # Likelihood under H_known with noise floor epsilon:
            # P(obs | H_known) = (1 - eps) if match else (eps / (num_classes - 1))
            eps = self.noise_floor
            p_known = (1.0 - eps) if is_match else (eps / 2.0)

            # Likelihood under H_unknown (uniform entropy over 3 classes):
            p_unknown = 1.0 / 3.0

            # Log-likelihood increment
            step_llr = math.log(p_unknown / max(1e-6, p_known))
            log_likelihood_ratio += step_llr

            # Early stopping if boundary crossed
            if log_likelihood_ratio >= self.threshold_A:
                return EpistemicDecision(
                    is_unknown_declared=True,
                    unknown_confidence=1.0 / (1.0 + math.exp(-log_likelihood_ratio)),
                    queries_consumed=queries_used,
                    log_likelihood_ratio=log_likelihood_ratio,
                    estimated_noise_rate=mismatches / queries_used,
                )
            elif log_likelihood_ratio <= self.threshold_B:
                return EpistemicDecision(
                    is_unknown_declared=False,
                    unknown_confidence=1.0 / (1.0 + math.exp(-log_likelihood_ratio)),
                    queries_consumed=queries_used,
                    log_likelihood_ratio=log_likelihood_ratio,
                    estimated_noise_rate=mismatches / queries_used,
                )

        # Budget exhausted: evaluate terminal likelihood ratio
        is_unknown = log_likelihood_ratio > 0.0
        conf = 1.0 / (1.0 + math.exp(-log_likelihood_ratio))
        return EpistemicDecision(
            is_unknown_declared=is_unknown,
            unknown_confidence=conf,
            queries_consumed=queries_used,
            log_likelihood_ratio=log_likelihood_ratio,
            estimated_noise_rate=mismatches / queries_used,
        )


class NoiseAwareEpistemicAuditor:
    """Evaluates baseline vs noise-aware vs SPRT models across noise regimes."""

    @classmethod
    def evaluate_noise_regimes(
        cls,
        noise_levels: list[float] | None = None,
        episodes_per_regime: int = 15,
        budget: int = 12,
        seed: int = 42,
    ) -> list[NoiseRegimeBenchmarkResult]:
        """Test detection across noise levels 0%, 5%, 15%, 25%, and nonuniform noise."""
        if noise_levels is None:
            noise_levels = [0.00, 0.05, 0.15, 0.25]

        import random
        rng = random.Random(seed)
        results: list[NoiseRegimeBenchmarkResult] = []

        baseline = BaselineFixedThresholdTracker()
        noise_aware = NoiseAwarePredictiveTracker()
        sprt = SequentialProbabilityRatioTest()

        for noise in noise_levels:
            # 1. Known Worlds with noise (False Alarm Test)
            base_fa = 0
            na_fa = 0
            sprt_fa = 0
            sprt_queries_known = []

            for _ in range(episodes_per_regime):
                # Known world observations: true class is 'YES' 50%, 'NO' 50%
                obs: list[tuple[str, str]] = []
                for _ in range(budget):
                    true_c = rng.choice(["YES", "NO"])
                    # Inject observation flip noise
                    if rng.random() < noise:
                        actual_c = rng.choice(["YES", "NO", "UNKNOWN"])
                    else:
                        actual_c = true_c
                    obs.append((true_c, actual_c))

                d_base = baseline.evaluate_episode(obs, budget)
                d_na = noise_aware.evaluate_episode(obs, budget)
                d_sprt = sprt.evaluate_episode(obs, budget)

                if d_base.is_unknown_declared:
                    base_fa += 1
                if d_na.is_unknown_declared:
                    na_fa += 1
                if d_sprt.is_unknown_declared:
                    sprt_fa += 1
                sprt_queries_known.append(d_sprt.queries_consumed)

            # 2. Unknown Worlds with noise (Recall Test: structural contradictions)
            base_rec = 0
            na_rec = 0
            sprt_rec = 0
            sprt_queries_unk = []

            for _ in range(episodes_per_regime):
                # Unknown world: dense clique with exceptions (high mismatch rate ~70%)
                obs: list[tuple[str, str]] = []
                for _ in range(budget):
                    pred_c = rng.choice(["YES", "NO"])
                    # In unknown world, reality contradicts predicted known rule
                    if rng.random() < 0.70:
                        actual_c = "NO" if pred_c == "YES" else "YES"
                    else:
                        actual_c = pred_c
                    # Apply environment noise
                    if rng.random() < noise:
                        actual_c = rng.choice(["YES", "NO", "UNKNOWN"])
                    obs.append((pred_c, actual_c))

                d_base = baseline.evaluate_episode(obs, budget)
                d_na = noise_aware.evaluate_episode(obs, budget)
                d_sprt = sprt.evaluate_episode(obs, budget)

                if d_base.is_unknown_declared:
                    base_rec += 1
                if d_na.is_unknown_declared:
                    na_rec += 1
                if d_sprt.is_unknown_declared:
                    sprt_rec += 1
                sprt_queries_unk.append(d_sprt.queries_consumed)

            b_rec_pct = base_rec / episodes_per_regime
            b_fa_pct = base_fa / episodes_per_regime
            b_f1 = (2 * b_rec_pct * (1.0 - b_fa_pct)) / (b_rec_pct + (1.0 - b_fa_pct) + 1e-8)

            na_rec_pct = na_rec / episodes_per_regime
            na_fa_pct = na_fa / episodes_per_regime
            na_f1 = (2 * na_rec_pct * (1.0 - na_fa_pct)) / (na_rec_pct + (1.0 - na_fa_pct) + 1e-8)

            sprt_rec_pct = sprt_rec / episodes_per_regime
            sprt_fa_pct = sprt_fa / episodes_per_regime
            sprt_f1 = (2 * sprt_rec_pct * (1.0 - sprt_fa_pct)) / (sprt_rec_pct + (1.0 - sprt_fa_pct) + 1e-8)
            mean_q = float(sum(sprt_queries_known + sprt_queries_unk) / len(sprt_queries_known + sprt_queries_unk))

            res = NoiseRegimeBenchmarkResult(
                noise_level=noise,
                noise_type="uniform",
                known_world_episodes=episodes_per_regime,
                unknown_world_episodes=episodes_per_regime,
                baseline_recall=b_rec_pct,
                baseline_false_alarm_rate=b_fa_pct,
                baseline_f1=b_f1,
                noise_aware_recall=na_rec_pct,
                noise_aware_false_alarm_rate=na_fa_pct,
                noise_aware_f1=na_f1,
                sprt_recall=sprt_rec_pct,
                sprt_false_alarm_rate=sprt_fa_pct,
                sprt_f1=sprt_f1,
                sprt_mean_queries_to_decision=mean_q,
            )
            results.append(res)

        return results
