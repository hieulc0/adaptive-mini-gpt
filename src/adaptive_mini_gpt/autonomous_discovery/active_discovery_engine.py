"""Milestone M92 Phase 2: Autonomous Hypothesis & Active Experimentation Engine.

Implements the active scientific discovery cycle:
1. Hypothesis Generation:
   - Maintains a structured space of candidate mathematical transformations (polynomials,
     modular scalings, bit rotations, and threshold piecewise logic).
2. Epistemic Consistency Evaluation:
   - Filters candidate hypotheses consistent with all observed input-output pairs.
   - Detects contradictions (unsupported tasks) -> triggers calibrated abstention.
3. Informative Query Selection (Active Experimentation):
   - Computes prediction disagreement / entropy across consistent candidates for all unqueried x in {0..15}.
   - Proposes minimally discriminative probe x* that maximizes information gain.
4. Bounded Active Discovery Loop:
   - Executes queries up to budget Q <= 3.
   - Resolves candidate ambiguity into a single validated rule or abstains when underdetermined.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence

from adaptive_mini_gpt.autonomous_discovery.unknown_rule_environment import (
    ProceduralRule,
    RuleObservation,
    UnknownRuleTask,
)


@dataclass(frozen=True)
class DiscoveryStepLog:
    """Record of an active experimentation step."""

    step_idx: int
    num_candidates_before: int
    selected_query_x: int
    oracle_response_y: int
    num_candidates_after: int
    disagreement_score: float


@dataclass(frozen=True)
class DiscoveryOutcome:
    """Final outcome of the autonomous discovery phase."""

    task_id: str
    decision: str  # "EXECUTE", "ABSTAIN", "AMBIGUOUS"
    is_supported: bool
    queries_used: int
    final_candidate_count: int
    selected_rule: ProceduralRule | None
    predicted_table: dict[int, int]
    step_logs: list[DiscoveryStepLog]
    confidence: float


class HypothesisSpace:
    """Space of mathematical procedural hypotheses over Z_16."""

    def __init__(self) -> None:
        self.hypotheses: list[ProceduralRule] = []
        self._build_standard_space()

    def _build_standard_space(self) -> None:
        """Construct comprehensive candidate hypotheses over Z_16."""
        # 1. Linear / Affine: y = (m*x + c) % 16
        for m in (1, 2, 3, 5, 7, 9, 11, 13, 15):
            for c in range(16):
                def make_linear(m_val: int, c_val: int) -> Callable[[int], int]:
                    return lambda x: (m_val * x + c_val) % 16

                self.hypotheses.append(
                    ProceduralRule(f"linear_m{m}_c{c}", "linear", make_linear(m, c))
                )

        # 2. Quadratic: y = (a*x^2 + b*x + c) % 16
        for a in (1, 2, 3, 5):
            for b in (0, 1, 2, 3):
                for c in range(16):
                    def make_quad(a_v: int, b_v: int, c_v: int) -> Callable[[int], int]:
                        return lambda x: (a_v * (x ** 2) + b_v * x + c_v) % 16

                    self.hypotheses.append(
                        ProceduralRule(f"quad_a{a}_b{b}_c{c}", "quadratic", make_quad(a, b, c))
                    )

        # 3. Piecewise / Threshold: if x < thresh: (m1*x + c1) % 16 else (m2*x + c2) % 16
        for thresh in (6, 8, 10):
            for m1, c1 in [(2, 1), (3, 2), (1, 4), (3, 2)]:
                for m2, c2 in [(15, 0), (1, 7), (2, 2), (5, 4)]:
                    def make_piece(t_v: int, m1_v: int, c1_v: int, m2_v: int, c2_v: int) -> Callable[[int], int]:
                        return lambda x: (m1_v * x + c1_v) % 16 if x < t_v else (m2_v * x + c2_v) % 16

                    self.hypotheses.append(
                        ProceduralRule(f"piece_t{thresh}_{m1}_{c1}_{m2}_{c2}", "piecewise", make_piece(thresh, m1, c1, m2, c2))
                    )

        # 4. Bitwise rotation and XOR
        for shift in (1, 2, 3):
            for mask in (0x3, 0x5, 0xA, 0xC):
                def make_bit(s_v: int, m_v: int) -> Callable[[int], int]:
                    return lambda x: (((x << s_v) | (x >> (4 - s_v))) & 0xF) ^ m_v

                self.hypotheses.append(
                    ProceduralRule(f"bit_s{shift}_m{mask}", "bit_perm", make_bit(shift, mask))
                )

        # 5. Cubic polynomials: y = (lead_a * x^3 + a*x + b) % 16
        for lead_a in (1, 2):
            for a in (0, 1, 3):
                for b in (0, 2, 5, 7):
                    def make_cubic(l_v: int, a_v: int, b_v: int) -> Callable[[int], int]:
                        return lambda x: (l_v * (x ** 3) + a_v * x + b_v) % 16

                    self.hypotheses.append(
                        ProceduralRule(f"cubic_l{lead_a}_a{a}_b{b}", "cubic", make_cubic(lead_a, a, b))
                    )


class ActiveDiscoveryEngine:
    """Autonomous hypothesis evaluation and discriminative experiment selector."""

    def __init__(self, hypothesis_space: HypothesisSpace | None = None) -> None:
        self.hypothesis_space = hypothesis_space or HypothesisSpace()

    def filter_consistent(
        self,
        hypotheses: Sequence[ProceduralRule],
        observations: Sequence[RuleObservation],
    ) -> list[ProceduralRule]:
        """Return subset of hypotheses consistent with all observed pairs."""
        consistent: list[ProceduralRule] = []
        for h in hypotheses:
            match = True
            for obs in observations:
                if h.apply(obs.x) != obs.y:
                    match = False
                    break
            if match:
                consistent.append(h)
        return consistent

    def compute_query_disagreement(
        self,
        candidates: Sequence[ProceduralRule],
        x: int,
    ) -> float:
        """Calculate prediction entropy / disagreement across candidate rules at probe x."""
        if not candidates:
            return 0.0
        counts: dict[int, int] = {}
        for c in candidates:
            y = c.apply(x)
            counts[y] = counts.get(y, 0) + 1

        total = len(candidates)
        # Shannon entropy: -sum(p * log2(p))
        entropy = 0.0
        for cnt in counts.values():
            p = cnt / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def select_informative_query(
        self,
        candidates: Sequence[ProceduralRule],
        observed_xs: set[int],
    ) -> tuple[int, float]:
        """Find unobserved probe point x* in {0..15} with maximum prediction entropy."""
        candidate_xs = [x for x in range(16) if x not in observed_xs]
        if not candidate_xs:
            return 0, 0.0

        best_x = candidate_xs[0]
        best_score = -1.0

        for x in candidate_xs:
            score = self.compute_query_disagreement(candidates, x)
            if score > best_score:
                best_score = score
                best_x = x

        return best_x, best_score

    def discover_rule(
        self,
        task: UnknownRuleTask,
        strategy: str = "active",  # "active", "random", "passive"
        confidence_threshold: float = 0.60,
    ) -> DiscoveryOutcome:
        """Run autonomous discovery loop on an unfamiliar task under query budget."""
        observed = list(task.initial_observations)
        observed_xs = {obs.x for obs in observed}
        logs: list[DiscoveryStepLog] = []

        # 1. Initial consistency check
        candidates = self.filter_consistent(self.hypothesis_space.hypotheses, observed)

        # Detect immediate contradiction
        if not candidates:
            return DiscoveryOutcome(
                task_id=task.task_id,
                decision="ABSTAIN",
                is_supported=False,
                queries_used=0,
                final_candidate_count=0,
                selected_rule=None,
                predicted_table={},
                step_logs=[],
                confidence=0.0,
            )

        # 2. Interactive experimentation loop
        while task.remaining_budget > 0 and len(candidates) > 1 and strategy != "passive":
            if strategy == "active":
                probe_x, score = self.select_informative_query(candidates, observed_xs)
                if score <= 0.0:
                    # Candidates agree on all remaining points!
                    break
            elif strategy == "random":
                avail = [x for x in range(16) if x not in observed_xs]
                if not avail:
                    break
                probe_x = random.choice(avail)
                score = self.compute_query_disagreement(candidates, probe_x)
            else:
                break

            # Execute query on environment
            new_obs = task.oracle_query(probe_x)
            observed.append(new_obs)
            observed_xs.add(probe_x)

            cands_before = len(candidates)
            candidates = self.filter_consistent(candidates, observed)
            cands_after = len(candidates)

            logs.append(
                DiscoveryStepLog(
                    step_idx=task.queries_used,
                    num_candidates_before=cands_before,
                    selected_query_x=probe_x,
                    oracle_response_y=new_obs.y,
                    num_candidates_after=cands_after,
                    disagreement_score=score,
                )
            )

            if not candidates:
                # Contradiction emerged during queries!
                return DiscoveryOutcome(
                    task_id=task.task_id,
                    decision="ABSTAIN",
                    is_supported=False,
                    queries_used=task.queries_used,
                    final_candidate_count=0,
                    selected_rule=None,
                    predicted_table={},
                    step_logs=logs,
                    confidence=0.0,
                )

        # 3. Decision synthesis
        if not candidates:
            return DiscoveryOutcome(
                task_id=task.task_id,
                decision="ABSTAIN",
                is_supported=False,
                queries_used=task.queries_used,
                final_candidate_count=0,
                selected_rule=None,
                predicted_table={},
                step_logs=logs,
                confidence=0.0,
            )

        # Check confidence (1.0 / count if multiple remain)
        confidence = 1.0 if len(candidates) == 1 else (1.0 / len(candidates))
        selected = candidates[0]
        table = selected.generate_table()

        if confidence < confidence_threshold and len(candidates) > 4:
            decision = "ABSTAIN"
        else:
            decision = "EXECUTE"

        return DiscoveryOutcome(
            task_id=task.task_id,
            decision=decision,
            is_supported=True,
            queries_used=task.queries_used,
            final_candidate_count=len(candidates),
            selected_rule=selected,
            predicted_table=table,
            step_logs=logs,
            confidence=confidence,
        )
