"""Milestone M104-A Autonomous Active Practice Environment and Policy Selectors.

Provides a bounded active-practice environment where the student:
1. Observes candidate experiments from an unobserved pool.
2. Selects which experiment to query using a practice policy:
   - RandomPracticeSelector: Uniform random sampling.
   - FixedCurriculumSelector: Deterministic static ordering.
   - UncertaintyPracticeSelector: Maximum predictive entropy / margin uncertainty.
   - TrainablePracticeSelector: Policy network trained on error-reduction signal.
3. Predicts the outcome prior to observing simulator feedback.
4. Queries simulator (consuming 1 budget unit).
5. Updates internal neural representations with bounded adaptation.
6. Evaluates sample efficiency and held-out learning curves without accessing sealed test labels.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn

from adaptive_mini_gpt.natural_language_learning_m103.causal_world_m103_g1 import (
    CausalItemM103_G1,
    ProceduralCausalWorldM103_G1,
)


@dataclass
class PracticeStepRecordM104_A:
    step_index: int
    object_name: str
    action_name: str
    predicted_state: str
    predicted_confidence: float
    true_state: str
    was_correct: bool
    policy_uncertainty: float
    simulator_calls_total: int


@dataclass
class PolicyLearningTrajectoryM104_A:
    policy_name: str
    simulator_budget: int
    simulator_calls_used: int
    steps_records: list[PracticeStepRecordM104_A]
    held_out_accuracy_curve: list[tuple[int, float]]  # (simulator_calls, test_acc_pct)
    final_held_out_acc: float
    area_under_curve: float
    sample_efficiency_ratio: float
    training_duration_sec: float


class TrainablePolicyNetworkM104_A(nn.Module):
    """Neural policy network selecting experiment candidates based on representations and uncertainty."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        # Input: latent representation z (128) + 1 uncertainty scalar + 1 novelty scalar = 130
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim + 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, z_candidates: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Computes selection logits for candidates. Shape: (N, 1) -> (N,)."""
        x = torch.cat([z_candidates, features], dim=-1)
        return self.scorer(x).squeeze(-1)


class ActivePracticeEnvironmentM104_A:
    """Bounded active-practice environment with strict budget and sealed test isolation."""

    def __init__(
        self,
        causal_world: ProceduralCausalWorldM103_G1,
        candidate_pool: list[CausalItemM103_G1],
        simulator_budget: int = 15,
        seed: int = 42,
    ) -> None:
        self.causal_world = causal_world
        self.candidate_pool = list(candidate_pool)
        self.simulator_budget = simulator_budget
        self.seed = seed
        self.rng = random.Random(seed)
        self.simulator_calls_used = 0

    def query_simulator(self, item: CausalItemM103_G1) -> str:
        """Ground truth outcome from simulator. Consumes 1 budget unit.

        Strict check: Throws error if budget exhausted.
        """
        if self.simulator_calls_used >= self.simulator_budget:
            raise RuntimeError(f"Simulator budget exhausted ({self.simulator_budget} calls)")
        self.simulator_calls_used += 1
        return self.causal_world.transition_state(
            item.object_name,
            item.action_name,
            current_state=item.initial_state,
            realm="Shadow" if item.is_counterfactual else "Alchemy",
        )
