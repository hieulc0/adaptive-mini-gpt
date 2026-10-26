"""Milestone M104-C Active Practice Selection and Meta-Transfer Framework.

Compares 6 practice selection policies across candidate pools (28, 100, 200):
1. 'random': uniform random candidate selection.
2. 'fixed': sequential curriculum.
3. 'calibrated_uncertainty': predictive entropy with temperature scaling (T=1.5).
4. 'diversity_coverage': greedy farthest-point exploration in representation space (1 - max_sim).
5. 'hybrid_novelty_entropy': fixed multiplicative combination (Novelty * Calibrated Entropy).
6. 'meta_trainable_policy': learned policy network mapping latent representations and dynamics
   features into expected learning value, trained across worlds and evaluated on unseen transfer worlds.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.natural_language_learning_m103.causal_world_m104_b import (
    CausalItemM104_B,
    ProceduralCausalWorldM104_B,
)


@dataclass
class PracticeStepRecordM104_C:
    step_index: int
    object_name: str
    action_name: str
    predicted_state: str
    predicted_confidence: float
    true_state: str
    was_correct: bool
    entropy: float
    novelty_dist: float
    selected_score: float
    simulator_calls_total: int


@dataclass
class PolicyLearningTrajectoryM104_C:
    policy_name: str
    candidate_pool_size: int
    simulator_budget: int
    simulator_calls_used: int
    steps_records: list[PracticeStepRecordM104_C]
    held_out_accuracy_curve: list[tuple[int, float]]
    final_held_out_acc: float
    area_under_curve: float
    sample_efficiency_ratio: float
    training_duration_sec: float


class MetaTrainablePracticePolicyM104_C(nn.Module):
    """Learned practice policy network scoring candidate experiments by expected learning value."""

    def __init__(self, latent_dim: int = 128, feature_dim: int = 3, hidden_dim: int = 64) -> None:
        super().__init__()
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.feat_proj = nn.Linear(feature_dim, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cand_latents: torch.Tensor, hand_crafted_feats: torch.Tensor) -> torch.Tensor:
        """Computes scalar expected learning value scores for candidates (N,)."""
        z_emb = F.gelu(self.latent_proj(cand_latents))
        f_emb = F.gelu(self.feat_proj(hand_crafted_feats))
        comb = torch.cat([z_emb, f_emb], dim=-1)
        return self.scorer(comb).squeeze(-1)


class ActivePracticeEnvironmentM104_C:
    """Isolated practice environment enforcing simulation call limits and anti-leakage barriers."""

    def __init__(
        self,
        causal_world: ProceduralCausalWorldM104_B,
        candidate_pool: list[CausalItemM104_B],
        simulator_budget: int = 20,
        seed: int = 42,
    ) -> None:
        self.causal_world = causal_world
        self.candidate_pool = list(candidate_pool)
        self.simulator_budget = simulator_budget
        self.simulator_calls_used = 0
        self.rng = random.Random(seed)

    def query_simulator(self, item: CausalItemM104_B) -> str:
        """Executes a single simulation query against ground truth causal dynamics."""
        if self.simulator_calls_used >= self.simulator_budget:
            raise RuntimeError(f"Simulator budget exhausted ({self.simulator_budget} calls)")
        self.simulator_calls_used += 1
        return self.causal_world.transition_state(
            item.object_name, item.action_name, item.initial_state, realm="Alchemy"
        )
