"""Stage B.0 Autonomous Practice Selection Policies.

Implements matched practice selection mechanisms:
1. Uniform Random Selection (B0_ARM_1, B0_ARM_2)
2. Fixed Uncertainty/Disagreement Heuristic (B0_ARM_3)
3. Compact Trainable Neural Practice Policy (B0_ARM_4, <1,000 parameters)
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_binary_entropy(probs: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Computes binary entropy H(p) = -p log2(p) - (1-p) log2(1-p)."""
    p = probs.clamp(min=eps, max=1.0 - eps)
    h = -p * torch.log2(p) - (1.0 - p) * torch.log2(1.0 - p)
    return h.mean(dim=-1)  # (N,)


class FixedHeuristicSelectorB0:
    """Fixed active learning heuristic: maximum predictive entropy + boundary ambiguity."""

    @staticmethod
    def rank_candidates(
        logits: torch.Tensor,
        unqueried_indices: list[int],
        k: int = 16,
    ) -> list[int]:
        """Ranks candidates by predictive entropy."""
        sub_logits = logits[unqueried_indices]
        probs = torch.sigmoid(sub_logits)
        entropy = compute_binary_entropy(probs)
        # Higher entropy = higher uncertainty
        top_k_local = torch.topk(entropy, min(k, len(unqueried_indices))).indices.tolist()
        return [unqueried_indices[i] for i in top_k_local]


class NeuralPracticePolicyB0(nn.Module):
    """Compact 2-layer MLP policy (<1,000 parameters) predicting candidate selection utility.

    Features (4D):
    0: Predictive entropy H(p)
    1: Logit margin tanh(|logits|)
    2: Cosine novelty to replay representations
    3: Recent validation learning progress / loss slope
    """

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.param_count = sum(p.numel() for p in self.parameters())
        assert self.param_count < 1000, f"Policy parameter count exceeds 1,000: {self.param_count}"

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.saved_log_probs: list[torch.Tensor] = []

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting scalar priority scores (N,)."""
        x = self.relu(self.fc1(features))
        out = self.fc2(x).squeeze(-1)
        return out

    def select_candidates(
        self,
        features: torch.Tensor,
        unqueried_indices: list[int],
        k: int = 16,
        temperature: float = 1.0,
        sample: bool = True,
    ) -> tuple[list[int], torch.Tensor | None]:
        """Selects k candidate indices based on policy scores."""
        sub_feats = features[unqueried_indices]
        scores = self.forward(sub_feats)  # (M,)
        k_actual = min(k, len(unqueried_indices))

        if sample and len(unqueried_indices) > k_actual:
            log_probs = F.log_softmax(scores / temperature, dim=-1)
            probs = torch.exp(log_probs.detach())
            # Sample without replacement
            sampled_locals = torch.multinomial(probs, k_actual, replacement=False)
            log_prob_sum = log_probs.gather(0, sampled_locals).sum()
            selected = [unqueried_indices[int(i)] for i in sampled_locals.tolist()]
            return selected, log_prob_sum
        else:
            top_k_local = torch.topk(scores, k_actual).indices.tolist()
            selected = [unqueried_indices[i] for i in top_k_local]
            return selected, None

    def update_policy(self, log_prob: torch.Tensor, reward: float, baseline: float = 0.0) -> float:
        """REINFORCE policy gradient update."""
        advantage = reward - baseline
        loss = -log_prob * advantage
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


def extract_candidate_features(
    logits: torch.Tensor,
    h_cands: torch.Tensor,
    h_replay: torch.Tensor | None,
    loss_slope: float,
) -> torch.Tensor:
    """Constructs (N, 4) candidate feature tensor.

    Features:
    0: Predictive binary entropy H(p)
    1: Logit margin tanh(|logits|)
    2: Cosine novelty to replay representations
    3: Recent validation loss slope
    """
    device = logits.device
    n = logits.size(0)

    # 0. Entropy
    probs = torch.sigmoid(logits)
    entropy = compute_binary_entropy(probs).unsqueeze(-1)  # (N, 1)

    # 1. Logit margin
    margin = torch.tanh(logits.abs().mean(dim=-1, keepdim=True))  # (N, 1)

    # 2. Novelty to replay
    if h_replay is not None and h_replay.size(0) > 0:
        sim = F.cosine_similarity(h_cands.unsqueeze(1), h_replay.unsqueeze(0), dim=-1)  # (N, R)
        max_sim = sim.max(dim=-1, keepdim=True).values
        novelty = (1.0 - max_sim).clamp(min=0.0, max=2.0)
    else:
        novelty = torch.ones((n, 1), device=device)

    # 3. Loss slope
    slope_feat = torch.full((n, 1), float(loss_slope), device=device)

    features = torch.cat([entropy, margin, novelty, slope_feat], dim=-1)
    return features


def uniform_random_select(
    unqueried_indices: list[int],
    k: int = 16,
    rng: random.Random | None = None,
) -> list[int]:
    """Uniform random selection from available unqueried candidate pool."""
    if rng is None:
        rng = random.Random()
    k_actual = min(k, len(unqueried_indices))
    return rng.sample(unqueried_indices, k_actual)
