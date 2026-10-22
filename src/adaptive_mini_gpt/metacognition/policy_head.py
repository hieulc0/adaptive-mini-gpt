"""M35: Neural Metacognitive Action Policy Head.

Extracts pooled representations from the MiniGPT backbone and predicts:
  1. Action distribution over the 7 LearningActions.
  2. Model confidence and predictive entropy (uncertainty estimate).
Supports both standard supervised classification and outcome-aware utility training.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from adaptive_mini_gpt.metacognition.action_space import LearningAction
from adaptive_mini_gpt.model.transformer import MiniGPT


ACTION_LIST = list(LearningAction)
ACTION_TO_IDX = {act: idx for idx, act in enumerate(ACTION_LIST)}
IDX_TO_ACTION = {idx: act for idx, act in enumerate(ACTION_LIST)}


class NeuralMetacognitivePolicy(nn.Module):
    """Compact neural head that maps backbone representations to metacognitive actions."""

    def __init__(
        self,
        d_model: int = 128,
        d_hidden: int = 64,
        num_actions: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_actions = num_actions
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, num_actions),
        )

    def extract_pooled_representation(self, model: MiniGPT, idx: Tensor) -> Tensor:
        """Extract mean-pooled sequence representation from frozen MiniGPT backbone."""
        b, t = idx.size()
        device = idx.device
        tok_emb = model.wte(idx)
        if model.wpe is not None:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = model.wpe(pos)
            x = model.drop(tok_emb + pos_emb)
        else:
            x = model.drop(tok_emb)

        for block in model.blocks:
            x, _ = block(x)
        x = model.ln_f(x)
        # Mean-pool over sequence length
        return x.mean(dim=1)

    def forward(self, pooled_features: Tensor) -> Tensor:
        """Compute action logits from pooled representations."""
        return self.mlp(pooled_features)

    def predict(
        self,
        model: MiniGPT,
        idx: Tensor,
    ) -> tuple[LearningAction, float, float, Tensor]:
        """Predict action, confidence, and normalized entropy uncertainty.

        Returns:
            (predicted_action, confidence_score, uncertainty_score, action_probabilities)
        """
        self.eval()
        with torch.no_grad():
            pooled = self.extract_pooled_representation(model, idx)
            logits = self.forward(pooled)
            probs = torch.softmax(logits, dim=-1)[0]
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(probs[pred_idx].item())

            # Normalized predictive entropy H(p) / log(N) in [0, 1]
            eps = 1e-8
            entropy = -torch.sum(probs * torch.log(probs + eps)).item()
            max_entropy = math.log(self.num_actions)
            normalized_uncertainty = max(0.0, min(1.0, entropy / max_entropy))

            pred_action = IDX_TO_ACTION[pred_idx]
            return pred_action, confidence, normalized_uncertainty, probs


def train_supervised_policy(
    policy: NeuralMetacognitivePolicy,
    features: Tensor,
    targets: Tensor,
    epochs: int = 25,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
) -> list[float]:
    """Train policy head with standard supervised cross-entropy loss."""
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []

    policy.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = policy(features)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        losses.append(round(loss.item(), 4))

    policy.eval()
    return losses


def train_outcome_aware_policy(
    policy: NeuralMetacognitivePolicy,
    features: Tensor,
    target_action_indices: Tensor,
    action_utilities: Tensor,
    epochs: int = 25,
    lr: float = 1e-3,
    utility_weight: float = 0.5,
    weight_decay: float = 0.01,
) -> list[float]:
    """Train policy head using cross-entropy augmented with empirical action utility.

    Loss combines classification loss with utility reward maximization:
      L = L_CE - lambda * mean(action_utility * log(prob_taken))
    """
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    ce_loss_fn = nn.CrossEntropyLoss()
    losses: list[float] = []

    policy.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = policy(features)
        ce_loss = ce_loss_fn(logits, target_action_indices)

        probs = torch.softmax(logits, dim=-1)
        # Gather chosen action probability
        chosen_probs = probs.gather(1, target_action_indices.unsqueeze(1)).squeeze(1)
        log_chosen = torch.log(chosen_probs + 1e-8)
        # Maximize utility => minimize - (utility * log_prob)
        utility_loss = -torch.mean(action_utilities * log_chosen)

        total_loss = ce_loss + utility_weight * utility_loss
        total_loss.backward()
        optimizer.step()
        losses.append(round(total_loss.item(), 4))

    policy.eval()
    return losses
