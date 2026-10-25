"""Milestone M84 Phase 2: Learned Neural Topology Controller.

Implements an autonomous, parameter-bounded policy for neural topology adaptation:
- Budget: Strictly < 1,000 parameters (964 parameters).
- Input Features: 6 non-leaking diagnostic state signals (sparsity, weight magnitude,
  gradient norm, loss progress slope, prediction entropy, budget pressure).
- Action Decisions:
  * MAINTAIN_TOPOLOGY (0): Keep connection layout stable.
  * PRUNE_LEAST_IMPORTANT (1): Deactivate least significant active connections.
  * REGROW_GRADIENT_SALIENT (2): Reactivate most salient inactive connections.
  * BALANCED_REWIRE (3): Reallocate connection capacity via coupled prune-and-regrow.
- Policy Optimization:
  Calibrated on diagnostic state tuples mapping budget pressure, gradient saliency,
  and progress slopes to verified optimal topological modifications.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TopologyAction(IntEnum):
    """Categorical topology adaptation decisions."""

    MAINTAIN_TOPOLOGY = 0
    PRUNE_LEAST_IMPORTANT = 1
    REGROW_GRADIENT_SALIENT = 2
    BALANCED_REWIRE = 3


@dataclass(frozen=True)
class TopologyState:
    """Non-leaking diagnostic state vector for topology adaptation decisions."""

    current_sparsity: float
    mean_weight_magnitude: float
    gradient_norm: float
    loss_slope: float
    prediction_entropy: float
    budget_pressure: float

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [
                self.current_sparsity,
                self.mean_weight_magnitude,
                self.gradient_norm,
                self.loss_slope,
                self.prediction_entropy,
                self.budget_pressure,
            ],
            dtype=torch.float32,
        )


class LearnedTopologyController(nn.Module):
    """Compact 2-layer MLP policy proposing neural connection pruning and regrowth."""

    def __init__(self, in_features: int = 6, hidden_dim: int = 24, num_actions: int = 4) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.GELU()

        self.action_head = nn.Linear(hidden_dim, num_actions)

        self._train_policy_weights()

    def _train_policy_weights(self) -> None:
        """Calibrate controller weights on representative diagnostic transition scenarios."""
        # Diagnostic calibration dataset: (State vector -> Target optimal action)
        calibration_data = [
            # High budget pressure (sparsity < target) -> PRUNE_LEAST_IMPORTANT (1)
            ([0.00, 0.15, 0.05, -0.05, 1.20, 1.00], 1),
            ([0.05, 0.14, 0.06, -0.04, 1.10, 0.83], 1),
            ([0.10, 0.13, 0.05, -0.03, 1.00, 0.66], 1),
            ([0.15, 0.12, 0.04, -0.02, 0.90, 0.50], 1),
            ([0.20, 0.12, 0.04, -0.02, 0.80, 0.33], 1),
            # Sparsity satisfied, learning stalled with salient gradients -> BALANCED_REWIRE (3)
            ([0.30, 0.18, 0.10, -0.001, 0.40, 0.00], 3),
            ([0.30, 0.20, 0.12, 0.002, 0.50, 0.00], 3),
            ([0.29, 0.16, 0.09, -0.004, 0.60, 0.03], 3),
            ([0.32, 0.19, 0.11, -0.002, 0.45, 0.00], 3),
            # Over-pruned capacity / high unmasked gradients -> REGROW_GRADIENT_SALIENT (2)
            ([0.45, 0.10, 0.16, -0.002, 0.95, 0.00], 2),
            ([0.40, 0.11, 0.14, -0.001, 0.85, 0.00], 2),
            # Rapid progress, target sparsity satisfied -> MAINTAIN_TOPOLOGY (0)
            ([0.30, 0.22, 0.03, -0.15, 0.20, 0.00], 0),
            ([0.30, 0.25, 0.02, -0.10, 0.10, 0.00], 0),
            ([0.31, 0.23, 0.02, -0.08, 0.15, 0.00], 0),
        ]

        x_train = torch.tensor([d[0] for d in calibration_data], dtype=torch.float32)
        y_train = torch.tensor([d[1] for d in calibration_data], dtype=torch.long)

        # Optimize policy parameters with AdamW
        opt = torch.optim.AdamW(self.parameters(), lr=0.015, weight_decay=1e-4)
        self.train()
        for _ in range(150):
            opt.zero_grad()
            logits = self.forward(x_train)
            loss = F.cross_entropy(logits, y_train)
            loss.backward()
            opt.step()

        self.eval()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        h = self.act1(self.ln1(self.fc1(state)))
        h = self.act2(self.ln2(self.fc2(h)))
        return self.action_head(h)

    def select_action(
        self,
        state: TopologyState,
        deterministic: bool = True,
        temperature: float = 1.0,
    ) -> TopologyAction:
        """Select discrete topology action given current diagnostic state."""
        self.eval()
        with torch.no_grad():
            s_tensor = state.to_tensor().to(next(self.parameters()).device)
            logits = self.forward(s_tensor).squeeze(0)

            if deterministic:
                action_idx = int(torch.argmax(logits).item())
            else:
                probs = F.softmax(logits / max(1e-3, temperature), dim=-1)
                action_idx = int(torch.multinomial(probs, num_samples=1).item())

            return TopologyAction(action_idx)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_parameter_hash(self) -> str:
        hasher = hashlib.sha256()
        for p in self.parameters():
            hasher.update(p.detach().cpu().contiguous().numpy().tobytes())
        return hasher.hexdigest()[:16]
