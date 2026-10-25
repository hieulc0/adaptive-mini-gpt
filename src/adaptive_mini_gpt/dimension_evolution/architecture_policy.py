"""Milestone M85 Phase 3: Learned Architecture Policy Controller.

Implements a compact autonomous policy (< 1,000 parameters) proposing architectural decisions:
- Input Features (6 diagnostic signals):
  1. loss_slope: Recent derivative of cross-entropy loss progress.
  2. validation_loss: Current task qualification error.
  3. prediction_entropy: Output probability uncertainty.
  4. gradient_norm: Adapter backprop gradient magnitude.
  5. capacity_pressure: Current parameter footprint relative to budget ceiling.
  6. interference_risk: Cosine alignment with existing skills.

- Discrete Action Decisions (5 choices):
  0: REUSE_EXISTING (Route directly to mastered skill without changes)
  1: ADAPT_CURRENT (Continue fine-tuning at current dimension)
  2: EXPAND_RANK (Enlarge bottleneck rank: 2->4, 4->8, 8->16)
  3: SHRINK_RANK (Contract bottleneck rank: 16->8, 8->4, 4->2)
  4: RETIRE_REDUNDANT (Deallocate obsolete or superseded module)

- Parameter Accounting:
  Strictly < 1,000 parameters (exactly 989 parameters).
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


class ArchitectureAction(IntEnum):
    """Categorical architectural decisions."""

    REUSE_EXISTING = 0
    ADAPT_CURRENT = 1
    EXPAND_RANK = 2
    SHRINK_RANK = 3
    RETIRE_REDUNDANT = 4


@dataclass(frozen=True)
class ArchitectureState:
    """Non-leaking diagnostic state vector for architectural decision-making."""

    loss_slope: float
    validation_loss: float
    prediction_entropy: float
    gradient_norm: float
    capacity_pressure: float
    interference_risk: float

    def to_tensor(self) -> torch.Tensor:
        """Convert state into normalized 1D tensor [6]."""
        return torch.tensor(
            [
                self.loss_slope,
                self.validation_loss,
                self.prediction_entropy,
                self.gradient_norm,
                self.capacity_pressure,
                self.interference_risk,
            ],
            dtype=torch.float32,
        )


class LearnedArchitecturePolicy(nn.Module):
    """Compact 2-layer MLP policy proposing neural dimension and module mutations."""

    def __init__(self, in_features: int = 6, hidden_dim: int = 24, num_actions: int = 5) -> None:
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

        self.head = nn.Linear(hidden_dim, num_actions)

        self._calibrate_policy_weights()

    def _calibrate_policy_weights(self) -> None:
        """Calibrate policy parameters on diagnostic scenarios."""
        # Calibration dataset: (State vector -> Target optimal action)
        calibration_data = [
            # High loss, stalled slope, high gradient, low capacity pressure -> EXPAND_RANK (2)
            ([0.002, 1.80, 1.40, 0.25, 0.20, 0.10], 2),
            ([-0.001, 1.50, 1.30, 0.20, 0.30, 0.05], 2),
            ([0.001, 1.95, 1.50, 0.30, 0.15, 0.15], 2),
            # Low loss, rapid progress, high capacity pressure -> SHRINK_RANK (3)
            ([-0.15, 0.05, 0.10, 0.02, 0.85, 0.00], 3),
            ([-0.10, 0.08, 0.15, 0.03, 0.75, 0.02], 3),
            ([-0.08, 0.12, 0.20, 0.04, 0.80, 0.01], 3),
            # Zero loss, low entropy, matching existing representation -> REUSE_EXISTING (0)
            ([0.000, 0.01, 0.02, 0.001, 0.50, 0.00], 0),
            ([-0.001, 0.02, 0.03, 0.002, 0.40, 0.01], 0),
            # Normal steady convergence -> ADAPT_CURRENT (1)
            ([-0.04, 0.60, 0.80, 0.08, 0.40, 0.05], 1),
            ([-0.05, 0.45, 0.60, 0.06, 0.45, 0.03], 1),
            ([-0.03, 0.80, 0.90, 0.09, 0.35, 0.04], 1),
            # High interference / redundant duplicate -> RETIRE_REDUNDANT (4)
            ([0.000, 0.05, 0.10, 0.01, 0.95, 0.90], 4),
            ([0.001, 0.08, 0.12, 0.02, 0.90, 0.85], 4),
        ]

        x_train = torch.tensor([d[0] for d in calibration_data], dtype=torch.float32)
        y_train = torch.tensor([d[1] for d in calibration_data], dtype=torch.long)

        opt = torch.optim.AdamW(self.parameters(), lr=0.015, weight_decay=1e-4)
        self.train()
        for _ in range(160):
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
        return self.head(h)

    def select_action(
        self,
        state: ArchitectureState,
        deterministic: bool = True,
        temperature: float = 1.0,
    ) -> ArchitectureAction:
        """Select discrete architectural action given current diagnostic state."""
        self.eval()
        with torch.no_grad():
            s_tensor = state.to_tensor().to(next(self.parameters()).device)
            logits = self.forward(s_tensor).squeeze(0)

            if deterministic:
                action_idx = int(torch.argmax(logits).item())
            else:
                probs = F.softmax(logits / max(1e-3, temperature), dim=-1)
                action_idx = int(torch.multinomial(probs, num_samples=1).item())

            return ArchitectureAction(action_idx)

    def count_parameters(self) -> int:
        """Count total trainable parameters (must be < 1,000)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_parameter_hash(self) -> str:
        """Compute SHA-256 fingerprint of policy parameters."""
        hasher = hashlib.sha256()
        for p in self.parameters():
            hasher.update(p.detach().cpu().contiguous().numpy().tobytes())
        return hasher.hexdigest()[:16]
