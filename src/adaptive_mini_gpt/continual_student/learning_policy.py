"""Milestone M86: Unified Continual Learning Policy.

Compact trainable neural policy (< 1,000 parameters, exactly 989 params)
mapping diagnostic multi-task state signals to continuous student actions:
- REUSE_OR_COMPOSE: Exploit existing repertoire without retraining.
- ACQUIRE_NEW_SKILL: Allocate a new compact module (Rank 4) for unfamiliar rules.
- EXPAND_CAPACITY: Dynamically scale bottleneck rank (4 -> 8) when learning stalls.
- CONSOLIDATE_SHRINK: Consolidate overlapping representations or shrink rank (8 -> 4).
- RETIRE_OBSOLETE: Prune redundant modules to recover parameter budget.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentAction(StrEnum):
    """Discrete capacity and learning actions available to the continual student."""

    REUSE_OR_COMPOSE = "REUSE_OR_COMPOSE"
    ACQUIRE_NEW_SKILL = "ACQUIRE_NEW_SKILL"
    EXPAND_CAPACITY = "EXPAND_CAPACITY"
    CONSOLIDATE_SHRINK = "CONSOLIDATE_SHRINK"
    RETIRE_OBSOLETE = "RETIRE_OBSOLETE"


@dataclass(frozen=True)
class ContinualStudentStateSignals:
    """Observable diagnostic signals from the environment and student internal state."""

    max_existing_skill_em: float      # Highest exact match among existing repertoire [0..1]
    current_loss: float               # Validation cross-entropy loss
    loss_slope: float                 # Loss derivative (negative = improving)
    prediction_entropy: float         # Output prediction entropy / epistemic uncertainty
    capacity_pressure: float          # Allocated parameters / budget ceiling [0..1]
    repertoire_overlap_score: float   # Max functional/feature overlap with existing skills [0..1]

    def to_tensor(self, device: torch.device | None = None) -> torch.Tensor:
        """Convert state signals to normalized 6-dimensional tensor."""
        vals = [
            float(self.max_existing_skill_em),
            float(self.current_loss),
            float(self.loss_slope),
            float(self.prediction_entropy),
            float(self.capacity_pressure),
            float(self.repertoire_overlap_score),
        ]
        return torch.tensor(vals, dtype=torch.float32, device=device).unsqueeze(0)


class UnifiedLearningPolicy(nn.Module):
    """Compact 2-layer MLP policy (< 1,000 parameters) driving autonomous continual decisions."""

    def __init__(
        self,
        in_features: int = 6,
        hidden_dim: int = 24,
        num_actions: int = 5,
    ) -> None:
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
        calib_data = [
            # High existing EM >= 0.85 -> REUSE_OR_COMPOSE (0)
            ([0.95, 0.05, -0.05, 0.10, 0.20, 0.90], 0),
            ([0.90, 0.08, -0.02, 0.12, 0.30, 0.85], 0),
            ([1.00, 0.01, 0.00, 0.05, 0.10, 0.95], 0),
            # Low existing EM, normal loss, moderate slope -> ACQUIRE_NEW_SKILL (1)
            ([0.00, 1.20, -0.05, 0.80, 0.20, 0.10], 1),
            ([0.10, 0.90, -0.04, 0.70, 0.25, 0.05], 1),
            ([0.00, 1.50, -0.06, 0.90, 0.15, 0.02], 1),
            # Stalled progress (slope near 0, loss high) -> EXPAND_CAPACITY (2)
            ([0.00, 1.80, -0.005, 1.40, 0.20, 0.10], 2),
            ([0.05, 1.95, 0.001, 1.50, 0.15, 0.05], 2),
            ([0.00, 1.65, -0.002, 1.35, 0.25, 0.08], 2),
            # Converged / simple task (loss very low, rapid progress) -> CONSOLIDATE_SHRINK (3)
            ([0.00, 0.05, -0.15, 0.10, 0.80, 0.75], 3),
            ([0.00, 0.08, -0.10, 0.15, 0.70, 0.80], 3),
            ([0.00, 0.02, -0.08, 0.05, 0.60, 0.70], 3),
            # Extreme capacity pressure (pressure > 0.9) -> RETIRE_OBSOLETE (4)
            ([0.00, 1.50, -0.01, 1.20, 0.95, 0.20], 4),
            ([0.00, 0.80, -0.02, 0.90, 0.98, 0.30], 4),
        ]

        x_tr = torch.tensor([d[0] for d in calib_data], dtype=torch.float32)
        y_tr = torch.tensor([d[1] for d in calib_data], dtype=torch.long)

        opt = torch.optim.AdamW(self.parameters(), lr=0.015, weight_decay=1e-4)
        self.train()
        for _ in range(160):
            opt.zero_grad()
            h = self.act1(self.ln1(self.fc1(x_tr)))
            h = self.act2(self.ln2(self.fc2(h)))
            out = self.head(h)
            loss = F.cross_entropy(out, y_tr)
            loss.backward()
            opt.step()

        self.eval()

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Compute action logits from state tensor."""
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        h = self.act1(self.ln1(self.fc1(state_tensor)))
        h = self.act2(self.ln2(self.fc2(h)))
        return self.head(h)

    def select_action(
        self,
        state: ContinualStudentStateSignals,
        deterministic: bool = True,
        device: torch.device | None = None,
    ) -> StudentAction:
        """Select discrete student action from observable signals."""
        # 1. Deterministic Rule Override for zero-shot reuse
        if state.max_existing_skill_em >= 0.85:
            return StudentAction.REUSE_OR_COMPOSE

        # 2. Extreme capacity pressure override
        if state.capacity_pressure >= 0.95:
            return StudentAction.RETIRE_OBSOLETE

        # 3. Neural policy evaluation
        x = state.to_tensor(device=device)
        logits = self.forward(x).squeeze(0)

        if deterministic:
            act_idx = int(torch.argmax(logits).item())
        else:
            probs = F.softmax(logits, dim=-1)
            act_idx = int(torch.multinomial(probs, num_samples=1).item())

        actions = [
            StudentAction.REUSE_OR_COMPOSE,
            StudentAction.ACQUIRE_NEW_SKILL,
            StudentAction.EXPAND_CAPACITY,
            StudentAction.CONSOLIDATE_SHRINK,
            StudentAction.RETIRE_OBSOLETE,
        ]
        return actions[act_idx]

    def count_parameters(self) -> int:
        """Count total trainable parameters (must remain strictly < 1,000)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_parameter_hash(self) -> str:
        """Compute SHA-256 fingerprint over all active policy weights."""
        hasher = hashlib.sha256()
        for p in self.parameters():
            hasher.update(p.detach().cpu().contiguous().numpy().tobytes())
        return hasher.hexdigest()[:16]
