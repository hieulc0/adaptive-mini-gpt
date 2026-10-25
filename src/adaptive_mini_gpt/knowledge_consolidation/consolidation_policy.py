"""Milestone M83 Phase 4: Learned Consolidation Policy Controller.

Implements a compact trainable policy (< 1,000 parameters) that inspects candidate compatibility signals:
- Diagnostic State (6 features):
  1. weight_similarity
  2. representation_similarity
  3. functional_agreement
  4. gradient_alignment
  5. compression_potential
  6. memory_budget_pressure

- Bounded Actions (3 categorical choices):
  0: CONSOLIDATE_SHARED (High compression, pure shared trunk)
  1: CONSOLIDATE_WITH_RESIDUALS (Balanced compression, shared trunk + residual heads)
  2: REJECT_INDEPENDENT (Preserve separate adapters, prevent destructive interference)

- Policy Optimization:
  Trained on consolidation episodes rewarding parameter savings and penalizing retention drop / rollback.
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


class ConsolidationAction(IntEnum):
    """Categorical actions for knowledge consolidation."""

    CONSOLIDATE_SHARED = 0
    CONSOLIDATE_WITH_RESIDUALS = 1
    REJECT_INDEPENDENT = 2


@dataclass(frozen=True)
class ConsolidationState:
    """State feature vector representing multi-metric compatibility and memory pressure."""

    weight_similarity: float
    representation_similarity: float
    functional_agreement: float
    gradient_alignment: float
    compression_potential: float
    memory_budget_pressure: float

    def to_tensor(self) -> torch.Tensor:
        """Convert state into normalized 1D tensor [6]."""
        f0 = min(max(self.weight_similarity, -1.0), 1.0)
        f1 = min(max(self.representation_similarity, -1.0), 1.0)
        f2 = min(max(self.functional_agreement, 0.0), 1.0)
        f3 = min(max(self.gradient_alignment, -1.0), 1.0)
        f4 = min(max(self.compression_potential, 0.0), 1.0)
        f5 = min(max(self.memory_budget_pressure, 0.0), 1.0)
        return torch.tensor([f0, f1, f2, f3, f4, f5], dtype=torch.float32)


class LearnedConsolidationController(nn.Module):
    """Compact neural consolidation policy (< 1,000 parameters)."""

    def __init__(self, in_features: int = 6, hidden_dim: int = 24) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 3)

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Compute logits over the 3 consolidation actions [B, 3]."""
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        h = self.act(self.ln1(self.fc1(state_tensor)))
        h = self.act(self.ln2(self.fc2(h)))
        return self.head(h)

    def select_decision(
        self,
        state: ConsolidationState,
        deterministic: bool = True,
    ) -> ConsolidationAction:
        """Predict consolidation action for candidate skill cluster."""
        self.eval()
        device = next(self.parameters()).device
        state_tensor = state.to_tensor().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = self(state_tensor)
            if deterministic:
                idx = int(torch.argmax(logits, dim=-1).item())
            else:
                dist = torch.distributions.Categorical(logits=logits)
                idx = int(dist.sample().item())
        return ConsolidationAction(idx)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_parameter_hash(self) -> str:
        """Compute SHA-256 fingerprint of controller parameters."""
        hasher = hashlib.sha256()
        for p in self.parameters():
            hasher.update(p.detach().cpu().contiguous().numpy().tobytes())
        return hasher.hexdigest()[:16]
