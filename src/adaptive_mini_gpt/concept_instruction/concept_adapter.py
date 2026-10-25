"""Milestone M90: Instruction-Conditioned Shared Neural Concept Adapter.

Implements the low-rank concept execution trunk:
1. Low-Rank Bottleneck:
   - Input: Hidden state representations h in R^{B x L_sym x d_model} extracted from
     symbol positions of the frozen MiniGPT backbone.
   - Transformation:
       z = down_proj(h)
       z = LayerNorm(z)
       z = GELU(z)
       logits = head(h + up_proj(z))
2. Parameter Budgeting:
   - Rank 16: 6,176 parameters
   - Rank 32: 10,304 parameters
   - Rank 48: 14,432 parameters
   All strictly <= 20,000 ceiling.
3. Completely eliminates task-id index lookup tables; computation is conditioned
   directly on the contextual representations formed by MiniGPT's causal self-attention.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class InstructionConceptTrunk(nn.Module):
    """Shared low-rank concept adapter for instruction-guided neural execution."""

    def __init__(
        self,
        d_model: int = 128,
        d_shared: int = 32,
        num_symbols: int = 16,
        budget_ceiling: int = 20_000,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_shared = d_shared
        self.num_symbols = num_symbols
        self.budget_ceiling = budget_ceiling

        self.down_proj = nn.Linear(d_model, d_shared, bias=False)
        self.norm = nn.LayerNorm(d_shared)
        self.up_proj = nn.Linear(d_shared, d_model, bias=False)
        self.head = nn.Linear(d_model, num_symbols, bias=False)

        self._reset_parameters()

        total_p = self.count_parameters()
        if total_p > self.budget_ceiling:
            raise ValueError(
                f"Parameter count {total_p} exceeds budget ceiling {budget_ceiling}"
            )

    def _reset_parameters(self) -> None:
        """Kaiming uniform initialization with standard adapter scale."""
        nn.init.kaiming_uniform_(self.down_proj.weight, a=5**0.5)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.kaiming_uniform_(self.head.weight, a=5**0.5)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass over contextual hidden states at symbol positions.

        Args:
            h: Hidden state tensor of shape (batch, seq_len, d_model)

        Returns:
            Logits of shape (batch, seq_len, num_symbols)
        """
        z = self.down_proj(h)
        z = self.norm(z)
        z = F.gelu(z)
        z_up = self.up_proj(z)
        logits = self.head(h + z_up)
        return logits

    def count_parameters(self) -> int:
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def clone(self) -> InstructionConceptTrunk:
        """Return a deep copy of this trunk with identical architecture and weights."""
        return copy.deepcopy(self)
