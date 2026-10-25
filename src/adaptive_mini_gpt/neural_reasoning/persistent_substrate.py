"""Milestone M91: Persistent Concept Reasoning Substrate with Compressed Modular Residuals.

Unifies the shared concept trunk (for multi-step instruction-conditioned reasoning)
with selective compressed modular residuals (for persistent skill retention without forgetting):
1. Shared Concept Trunk:
   - Rank-32 procedural transformation: 10,304 parameters
   - Handles multi-step instruction decoding and shared operator execution.
2. Compressed Modular Residuals:
   - Rank-2 task-isolated residuals: 516 parameters per task
       res_down: 128 x 2 = 256
       res_norm: 2 x 2 = 4
       res_up: 2 x 128 = 256
       Total per skill = 516 parameters
   - Zero-initialized up-projection ensures function-preserving addition:
       M_new(x) = M_trunk(x) at initialization (max |delta| = 0.0)
3. Total Budget Accounting:
   - Trunk (10,304) + 6 Skills (3,096) = 13,400 parameters (6,600 headroom below 20,000 cap).
   - Allows up to 18 distinct modular skills concurrently without exceeding budget.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompressedResidual(nn.Module):
    """Rank-2 function-preserving modular residual."""

    def __init__(self, d_model: int = 128, rank: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.down = nn.Linear(d_model, rank, bias=False)
        self.norm = nn.LayerNorm(rank)
        self.up = nn.Linear(rank, d_model, bias=False)

        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)  # Zero initialization preserves base trunk output

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = F.gelu(self.norm(self.down(h)))
        return self.up(z)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class PersistentReasoningSubstrate(nn.Module):
    """Unified hybrid substrate combining shared concept trunk and compressed modular residuals."""

    def __init__(
        self,
        d_model: int = 128,
        d_shared: int = 32,
        num_symbols: int = 16,
        rank_residual: int = 2,
        budget_ceiling: int = 20_000,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_shared = d_shared
        self.num_symbols = num_symbols
        self.rank_residual = rank_residual
        self.budget_ceiling = budget_ceiling

        # Shared Concept Trunk
        self.down_proj = nn.Linear(d_model, d_shared, bias=False)
        self.norm = nn.LayerNorm(d_shared)
        self.up_proj = nn.Linear(d_shared, d_model, bias=False)
        self.head = nn.Linear(d_model, num_symbols, bias=False)

        # Selective Compressed Residuals
        self.residuals = nn.ModuleDict()

        self._reset_trunk_parameters()

        total = self.count_total_parameters()
        if total > self.budget_ceiling:
            raise ValueError(f"Parameters {total} exceed ceiling {budget_ceiling}")

    def _reset_trunk_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down_proj.weight, a=5**0.5)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.kaiming_uniform_(self.head.weight, a=5**0.5)

    def count_trunk_parameters(self) -> int:
        """Count parameters of the shared trunk only."""
        return (
            sum(p.numel() for p in self.down_proj.parameters())
            + sum(p.numel() for p in self.norm.parameters())
            + sum(p.numel() for p in self.up_proj.parameters())
            + sum(p.numel() for p in self.head.parameters())
        )

    def count_residual_parameters(self) -> int:
        """Count total parameters across all registered modular residuals."""
        return sum(p.numel() for p in self.residuals.parameters())

    def count_total_parameters(self) -> int:
        """Count all trainable persistent parameters."""
        return self.count_trunk_parameters() + self.count_residual_parameters()

    def add_residual(self, task_id: str) -> None:
        """Register a compressed rank-2 residual for a task."""
        if task_id in self.residuals:
            return
        res = CompressedResidual(d_model=self.d_model, rank=self.rank_residual)
        new_total = self.count_total_parameters() + res.count_parameters()
        if new_total > self.budget_ceiling:
            raise RuntimeError(
                f"Cannot add residual for {task_id}: would reach {new_total} params > {self.budget_ceiling}"
            )
        self.residuals[task_id] = res

    def freeze_trunk(self) -> None:
        """Freeze shared trunk weights while leaving residuals trainable."""
        for p in self.down_proj.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False
        for p in self.up_proj.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = False

    def unfreeze_trunk(self) -> None:
        """Unfreeze shared trunk weights."""
        for p in self.down_proj.parameters():
            p.requires_grad = True
        for p in self.norm.parameters():
            p.requires_grad = True
        for p in self.up_proj.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def forward(self, h: torch.Tensor, task_id: str | None = None) -> torch.Tensor:
        """Forward pass over contextual hidden states.

        Args:
            h: Hidden states of shape (batch, seq_len, d_model)
            task_id: Optional task identifier to activate a compressed modular residual

        Returns:
            Logits of shape (batch, seq_len, num_symbols)
        """
        z = F.gelu(self.norm(self.down_proj(h)))
        out = h + self.up_proj(z)

        if task_id is not None and task_id in self.residuals:
            out = out + self.residuals[task_id](h)

        return self.head(out)

    def clone(self) -> PersistentReasoningSubstrate:
        """Return an exact deep copy of this substrate."""
        return copy.deepcopy(self)
