"""Milestone M88 Phase 1: Shared Neural Knowledge Substrate.

Provides the foundational shared representation architecture:
1. SharedKnowledgeTrunk:
   - Common low-rank neural knowledge layer operating on frozen transformer representations (d_model=128).
   - Projects into shared bottleneck (d_shared=16) with LayerNorm and GELU, projecting back to output vocabulary.
   - Encodes shared procedural knowledge across arithmetic, logic, place-value, and sequence domains.
   - Total trunk parameters: 6,176 parameters (strictly <= 20,000 budget).
2. SelectiveResidualAdapter:
   - Ultra-compact Rank-1 task-specific residual module (144 parameters).
   - Only instantiated when a task requires specialized non-shared offset or operation modulation.
   - Enables dozens of tasks to coexist simultaneously in memory without deallocation or modular amnesia.
3. SharedFoundationalSubstrate:
   - Coordinates the shared trunk and active task residuals.
   - Supports forward execution, residual allocation, parameter accounting, and bit-exact state checkpoints.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedKnowledgeTrunk(nn.Module):
    """Common foundational neural representation layer across all tasks."""

    def __init__(
        self,
        d_model: int = 128,
        d_shared: int = 16,
        num_symbols: int = 16,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_shared = d_shared
        self.num_symbols = num_symbols

        self.down_proj = nn.Linear(d_model, d_shared, bias=False)
        self.norm = nn.LayerNorm(d_shared)
        self.up_proj = nn.Linear(d_shared, d_model, bias=False)
        self.head = nn.Linear(d_model, num_symbols, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down_proj.weight, a=5**0.5)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.kaiming_uniform_(self.head.weight, a=5**0.5)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass through shared knowledge trunk: (B, L, d_model) -> (B, L, num_symbols)."""
        z = self.down_proj(h)
        z = self.norm(z)
        z = F.gelu(z)
        z_up = self.up_proj(z)
        logits = self.head(h + z_up)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SelectiveResidualAdapter(nn.Module):
    """Ultra-compact Rank-1 task-specific residual adapter (144 parameters)."""

    def __init__(
        self,
        d_model: int = 128,
        num_symbols: int = 16,
        rank: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_symbols = num_symbols
        self.rank = rank

        self.res_down = nn.Linear(d_model, rank, bias=False)
        self.res_up = nn.Linear(rank, num_symbols, bias=False)

        nn.init.kaiming_uniform_(self.res_down.weight, a=5**0.5)
        nn.init.normal_(self.res_up.weight, std=0.02)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass of compact residual: (B, L, d_model) -> (B, L, num_symbols)."""
        return self.res_up(F.relu(self.res_down(h)))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SharedFoundationalSubstrate(nn.Module):
    """Unified Foundational Substrate coordinating shared trunk and compact residuals."""

    def __init__(
        self,
        d_model: int = 128,
        d_shared: int = 16,
        num_symbols: int = 16,
        budget_ceiling: int = 20_000,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_shared = d_shared
        self.num_symbols = num_symbols
        self.budget_ceiling = budget_ceiling

        self.trunk = SharedKnowledgeTrunk(
            d_model=d_model, d_shared=d_shared, num_symbols=num_symbols
        )
        self.residuals = nn.ModuleDict()
        self.lru_order: list[str] = []

    def forward(self, h: torch.Tensor, task_id: str | None = None) -> torch.Tensor:
        """Forward pass combining shared foundational trunk with optional task residual."""
        base_logits = self.trunk(h)
        if task_id is not None:
            sanitized_id = task_id.replace(".", "_").replace("-", "_")
            if sanitized_id in self.residuals:
                if sanitized_id in self.lru_order:
                    self.lru_order.remove(sanitized_id)
                    self.lru_order.append(sanitized_id)
                res_logits = self.residuals[sanitized_id](h)
                return base_logits + res_logits
        return base_logits

    def allocate_residual(self, task_id: str, rank: int = 1) -> SelectiveResidualAdapter:
        """Allocate ultra-compact task residual if budget permits, or retire least recently accessed."""
        sanitized_id = task_id.replace(".", "_").replace("-", "_")
        if sanitized_id in self.residuals:
            if sanitized_id in self.lru_order:
                self.lru_order.remove(sanitized_id)
            self.lru_order.append(sanitized_id)
            return self.residuals[sanitized_id]

        dev = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        new_adapter = SelectiveResidualAdapter(
            d_model=self.d_model, num_symbols=self.num_symbols, rank=rank
        ).to(dev)

        # Enforce parameter budget via LRU retirement if needed
        adapter_params = new_adapter.count_parameters()
        while (self.count_parameters() + adapter_params) > self.budget_ceiling and self.lru_order:
            oldest = self.lru_order.pop(0)
            if oldest in self.residuals:
                del self.residuals[oldest]

        # Check parameter budget
        projected = self.count_parameters() + adapter_params
        if projected > self.budget_ceiling:
            raise RuntimeError(
                f"Parameter budget exceeded: {projected} > {self.budget_ceiling}"
            )

        self.residuals[sanitized_id] = new_adapter
        self.lru_order.append(sanitized_id)
        return new_adapter

    def has_residual(self, task_id: str) -> bool:
        sanitized_id = task_id.replace(".", "_").replace("-", "_")
        return sanitized_id in self.residuals

    def get_residual(self, task_id: str) -> SelectiveResidualAdapter | None:
        sanitized_id = task_id.replace(".", "_").replace("-", "_")
        return self.residuals.get(sanitized_id, None)

    def count_parameters(self) -> int:
        """Count total trainable parameters across trunk and all residuals."""
        trunk_p = self.trunk.count_parameters()
        res_p = sum(r.count_parameters() for r in self.residuals.values())
        return trunk_p + res_p

    def count_trunk_parameters(self) -> int:
        return self.trunk.count_parameters()

    def count_residual_parameters(self) -> int:
        return sum(r.count_residual_parameters() if hasattr(r, "count_residual_parameters") else r.count_parameters() for r in self.residuals.values())

    def save_state(self) -> dict[str, Any]:
        """Create exact serializable state dict copy for bit-exact checkpoint rollback."""
        return {
            "trunk_state": copy.deepcopy(self.trunk.state_dict()),
            "residuals_state": {k: copy.deepcopy(v.state_dict()) for k, v in self.residuals.items()},
            "lru_order": list(self.lru_order),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore exact state dict for bit-exact rollback."""
        dev = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        self.trunk.load_state_dict(state["trunk_state"])
        # Clear extra residuals and restore saved ones
        self.residuals.clear()
        self.lru_order = list(state.get("lru_order", []))
        for k, v in state["residuals_state"].items():
            adapter = SelectiveResidualAdapter(
                d_model=self.d_model, num_symbols=self.num_symbols, rank=1
            ).to(dev)
            adapter.load_state_dict(v)
            self.residuals[k] = adapter
