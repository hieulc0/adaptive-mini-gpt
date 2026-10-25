"""Milestone M89 Phase 1: Task-Conditioned Shared Neural Knowledge Substrate.

Provides explicit task conditioning to eliminate the 1-to-many mapping failure mode
discovered during the M88 forensic audit:
1. TaskConditionedKnowledgeTrunk:
   - Shared low-rank procedural trunk: d_model (128) -> d_shared (16) -> d_model (128).
   - Task Conditioning Embedding: compact task vector e_t in R^{d_shared} (16 parameters per task)
     injected into the shared bottleneck:
       z = down_proj(h) + e_task
       z = LayerNorm(z)
       z = GELU(z)
       logits = head(h + up_proj(z))
   - Parameters:
     Trunk: 6,176 parameters.
     Task embeddings (up to 100 tasks): 100 * 16 = 1,600 parameters.
     Total: 7,776 parameters (strictly <= 20,000 budget, 12,224 headroom).
2. Supports forward execution with task index/id, feature extraction, and bit-exact state checkpoints.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskConditionedKnowledgeTrunk(nn.Module):
    """Shared knowledge trunk with explicit low-rank task conditioning embeddings."""

    def __init__(
        self,
        max_tasks: int = 100,
        d_model: int = 128,
        d_shared: int = 16,
        num_symbols: int = 16,
        budget_ceiling: int = 20_000,
    ) -> None:
        super().__init__()
        self.max_tasks = max_tasks
        self.d_model = d_model
        self.d_shared = d_shared
        self.num_symbols = num_symbols
        self.budget_ceiling = budget_ceiling

        # Shared foundational transformation trunk
        self.down_proj = nn.Linear(d_model, d_shared, bias=False)
        self.norm = nn.LayerNorm(d_shared)
        self.up_proj = nn.Linear(d_shared, d_model, bias=False)
        self.head = nn.Linear(d_model, num_symbols, bias=False)

        # Compact task-conditioning embeddings: (max_tasks, d_shared)
        self.task_embeddings = nn.Embedding(max_tasks, d_shared)

        # Mapping from task_id (str) -> task_idx (int)
        self.task_registry: dict[str, int] = {}

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.down_proj.weight, a=5**0.5)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.kaiming_uniform_(self.head.weight, a=5**0.5)
        nn.init.normal_(self.task_embeddings.weight, std=0.02)

    def register_task(self, task_id: str) -> int:
        """Register a task string and return its integer index."""
        if task_id not in self.task_registry:
            if len(self.task_registry) >= self.max_tasks:
                raise RuntimeError(f"Max tasks capacity reached: {self.max_tasks}")
            idx = len(self.task_registry)
            self.task_registry[task_id] = idx
        return self.task_registry[task_id]

    def forward(
        self,
        h: torch.Tensor,
        task_id: str | int | None = None,
    ) -> torch.Tensor:
        """Forward pass combining shared trunk representations with task conditioning vector."""
        z = self.down_proj(h)

        if task_id is not None:
            if isinstance(task_id, str):
                idx = self.register_task(task_id)
            else:
                idx = task_id
            t_idx = torch.tensor(idx, device=h.device)
            e_t = self.task_embeddings(t_idx)  # (d_shared,)
            # Broadcast across batch and sequence dimensions: (1, 1, d_shared)
            z = z + e_t.view(1, 1, self.d_shared)

        z = self.norm(z)
        z = F.gelu(z)
        z_up = self.up_proj(z)
        logits = self.head(h + z_up)
        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters (trunk + task embeddings)."""
        return sum(p.numel() for p in self.parameters())

    def count_trunk_parameters(self) -> int:
        """Count parameters of the shared trunk excluding task embeddings."""
        return (
            sum(p.numel() for p in self.down_proj.parameters())
            + sum(p.numel() for p in self.norm.parameters())
            + sum(p.numel() for p in self.up_proj.parameters())
            + sum(p.numel() for p in self.head.parameters())
        )

    def count_task_embedding_parameters(self) -> int:
        """Count parameters of the task conditioning embeddings."""
        return sum(p.numel() for p in self.task_embeddings.parameters())

    def save_state(self) -> dict[str, Any]:
        """Save exact state for bit-exact rollback."""
        return {
            "state_dict": copy.deepcopy(self.state_dict()),
            "task_registry": dict(self.task_registry),
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore exact state for bit-exact rollback."""
        self.load_state_dict(state["state_dict"])
        self.task_registry = dict(state["task_registry"])
