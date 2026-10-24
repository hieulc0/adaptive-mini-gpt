"""Milestone M80 Phase 2: Evolvable Adapter Architecture (Rank-16 and Rank-8).

Defines:
- Rank16SequenceAdapter: Canonical 2,592-parameter bottleneck adapter.
- Rank8SequenceAdapter: Compact 1,432-parameter bottleneck adapter for capacity ablation.
- Weight shuffling, parameter hashing, and zero-residual projection utilities.
"""

from __future__ import annotations

import copy
import hashlib

import torch
import torch.nn as nn


class Rank8SequenceAdapter(nn.Module):
    """Ultra-compact 1,432-parameter rank-8 bottleneck adapter for procedural skills."""

    def __init__(
        self,
        d_model: int = 128,
        bottleneck_dim: int = 8,
        num_symbols: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.num_symbols = num_symbols
        self.ln = nn.LayerNorm(d_model)
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck_dim, num_symbols)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Map hidden states [B, L, d_model] -> symbol logits [B, L, num_symbols]."""
        x = self.ln(hidden_states)
        x = self.down_proj(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.up_proj(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def compute_parameter_hash(self) -> str:
        hasher = hashlib.sha256()
        for p in self.parameters():
            hasher.update(p.detach().cpu().contiguous().numpy().tobytes())
        return hasher.hexdigest()[:16]

    def shuffle_weights(self, seed: int = 42) -> Rank8SequenceAdapter:
        """Create a clone with permanently scrambled weight tensors."""
        shuffled = copy.deepcopy(self)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in shuffled.parameters():
                flat = p.flatten()
                perm = torch.randperm(flat.numel(), generator=g)
                p.copy_(flat[perm].view_as(p))
        return shuffled


class EvolvableSkillAdapterFactory:
    """Factory creating bounded residual adapters for autonomous skill acquisition."""

    @classmethod
    def create_adapter(cls, rank: int = 16, d_model: int = 128, num_symbols: int = 16) -> nn.Module:
        """Create bounded adapter of specified rank."""
        if rank == 16:
            from adaptive_mini_gpt.neural_weight_learning.sequence_adapter import (
                Rank16SequenceAdapter,
            )
            return Rank16SequenceAdapter(d_model=d_model, bottleneck_dim=16, num_symbols=num_symbols)
        elif rank == 8:
            return Rank8SequenceAdapter(d_model=d_model, bottleneck_dim=8, num_symbols=num_symbols)
        else:
            raise ValueError(f"Unsupported adapter rank: {rank}")
