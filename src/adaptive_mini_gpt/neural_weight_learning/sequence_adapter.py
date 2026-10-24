"""Phase 2 & 8: Compact Rank-16 Sequence Adapter Architecture & Rank-32 Diagnostic.

Implements:
- Rank16SequenceAdapter: 2,592-parameter compact bottleneck adapter mapping Transformer hidden
  states at sequence positions to output symbol distributions.
- Rank32SequenceAdapter: 4,912-parameter diagnostic capacity variant for Phase 8.
- Exact parameter tracking, bit-exact parameter hashing, and weight shuffling utilities.
"""

from __future__ import annotations

import hashlib

import torch
import torch.nn as nn


class Rank16SequenceAdapter(nn.Module):
    """Canonical 2,592-parameter rank-16 bottleneck adapter for procedural sequence mapping."""

    def __init__(
        self,
        d_model: int = 128,
        bottleneck_dim: int = 16,
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

    def shuffle_weights(self, seed: int = 42) -> Rank16SequenceAdapter:
        """Create a clone with permanently shuffled weight tensors (for causal ablation)."""
        import copy

        shuffled = copy.deepcopy(self)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in shuffled.parameters():
                flat = p.flatten()
                perm = torch.randperm(flat.numel(), generator=g)
                p.copy_(flat[perm].view_as(p))
        return shuffled

    def reset_weights(self, seed: int = 42) -> Rank16SequenceAdapter:
        """Create a fresh randomly initialized clone with identical architecture."""
        torch.manual_seed(seed)
        fresh = Rank16SequenceAdapter(
            d_model=self.d_model,
            bottleneck_dim=self.bottleneck_dim,
            num_symbols=self.num_symbols,
        )
        return fresh


class Rank32SequenceAdapter(nn.Module):
    """4,912-parameter rank-32 diagnostic capacity variant (Phase 8)."""

    def __init__(
        self,
        d_model: int = 128,
        bottleneck_dim: int = 32,
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
