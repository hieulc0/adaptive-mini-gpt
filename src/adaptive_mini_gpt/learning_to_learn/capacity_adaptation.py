"""Milestone M81 Phase 3: Minimal Capacity Adaptation and Bounded Rank Expansion.

Implements principled, cost-conscious architectural capacity selection:
- No new allocation (0 parameters): reuse existing primitive if qualification EM >= 0.85.
- Rank-4 adapter (852 parameters): ultra-compact candidate.
- Rank-8 adapter (1,432 parameters): intermediate candidate.
- Rank-16 adapter (2,592 parameters): standard bounded residual candidate.
- Bounded rank expansion: searches smallest candidate first (Rank-4 -> Rank-8 -> Rank-16).
- Candidate pruning and rollback: bit-exact rejection if qualification threshold is unmet.
"""

from __future__ import annotations

import copy
import hashlib
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

QUALIFICATION_ADMISSION_EM: float = 0.85


class CapacityDecision(StrEnum):
    """Categorical architectural decision made during capacity adaptation."""

    REUSE_EXISTING = "REUSE_EXISTING"
    ALLOCATE_RANK_4 = "ALLOCATE_RANK_4"
    ALLOCATE_RANK_8 = "ALLOCATE_RANK_8"
    ALLOCATE_RANK_16 = "ALLOCATE_RANK_16"
    EXPAND_AND_ALLOCATE = "EXPAND_AND_ALLOCATE"
    REJECT_AND_ROLLBACK = "REJECT_AND_ROLLBACK"


class BottleneckSequenceAdapter(nn.Module):
    """Parametric bottleneck adapter supporting arbitrary ranks (e.g. 4, 8, 16)."""

    def __init__(
        self,
        d_model: int = 128,
        bottleneck_dim: int = 4,
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
        """Map [B, L, d_model] -> [B, L, num_symbols]."""
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

    def shuffle_weights(self, seed: int = 42) -> BottleneckSequenceAdapter:
        """Create an identical clone with scrambled weight tensors for causal verification."""
        shuffled = copy.deepcopy(self)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in shuffled.parameters():
                flat = p.flatten()
                perm = torch.randperm(flat.numel(), generator=g)
                p.copy_(flat[perm].view_as(p))
        return shuffled


@dataclass
class CapacityTrialResult:
    """Outcome of capacity evaluation trial."""

    decision: CapacityDecision
    selected_rank: int
    parameters_allocated: int
    qualification_em: float
    expansion_history: list[dict[str, Any]]
    adapter: nn.Module | None
    parameter_hash: str
    repertoire_reused_skill_id: str | None = None


class CapacityAdaptationManager:
    """Manages hierarchical minimal capacity selection, bounded expansion, and rollback."""

    @classmethod
    def evaluate_capacity_hierarchy(
        cls,
        eval_fn: Any,  # Callable[[int], tuple[float, nn.Module]] -> (qual_em, trained_adapter)
        repertoire_check_fn: Any | None = None,
        candidate_ranks: list[int] | None = None,
        admission_threshold: float = QUALIFICATION_ADMISSION_EM,
    ) -> CapacityTrialResult:
        """Evaluate minimal capacity choices starting from reuse down to rank expansion.

        Steps:
        1. Reuse existing model without new weights (if repertoire check succeeds).
        2. Evaluate smallest candidate rank (Rank-4: 852 params).
        3. If qualification threshold met, accept Rank-4.
        4. If insufficient, expand to Rank-8 (1,432 params).
        5. If insufficient, expand to Rank-16 (2,592 params).
        6. If all candidates fail admission, reject and rollback cleanly.
        """
        if candidate_ranks is None:
            candidate_ranks = [4, 8, 16]

        expansion_history: list[dict[str, Any]] = []

        # Step 1: Reuse Check
        if repertoire_check_fn is not None:
            reused_id, reuse_em = repertoire_check_fn()
            if reused_id is not None and reuse_em >= admission_threshold:
                logger.info(
                    "Capacity Decision: REUSE existing skill '%s' (EM=%.4f, 0 parameters added)",
                    reused_id,
                    reuse_em,
                )
                return CapacityTrialResult(
                    decision=CapacityDecision.REUSE_EXISTING,
                    selected_rank=0,
                    parameters_allocated=0,
                    qualification_em=reuse_em,
                    expansion_history=[{"action": "REUSE", "skill_id": reused_id, "em": reuse_em}],
                    adapter=None,
                    parameter_hash="0000000000000000",
                    repertoire_reused_skill_id=reused_id,
                )

        # Step 2: Bounded Rank Expansion Search (minimal capacity preference)
        for idx, rank in enumerate(candidate_ranks):
            logger.info("Evaluating capacity allocation for Rank-%d...", rank)
            qual_em, adapter = eval_fn(rank)
            param_count = adapter.count_parameters()
            expansion_history.append(
                {
                    "rank": rank,
                    "parameters": param_count,
                    "qualification_em": round(qual_em, 4),
                    "passed_threshold": bool(qual_em >= admission_threshold),
                }
            )

            if qual_em >= admission_threshold:
                decision = (
                    CapacityDecision.ALLOCATE_RANK_4
                    if rank == 4
                    else (
                        CapacityDecision.ALLOCATE_RANK_8
                        if rank == 8
                        else (
                            CapacityDecision.ALLOCATE_RANK_16
                            if idx == 0
                            else CapacityDecision.EXPAND_AND_ALLOCATE
                        )
                    )
                )
                logger.info(
                    "Accepted minimal capacity: Rank-%d (%d parameters, EM=%.4f >= %.2f)",
                    rank,
                    param_count,
                    qual_em,
                    admission_threshold,
                )
                return CapacityTrialResult(
                    decision=decision,
                    selected_rank=rank,
                    parameters_allocated=param_count,
                    qualification_em=round(qual_em, 4),
                    expansion_history=expansion_history,
                    adapter=adapter,
                    parameter_hash=adapter.compute_parameter_hash(),
                )

            logger.warning(
                "Rank-%d failed admission gate (EM=%.4f < %.2f). Expanding capacity...",
                rank,
                qual_em,
                admission_threshold,
            )

        # Step 3: Rejection & Rollback if no candidate satisfied the gate
        logger.error("All candidate adapter ranks failed admission gate. Rejecting and rolling back.")
        return CapacityTrialResult(
            decision=CapacityDecision.REJECT_AND_ROLLBACK,
            selected_rank=0,
            parameters_allocated=0,
            qualification_em=expansion_history[-1]["qualification_em"] if expansion_history else 0.0,
            expansion_history=expansion_history,
            adapter=None,
            parameter_hash="REJECTED_ROLLBACK",
        )
