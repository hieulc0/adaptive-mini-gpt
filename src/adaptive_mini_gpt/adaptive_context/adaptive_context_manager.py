"""Adaptive Context Selection Policy for Milestone M97.

Implements:
1. Dynamic context mode routing:
   - SHORT_NATIVE (128 tokens, 0 retrieval)
   - SHORT_RETRIEVAL (128 tokens + 1-hop factual retrieval)
   - LONG_NATIVE (512-1024 tokens, procedural traces)
   - LONG_RETRIEVAL (512-1024 tokens + multi-hop factual retrieval)
   - ABSTAIN (unsupported, invalid syntax, or contradictory facts)
2. Cost-Aware Budget Optimization:
   - Evaluates input token length, entity reference density, and reasoning depth.
   - Selects minimal sufficient context window to save compute while achieving high accuracy.
   - Tracks attention compute cost proxy: Cost ~ (SeqLen)^2.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ContextMode(str, Enum):
    """Execution context modes for the adaptive student."""

    SHORT_NATIVE = "short_native"          # 128 tokens, no retrieval
    SHORT_RETRIEVAL = "short_retrieval"    # 128 tokens, memory lookup
    LONG_NATIVE = "long_native"            # 512-1024 tokens, no retrieval
    LONG_RETRIEVAL = "long_retrieval"      # 512-1024 tokens + multi-hop memory
    ABSTAIN = "abstain"                    # ungrounded / contradictory


@dataclass(frozen=True)
class ContextDecision:
    """Decision produced by the Adaptive Context Policy."""

    mode: ContextMode
    allocated_context_length: int
    enable_memory_retrieval: bool
    is_multi_hop: bool
    estimated_compute_cost: float  # Normalized attention quadratic cost proxy
    decision_reason: str


class AdaptiveContextPolicy:
    """Minimal, cost-aware policy deciding the optimal context configuration for a task."""

    def __init__(
        self,
        base_short_len: int = 128,
        extended_long_len: int = 1024,
    ) -> None:
        self.base_short_len = base_short_len
        self.extended_long_len = extended_long_len

    def evaluate_query(
        self,
        input_text: str,
        has_supported_entity: bool = True,
        is_contradictory: bool = False,
        requires_retrieval: bool = False,
    ) -> ContextDecision:
        """Decide the minimal sufficient context length and retrieval configuration."""
        if not has_supported_entity or is_contradictory:
            return ContextDecision(
                mode=ContextMode.ABSTAIN,
                allocated_context_length=0,
                enable_memory_retrieval=False,
                is_multi_hop=False,
                estimated_compute_cost=0.0,
                decision_reason="Epistemic abstention triggered: entity unsupported or contradictory.",
            )

        input_tokens = len(input_text.split())
        is_long_input = input_tokens > self.base_short_len

        # Detect multi-hop reasoning indicators
        lower_q = input_text.lower()
        multi_hop_keywords = ["founder of", "parent of", "derived from", "funding of", "step by step"]
        is_multi_hop = any(kw in lower_q for kw in multi_hop_keywords)

        if not requires_retrieval:
            if is_long_input:
                allocated = self.extended_long_len
                mode = ContextMode.LONG_NATIVE
                cost = (allocated / self.base_short_len) ** 2
                reason = "Long procedural input detected without external factual retrieval."
            else:
                allocated = self.base_short_len
                mode = ContextMode.SHORT_NATIVE
                cost = 1.0
                reason = "Short procedural query solved via native 128-token context."
        else:
            if is_long_input or is_multi_hop:
                allocated = self.extended_long_len
                mode = ContextMode.LONG_RETRIEVAL
                cost = (allocated / self.base_short_len) ** 2
                reason = "Complex multi-hop or long-context task requiring extended working memory."
            else:
                allocated = self.base_short_len
                mode = ContextMode.SHORT_RETRIEVAL
                cost = 1.0
                reason = "Single-hop factual query fits within compact 128-token context."

        return ContextDecision(
            mode=mode,
            allocated_context_length=allocated,
            enable_memory_retrieval=requires_retrieval,
            is_multi_hop=is_multi_hop,
            estimated_compute_cost=cost,
            decision_reason=reason,
        )
