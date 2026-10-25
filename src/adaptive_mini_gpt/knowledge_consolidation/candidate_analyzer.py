"""Milestone M83 Phase 1: Consolidation Candidate Discovery and Representation Compatibility.

Implements principled multi-metric compatibility analysis across learned neural skills:
1. Weight Cosine Similarity: Normalized parameter tensor alignment.
2. Representation / Activation Similarity: Bottleneck latent feature alignment on probe batches.
3. Functional Output Agreement: Exact token-level prediction agreement rate.
4. Gradient Alignment: Directional cosine similarity of backpropagated adapter loss gradients.
5. Overall Compatibility Index: Multi-objective score guarding against using weight distance alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.learning_to_learn.capacity_adaptation import (
    BottleneckSequenceAdapter,
)

logger = logging.getLogger(__name__)


@dataclass
class SkillPairCompatibility:
    """Pairwise compatibility diagnostic across two learned skills."""

    skill_id_a: str
    skill_id_b: str
    weight_similarity: float
    representation_similarity: float
    functional_agreement: float
    gradient_alignment: float
    composite_compatibility_score: float
    is_redundant_candidate: bool
    is_conflicting_candidate: bool


class ConsolidationCandidateAnalyzer:
    """Analyzes pairs/groups of neural adapters to discover redundancy and interference."""

    @classmethod
    def compute_weight_similarity(cls, adapter_a: nn.Module, adapter_b: nn.Module) -> float:
        """Compute cosine similarity of flattened adapter parameter vectors."""
        vec_a = torch.cat([p.detach().flatten() for p in adapter_a.parameters() if p.requires_grad])
        vec_b = torch.cat([p.detach().flatten() for p in adapter_b.parameters() if p.requires_grad])

        if vec_a.shape != vec_b.shape:
            # Different ranks: slice down to minimum common shape
            min_len = min(vec_a.numel(), vec_b.numel())
            vec_a = vec_a[:min_len]
            vec_b = vec_b[:min_len]

        norm_a = torch.linalg.norm(vec_a)
        norm_b = torch.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(torch.dot(vec_a, vec_b) / (norm_a * norm_b))

    @classmethod
    def compute_representation_similarity(
        cls,
        adapter_a: BottleneckSequenceAdapter,
        adapter_b: BottleneckSequenceAdapter,
        probe_h: torch.Tensor,
    ) -> float:
        """Compute latent activation cosine similarity at bottleneck layer."""
        adapter_a.eval()
        adapter_b.eval()
        with torch.no_grad():
            # [B, L, d_model] -> [B, L, rank]
            act_a = adapter_a.act(adapter_a.down_proj(adapter_a.ln(probe_h)))
            act_b = adapter_b.act(adapter_b.down_proj(adapter_b.ln(probe_h)))

            flat_a = act_a.flatten()
            flat_b = act_b.flatten()

            if flat_a.shape != flat_b.shape:
                min_len = min(flat_a.numel(), flat_b.numel())
                flat_a = flat_a[:min_len]
                flat_b = flat_b[:min_len]

            norm_a = torch.linalg.norm(flat_a)
            norm_b = torch.linalg.norm(flat_b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(torch.dot(flat_a, flat_b) / (norm_a * norm_b))

    @classmethod
    def compute_functional_agreement(
        cls,
        adapter_a: nn.Module,
        adapter_b: nn.Module,
        probe_h: torch.Tensor,
    ) -> float:
        """Compute proportion of token-level predictions that agree on probe inputs."""
        adapter_a.eval()
        adapter_b.eval()
        with torch.no_grad():
            preds_a = torch.argmax(adapter_a(probe_h), dim=-1)
            preds_b = torch.argmax(adapter_b(probe_h), dim=-1)
            agreement = (preds_a == preds_b).float().mean().item()
            return float(agreement)

    @classmethod
    def compute_gradient_alignment(
        cls,
        adapter_a: nn.Module,
        adapter_b: nn.Module,
        probe_h: torch.Tensor,
        probe_y_a: torch.Tensor,
        probe_y_b: torch.Tensor,
    ) -> float:
        """Compute cosine similarity of gradients on probe loss to detect conflict/synergy."""
        adapter_a.eval()
        adapter_b.eval()

        # Compute gradient for A
        out_a = adapter_a(probe_h)
        loss_a = F.cross_entropy(out_a.view(-1, 16), probe_y_a.view(-1))
        grads_a = torch.autograd.grad(loss_a, list(adapter_a.parameters()), retain_graph=False, create_graph=False)
        flat_g_a = torch.cat([g.flatten() for g in grads_a])

        # Compute gradient for B
        out_b = adapter_b(probe_h)
        loss_b = F.cross_entropy(out_b.view(-1, 16), probe_y_b.view(-1))
        grads_b = torch.autograd.grad(loss_b, list(adapter_b.parameters()), retain_graph=False, create_graph=False)
        flat_g_b = torch.cat([g.flatten() for g in grads_b])

        if flat_g_a.shape != flat_g_b.shape:
            min_len = min(flat_g_a.numel(), flat_g_b.numel())
            flat_g_a = flat_g_a[:min_len]
            flat_g_b = flat_g_b[:min_len]

        norm_a = torch.linalg.norm(flat_g_a)
        norm_b = torch.linalg.norm(flat_g_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(torch.dot(flat_g_a, flat_g_b) / (norm_a * norm_b))

    @classmethod
    def analyze_pair(
        cls,
        skill_id_a: str,
        adapter_a: BottleneckSequenceAdapter,
        skill_id_b: str,
        adapter_b: BottleneckSequenceAdapter,
        probe_h: torch.Tensor,
        probe_y_a: torch.Tensor,
        probe_y_b: torch.Tensor,
        redundancy_threshold: float = 0.50,
        conflict_threshold: float = -0.15,
    ) -> SkillPairCompatibility:
        """Perform comprehensive multi-metric compatibility evaluation."""
        w_sim = cls.compute_weight_similarity(adapter_a, adapter_b)
        rep_sim = cls.compute_representation_similarity(adapter_a, adapter_b, probe_h)
        func_agree = cls.compute_functional_agreement(adapter_a, adapter_b, probe_h)
        grad_align = cls.compute_gradient_alignment(adapter_a, adapter_b, probe_h, probe_y_a, probe_y_b)

        # Composite compatibility score:
        # Heavily weights activation alignment and gradient synergy, guarding against weight distance alone
        composite = (
            0.35 * max(rep_sim, 0.0)
            + 0.35 * max(grad_align, 0.0)
            + 0.20 * func_agree
            + 0.10 * max(w_sim, 0.0)
        )

        is_redundant = (composite >= redundancy_threshold) and (grad_align > -0.05)
        is_conflicting = (grad_align < conflict_threshold) or (func_agree < 0.10 and rep_sim > 0.50)

        return SkillPairCompatibility(
            skill_id_a=skill_id_a,
            skill_id_b=skill_id_b,
            weight_similarity=round(w_sim, 4),
            representation_similarity=round(rep_sim, 4),
            functional_agreement=round(func_agree, 4),
            gradient_alignment=round(grad_align, 4),
            composite_compatibility_score=round(composite, 4),
            is_redundant_candidate=is_redundant,
            is_conflicting_candidate=is_conflicting,
        )
