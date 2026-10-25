"""Milestone M89 Phase 2: Gradient Conflict and Representation Drift Diagnostics.

Measures the mathematical causes of forgetting during continual learning:
1. Task Gradient Vector Extraction:
   - Flattens gradients of all trainable parameters into a single vector g_t in R^P.
2. Pairwise Gradient Cosine Similarity Matrix:
   - S_{i, j} = <g_i, g_j> / (||g_i|| * ||g_j||).
   - Identifies whether task objectives are complementary (S > 0), orthogonal (S ~ 0), or conflicting (S < 0).
3. Representation Drift Metric:
   - Delta_rep(t) = ||mu_t(H) - mu_0(H)||_2 / ||mu_0(H)||_2, measuring shift in hidden layer activations.
4. Effective Dimensionality & Activation Rank:
   - Rank via singular value entropy: exp(-sum p_i log p_i) where p_i = s_i / sum(s_j).
5. Granular Token-Level Accuracy vs Sequence Exact Match:
   - Isolates whether errors are single-token slips or complete sequence divergence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GradientDiagnosticReport:
    """Diagnostic profile characterizing gradient conflict and representation stability."""

    task_count: int
    mean_pairwise_gradient_similarity: float
    min_pairwise_gradient_similarity: float
    max_pairwise_gradient_similarity: float
    gradient_conflict_pair_count: int
    mean_representation_drift: float
    mean_effective_dimension: float
    token_accuracy_mean: float
    sequence_exact_match_mean: float


class GradientConflictAuditor:
    """Audits and quantifies gradient conflict and representational drift across tasks."""

    @staticmethod
    def extract_flat_gradient(model: nn.Module) -> torch.Tensor:
        """Extract all parameter gradients into a flattened 1D tensor."""
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().view(-1))
            else:
                grads.append(torch.zeros(p.numel(), device=p.device))
        return torch.cat(grads)

    @staticmethod
    def compute_gradient_cosine_similarity_matrix(
        task_gradients: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the N x N pairwise gradient cosine similarity matrix."""
        N = len(task_gradients)
        sim_matrix = torch.zeros(N, N)
        for i in range(N):
            gi = task_gradients[i]
            norm_i = torch.norm(gi) + 1e-8
            for j in range(N):
                gj = task_gradients[j]
                norm_j = torch.norm(gj) + 1e-8
                sim_matrix[i, j] = (torch.dot(gi, gj) / (norm_i * norm_j)).item()
        return sim_matrix

    @staticmethod
    def compute_effective_dimension(h: torch.Tensor) -> float:
        """Compute effective dimensionality using participation ratio of singular values."""
        # h: (B, L, D) -> (B * L, D)
        flat_h = h.view(-1, h.size(-1)).float()
        try:
            _, s, _ = torch.svd(flat_h - flat_h.mean(dim=0, keepdim=True))
            s_sq = s**2
            sum_sq = torch.sum(s_sq)
            if sum_sq <= 1e-8:
                return 1.0
            p = s_sq / sum_sq
            # Shannon entropy of normalized singular values
            entropy = -torch.sum(p * torch.log(p + 1e-12)).item()
            return math.exp(entropy)
        except Exception:
            return float(flat_h.size(-1))

    @staticmethod
    def compute_token_and_sequence_metrics(
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[float, float]:
        """Compute both token-level accuracy and sequence-level exact match."""
        preds = logits.argmax(dim=-1)
        token_acc = (preds == targets).float().mean().item()
        seq_em = (preds == targets).all(dim=-1).float().mean().item()
        return token_acc, seq_em
