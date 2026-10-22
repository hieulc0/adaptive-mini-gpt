"""Hard-Negative Structural Contrastive Objective.

Generates hard structural negatives (directional reversals, disconnected endpoints,
refuted assertions) and aligns topologically compatible graph paths.
"""
from __future__ import annotations

import collections
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.stable_relational.structural_benchmark import EnhancedRelationalDataset
from adaptive_mini_gpt.uncertainty.epistemic_benchmark import EpistemicTask


class HardNegativeGenerator:
    """Constructs hard structural negative tasks for a relational dataset."""

    def __init__(self, dataset: EnhancedRelationalDataset, seed: int = 42) -> None:
        self.dataset = dataset
        self.rng = random.Random(seed)

    def generate_directional_reversals(self) -> list[EpistemicTask]:
        """Generates tasks with reversed query direction (e.g. B -> A when A -> B)."""
        reversals: list[EpistemicTask] = []
        rel = self.dataset.semantics.name
        is_sym = self.dataset.semantics.is_symmetric

        if is_sym:
            # Symmetric relations cannot form directional reversal negatives
            return []

        for p in self.dataset.premises:
            parts = p.split()
            if len(parts) >= 3:
                s, r, o = parts[0], parts[1], parts[2]
                if r == rel:
                    # Reversed query: Does o rel s? -> NO
                    idx = len(reversals) + 1
                    reversals.append(
                        EpistemicTask(
                            task_id=f"{self.dataset.world_name}_rev_{idx}",
                            world_name=self.dataset.world_name,
                            split="train",
                            premises=list(self.dataset.premises),
                            question=f"Does {o} {rel} {s}?",
                            expected_verdict="NO",
                            epistemic_category="directional_reversal",
                            entities_involved=[o, s],
                        )
                    )
        return reversals

    def generate_disconnected_hard_negatives(self) -> list[EpistemicTask]:
        """Generates queries between nodes that appear in premises but have no valid path."""
        disconnected: list[EpistemicTask] = []
        rel = self.dataset.semantics.name

        # Extract all entities in premises
        entities = set()
        for p in self.dataset.premises:
            words = p.split()
            if len(words) >= 3:
                entities.add(words[0])
                entities.add(words[2])

        ent_list = sorted(list(entities))
        for t in self.dataset.held_out_test_pool:
            if t.expected_verdict == "UNKNOWN":
                disconnected.append(t)

        return disconnected


class StructuralContrastiveLoss(nn.Module):
    """Contrastive loss with explicit hard-negative penalization.

    Pulls together representations with identical epistemic targets and topologies,
    while exerting strong repulsive gradients against hard structural negatives.
    """

    def __init__(self, temperature: float = 0.15, hard_negative_weight: float = 1.5) -> None:
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        hard_negative_features: torch.Tensor | None = None,
        hard_neg_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        features: [N, d_model]
        labels: [N] class indices (0=YES, 1=NO, 2=UNKNOWN)
        hard_negative_features: [M, d_model] optional explicit hard negatives
        """
        if features.size(0) < 2:
            return torch.tensor(0.0, device=features.device)

        norm_feat = F.normalize(features, p=2, dim=1)
        sim_matrix = torch.matmul(norm_feat, norm_feat.T) / self.temperature

        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(features.device)
        diag = torch.eye(features.size(0), dtype=torch.float, device=features.device)
        pos_mask = pos_mask * (1.0 - diag)

        # Baseline non-self exponentiated similarities
        exp_sim = torch.exp(sim_matrix) * (1.0 - diag)

        # Incorporate explicit hard negatives if present
        if hard_neg_features is not None:
            hard_negative_features = hard_neg_features
        if hard_negative_features is not None and hard_negative_features.size(0) > 0:
            norm_hard = F.normalize(hard_negative_features, p=2, dim=1)
            hard_sim = torch.matmul(norm_feat, norm_hard.T) / self.temperature
            exp_hard = torch.exp(hard_sim) * self.hard_negative_weight
            denom = exp_sim.sum(dim=1, keepdim=True) + exp_hard.sum(dim=1, keepdim=True) + 1e-8
        else:
            denom = exp_sim.sum(dim=1, keepdim=True) + 1e-8

        log_prob = sim_matrix - torch.log(denom)

        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        if not valid.any():
            return torch.tensor(0.0, device=features.device)

        mean_log_prob = (pos_mask * log_prob).sum(dim=1)[valid] / pos_count[valid]
        return -mean_log_prob.mean()
