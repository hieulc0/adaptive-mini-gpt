"""Milestone M103-G.1 Baselines for Causal Prediction.

Implements non-learning and matched baseline models:
1. MajorityClassBaseline: Always predicts the empirical mode of the training distribution.
2. AttributeOnlyHeuristicBaseline: Deterministic attribute lookup without representation learning.
3. NearestNeighborBaseline: Non-parametric 1-NN based on attribute Jaccard overlap.
4. MatchedLinearProbeBaseline: Parametric linear model mapping multi-hot attribute + action to state logits.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from adaptive_mini_gpt.natural_language_learning_m103.causal_world_m103_g1 import (
    CausalItemM103_G1,
    CausalObjectM103_G1,
)


@dataclass
class BaselineEvaluationReportM103_G1:
    baseline_name: str
    train_acc: float
    val_held_out_pair_acc: float
    novel_entity_acc: float
    multistep_acc: float
    counterfactual_acc: float
    mean_held_out_acc: float
    sample_predictions: list[dict[str, Any]]


class MajorityClassBaselineM103_G1:
    """Predicts the most frequent state observed in training."""

    def __init__(self) -> None:
        self.majority_state: str = "Dormant"

    def fit(self, train_items: list[CausalItemM103_G1]) -> None:
        counts = Counter(it.expected_state for it in train_items)
        if counts:
            self.majority_state = counts.most_common(1)[0][0]

    def predict(self, item: CausalItemM103_G1, obj: CausalObjectM103_G1 | None = None) -> str:
        return self.majority_state


class AttributeOnlyHeuristicBaselineM103_G1:
    """Handcrafted heuristic mapping primary attributes to standard states."""

    def predict(self, item: CausalItemM103_G1, obj: CausalObjectM103_G1 | None = None) -> str:
        tags = obj.tags if obj is not None else set()
        act = item.action_name

        if item.is_counterfactual:
            if act == "Apply_Fire":
                return "Ash" if ("flammable" in tags or "organic" in tags) else "Tempered"
            if act == "Apply_Spark":
                return "Dimmed" if ("magical" in tags or "radiant" in tags or "relic" in tags) else "Dormant"
            if act == "Apply_Frost":
                return "Brittle" if "heat_tolerant" in tags else "Frozen"
            return "Dormant"

        if act == "Apply_Fire":
            if "flammable" in tags or "organic" in tags:
                return "Ignited"
            if "metal" in tags or "clay" in tags or "heat_tolerant" in tags:
                return "Tempered"
            if "crystalline" in tags or "glass" in tags or "optical" in tags:
                return "Vitrified"
            return "Dormant"

        if act == "Apply_Frost":
            if item.second_action == "Apply_Frost" and item.action_name == "Apply_Fire":
                return "Extinguished" if "flammable" in tags else "Brittle"
            return "Frozen"

        if act == "Apply_Spark":
            if "conductive" in tags or "metal" in tags:
                return "Charged"
            if "crystalline" in tags or "optical" in tags:
                return "Resonating"
            if "magical" in tags or "relic" in tags:
                return "Awakened"
            return "Dormant"

        if act == "Apply_Solvent":
            if "container" in tags or "vessel" in tags:
                return "Purified"
            if "cloth" in tags or "organic" in tags:
                return "Dissolved"
            return "Dormant"

        if act == "Apply_Essence":
            if "magical" in tags or "relic" in tags or "inscribed" in tags:
                return "Awakened"
            return "Resonating"

        return "Dormant"


class NearestNeighborBaselineM103_G1:
    """Jaccard distance nearest neighbor over training items with matching action."""

    def __init__(self) -> None:
        self.memory: list[tuple[set[str], str, str]] = []  # (tags, action, expected_state)
        self.fallback_state: str = "Dormant"

    def fit(self, train_items: list[CausalItemM103_G1], objects: dict[str, CausalObjectM103_G1]) -> None:
        self.memory = []
        for it in train_items:
            tags = objects[it.object_name].tags if it.object_name in objects else set()
            self.memory.append((tags, it.action_name, it.expected_state))
        if train_items:
            counts = Counter(it.expected_state for it in train_items)
            self.fallback_state = counts.most_common(1)[0][0]

    def predict(self, item: CausalItemM103_G1, obj: CausalObjectM103_G1 | None = None) -> str:
        tags = obj.tags if obj is not None else set()
        best_score = -1.0
        best_state = self.fallback_state

        for m_tags, m_act, m_state in self.memory:
            if m_act != item.action_name:
                continue
            intersection = len(tags & m_tags)
            union = len(tags | m_tags)
            score = (intersection / union) if union > 0 else 0.0
            if score > best_score:
                best_score = score
                best_state = m_state

        return best_state


class MatchedLinearProbeBaselineM103_G1(nn.Module):
    """Linear classifier mapping multi-hot tags + one-hot action to state logits."""

    def __init__(self, num_tags: int = 24, num_actions: int = 8, num_states: int = 18) -> None:
        super().__init__()
        in_dim = num_tags + num_actions
        self.linear = nn.Linear(in_dim, num_states)

    def forward(self, tag_multi_hot: torch.Tensor, act_one_hot: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([tag_multi_hot, act_one_hot], dim=-1)
        return self.linear(feat)
