"""Milestone M89 Phase 1: Representation Capacity Feasibility Benchmark.

Tests whether a compact shared neural architecture has sufficient capacity to learn
and hold 5-10 foundational skills simultaneously before attempting sequential continual learning.
Gates:
- Joint multi-task training must reach >= 85.0% held-out exact match.
- If joint training fails, the failure is diagnosed as representation/optimization limits
  rather than continual forgetting.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.continual_retention.curriculum_m89 import (
    M89CurriculumGenerator,
    RetentionDataset,
    extract_retention_features,
)
from adaptive_mini_gpt.continual_retention.task_conditioning import (
    TaskConditionedKnowledgeTrunk,
)
from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.neural_weight_learning.sequence_adapter import (
    Rank16SequenceAdapter,
)


@dataclass(frozen=True)
class CapacityFeasibilityOutcome:
    """Outcomes of the Phase 1 Capacity Feasibility experiments."""

    independent_mean_em: float
    joint_multi_task_mean_em: float
    sequential_unregularized_mean_em: float
    capacity_feasibility_gate_passed: bool
    joint_to_independent_ratio: float
    per_task_joint_ems: list[float]
    per_task_independent_ems: list[float]
    total_joint_parameters: int


class CapacityFeasibilityBenchmark:
    """Executes Phase 1 capacity feasibility assessment."""

    @classmethod
    def evaluate_capacity_feasibility(
        cls,
        model: MiniGPT,
        tasks: list[RetentionDataset],
        device: torch.device,
        epochs: int = 50,
        lr: float = 0.015,
    ) -> CapacityFeasibilityOutcome:
        """Benchmark Independent Adapters vs Joint Multi-Task vs Sequential Unregularized."""
        task_data = []
        for t in tasks:
            tr_h, tr_y = extract_retention_features(model, t.train_examples, device)
            ql_h, ql_y = extract_retention_features(model, t.qualification_examples, device)
            task_data.append((t.task_id, tr_h, tr_y, ql_h, ql_y))

        num_tasks = len(tasks)

        # 1. Independent Adapters Upper Bound
        indep_ems = []
        for t_id, tr_h, tr_y, ql_h, ql_y in task_data:
            adapter = Rank16SequenceAdapter(d_model=128, bottleneck_dim=16, num_symbols=16).to(device)
            opt = torch.optim.AdamW(adapter.parameters(), lr=lr)
            for _ in range(epochs):
                logits = adapter(tr_h)
                loss = F.cross_entropy(logits.view(-1, 16), tr_y.view(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
            adapter.eval()
            with torch.no_grad():
                preds = adapter(ql_h).argmax(dim=-1)
                em = (preds == ql_y).all(dim=-1).float().mean().item()
                indep_ems.append(em)
        indep_mean = sum(indep_ems) / len(indep_ems)

        # 2. Joint Multi-Task Training (Shared Task-Conditioned Trunk)
        joint_trunk = TaskConditionedKnowledgeTrunk(
            max_tasks=max(10, num_tasks), d_model=128, d_shared=16, num_symbols=16
        ).to(device)
        opt_joint = torch.optim.AdamW(joint_trunk.parameters(), lr=lr)

        for _ in range(epochs):
            for t_id, tr_h, tr_y, _, _ in task_data:
                logits = joint_trunk(tr_h, task_id=t_id)
                loss = F.cross_entropy(logits.view(-1, 16), tr_y.view(-1))
                opt_joint.zero_grad()
                loss.backward()
                opt_joint.step()

        joint_trunk.eval()
        joint_ems = []
        with torch.no_grad():
            for t_id, _, _, ql_h, ql_y in task_data:
                preds = joint_trunk(ql_h, task_id=t_id).argmax(dim=-1)
                em = (preds == ql_y).all(dim=-1).float().mean().item()
                joint_ems.append(em)
        joint_mean = sum(joint_ems) / len(joint_ems)

        # 3. Sequential Unregularized (Vanilla Sequential)
        seq_trunk = TaskConditionedKnowledgeTrunk(
            max_tasks=max(10, num_tasks), d_model=128, d_shared=16, num_symbols=16
        ).to(device)

        for t_id, tr_h, tr_y, _, _ in task_data:
            opt_seq = torch.optim.AdamW(seq_trunk.parameters(), lr=lr)
            for _ in range(epochs):
                logits = seq_trunk(tr_h, task_id=t_id)
                loss = F.cross_entropy(logits.view(-1, 16), tr_y.view(-1))
                opt_seq.zero_grad()
                loss.backward()
                opt_seq.step()

        seq_trunk.eval()
        seq_ems = []
        with torch.no_grad():
            for t_id, _, _, ql_h, ql_y in task_data:
                preds = seq_trunk(ql_h, task_id=t_id).argmax(dim=-1)
                em = (preds == ql_y).all(dim=-1).float().mean().item()
                seq_ems.append(em)
        seq_mean = sum(seq_ems) / len(seq_ems)

        gate_passed = joint_mean >= 0.85
        ratio = joint_mean / max(1e-4, indep_mean)

        return CapacityFeasibilityOutcome(
            independent_mean_em=indep_mean,
            joint_multi_task_mean_em=joint_mean,
            sequential_unregularized_mean_em=seq_mean,
            capacity_feasibility_gate_passed=gate_passed,
            joint_to_independent_ratio=ratio,
            per_task_joint_ems=joint_ems,
            per_task_independent_ems=indep_ems,
            total_joint_parameters=joint_trunk.count_parameters(),
        )
