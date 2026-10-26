"""Milestone M90 Stage A: Shared Representation Capacity Feasibility Sweep.

Phase 1 Gate A Assessment:
Evaluates whether a shared low-rank neural bottleneck has sufficient capacity to learn
six foundational operations jointly before proceeding to instruction-conditioned transfer.

Sweep configurations:
- Rank 16: 6,336 parameters (trunk + task conditioning embeddings)
- Rank 32: 10,624 parameters
- Rank 48: 14,912 parameters
All strictly <= 20,000 parameter budget.

Evaluates:
- Joint multi-task training exact match
- Sequential unregularized training exact match
- Budget-matched modular reference (independent adapters)
- Verification of Gate A: Joint held-out EM >= 85.0%
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from adaptive_mini_gpt.continual_retention.curriculum_m89 import (
    M89CurriculumGenerator,
    extract_retention_features,
)
from adaptive_mini_gpt.continual_retention.task_conditioning import (
    TaskConditionedKnowledgeTrunk,
)
from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.neural_weight_learning.sequence_adapter import (
    Rank16SequenceAdapter,
)
from adaptive_mini_gpt.training.optimizer_contract import adamw_groups
from adaptive_mini_gpt.training.reproducibility import isolated_rng, stream_seed


@dataclass(frozen=True)
class RankFeasibilityResult:
    """Outcomes for a specific bottleneck rank configuration."""

    rank: int
    trainable_parameters: int
    budget_compliant: bool
    joint_qualification_em: float
    joint_test_em: float
    per_task_joint_ems: list[float]
    sequential_qualification_em: float
    modular_reference_em: float
    gate_a_passed: bool


@dataclass(frozen=True)
class StageACapacityReport:
    """Comprehensive Stage A feasibility sweep report."""

    ranks_evaluated: list[int]
    results_by_rank: dict[int, RankFeasibilityResult]
    gate_a_overall_passed: bool
    selected_rank: int
    modular_reference_em: float
    initialization_seeds: dict[str, int] = field(default_factory=dict)


class StageACapacityBenchmark:
    """Stage A Capacity Sweep and Gate A Feasibility Evaluator."""

    @classmethod
    def evaluate_capacity_sweep(
        cls,
        model: MiniGPT,
        device: torch.device,
        ranks: list[int] | None = None,
        num_tasks: int = 6,
        seed: int = 42,
        epochs: int = 60,
        lr: float = 0.015,
        selected_rank: int = 32,
    ) -> StageACapacityReport:
        """Execute Stage A capacity sweep across bottleneck dimensions."""
        if ranks is None:
            ranks = [16, 32, 48]

        curriculum_gen = M89CurriculumGenerator()
        tasks = curriculum_gen.generate_curriculum(num_tasks=num_tasks, seed=seed)

        # Pre-extract features for all 6 tasks
        task_data = []
        for t in tasks:
            tr_h, tr_y = extract_retention_features(model, t.train_examples, device)
            ql_h, ql_y = extract_retention_features(model, t.qualification_examples, device)
            te_h, te_y = extract_retention_features(model, t.sealed_test_examples, device)
            task_data.append((t.task_id, tr_h, tr_y, ql_h, ql_y, te_h, te_y))

        initialization_seeds = {}

        # 1. Modular Reference (Independent rank-16 adapters)
        indep_ems = []
        for task_id, tr_h, tr_y, _ql_h, _ql_y, te_h, te_y in task_data:
            name = f"reference:{task_id}"
            initialization_seeds[name] = stream_seed(seed, name)
            with isolated_rng(seed, name):
                adapter = Rank16SequenceAdapter(d_model=128, bottleneck_dim=16, num_symbols=16).to(
                    device
                )
            opt = torch.optim.AdamW(adamw_groups(adapter, 0.01), lr=lr)
            for _ in range(epochs):
                logits = adapter(tr_h)
                loss = F.cross_entropy(logits.view(-1, 16), tr_y.view(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
            adapter.eval()
            with torch.no_grad():
                preds = adapter(te_h).argmax(dim=-1)
                em = (preds == te_y).all(dim=-1).float().mean().item() * 100
                indep_ems.append(em)
        modular_ref_em = sum(indep_ems) / len(indep_ems)

        results_by_rank: dict[int, RankFeasibilityResult] = {}
        for rank in ranks:
            # Joint Multi-Task Training
            name = f"joint:{rank}"
            initialization_seeds[name] = stream_seed(seed, name)
            with isolated_rng(seed, name):
                joint_trunk = TaskConditionedKnowledgeTrunk(
                    max_tasks=max(10, num_tasks),
                    d_model=128,
                    d_shared=rank,
                    num_symbols=16,
                ).to(device)

            opt_joint = torch.optim.AdamW(adamw_groups(joint_trunk, 0.01), lr=lr)
            for _ in range(epochs):
                for t_id, tr_h, tr_y, _, _, _, _ in task_data:
                    logits = joint_trunk(tr_h, task_id=t_id)
                    loss = F.cross_entropy(logits.view(-1, 16), tr_y.view(-1))
                    opt_joint.zero_grad()
                    loss.backward()
                    opt_joint.step()

            joint_trunk.eval()
            joint_ql_ems = []
            joint_te_ems = []
            with torch.no_grad():
                for t_id, _, _, ql_h, ql_y, te_h, te_y in task_data:
                    p_ql = joint_trunk(ql_h, task_id=t_id).argmax(dim=-1)
                    em_ql = (p_ql == ql_y).all(dim=-1).float().mean().item() * 100
                    joint_ql_ems.append(em_ql)

                    p_te = joint_trunk(te_h, task_id=t_id).argmax(dim=-1)
                    em_te = (p_te == te_y).all(dim=-1).float().mean().item() * 100
                    joint_te_ems.append(em_te)

            mean_joint_ql = sum(joint_ql_ems) / len(joint_ql_ems)
            mean_joint_te = sum(joint_te_ems) / len(joint_te_ems)

            # Sequential Unregularized Training (shows forgetting baseline)
            name = f"sequential:{rank}"
            initialization_seeds[name] = stream_seed(seed, name)
            with isolated_rng(seed, name):
                seq_trunk = TaskConditionedKnowledgeTrunk(
                    max_tasks=max(10, num_tasks),
                    d_model=128,
                    d_shared=rank,
                    num_symbols=16,
                ).to(device)

            for t_id, tr_h, tr_y, _, _, _, _ in task_data:
                opt_seq = torch.optim.AdamW(adamw_groups(seq_trunk, 0.01), lr=lr)
                for _ in range(epochs):
                    logits = seq_trunk(tr_h, task_id=t_id)
                    loss = F.cross_entropy(logits.view(-1, 16), tr_y.view(-1))
                    opt_seq.zero_grad()
                    loss.backward()
                    opt_seq.step()

            seq_trunk.eval()
            seq_ql_ems = []
            with torch.no_grad():
                for t_id, _, _, ql_h, ql_y, _, _ in task_data:
                    p_ql = seq_trunk(ql_h, task_id=t_id).argmax(dim=-1)
                    em_ql = (p_ql == ql_y).all(dim=-1).float().mean().item() * 100
                    seq_ql_ems.append(em_ql)
            mean_seq_ql = sum(seq_ql_ems) / len(seq_ql_ems)

            params = joint_trunk.count_parameters()
            gate_a_passed = mean_joint_ql >= 85.0

            results_by_rank[rank] = RankFeasibilityResult(
                rank=rank,
                trainable_parameters=params,
                budget_compliant=params <= 20_000,
                joint_qualification_em=round(mean_joint_ql, 2),
                joint_test_em=round(mean_joint_te, 2),
                per_task_joint_ems=[round(x, 2) for x in joint_ql_ems],
                sequential_qualification_em=round(mean_seq_ql, 2),
                modular_reference_em=round(modular_ref_em, 2),
                gate_a_passed=gate_a_passed,
            )

        target_rank = (
            selected_rank if selected_rank in results_by_rank else list(results_by_rank.keys())[0]
        )
        gate_a_overall = results_by_rank[target_rank].gate_a_passed

        return StageACapacityReport(
            ranks_evaluated=ranks,
            results_by_rank=results_by_rank,
            gate_a_overall_passed=gate_a_overall,
            selected_rank=selected_rank,
            modular_reference_em=round(modular_ref_em, 2),
            initialization_seeds=initialization_seeds,
        )
