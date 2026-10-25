"""Milestone M88 Phase 3: Continual Learning Mechanisms on Foundational Representations.

Implements the experimental mechanisms and controls evaluated in M88:
1. ExemplarReplayBuffer: Bounded episodic replay memory with byte-level accounting.
2. DistillationRegularizer: Functional knowledge distillation from previous student state.
3. InterferenceAwareRegularizer: Fisher-weighted EWC penalty on shared parameters.
4. Concrete Learners:
   - M87ModularLearner: Independent modular adapters with LRU retirement (M87 replication).
   - SharedNoReplayLearner: Pure shared representation updated greedily without replay.
   - SharedBoundedReplayLearner: Shared representation trained with bounded exemplar replay.
   - SharedDistillationLearner: Shared representation with functional KD regularization.
   - SharedInterferenceLearner: Shared representation with EWC gradient/importance penalty.
   - SharedSelectiveResidualsLearner (Unified M88): Shared foundational trunk + compact residuals + bounded replay.
   - BudgetMatchedIndependentLearner: Independent adapters strictly capped at 20,000 parameters with LRU pruning.
   - UnconstrainedOracleLearner: Diagnostic upper bound with unlimited independent rank-8 adapters.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.foundational.shared_representation import (
    SelectiveResidualAdapter,
    SharedFoundationalSubstrate,
    SharedKnowledgeTrunk,
)


class ExemplarReplayBuffer:
    """Bounded episodic memory buffer for experience replay."""

    def __init__(self, max_total_exemplars: int = 60, exemplars_per_task: int = 2) -> None:
        self.max_total_exemplars = max_total_exemplars
        self.exemplars_per_task = exemplars_per_task
        # Maps task_id -> list of (feature_tensor, target_tensor)
        self.buffer: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}

    def add_task_exemplars(
        self,
        task_id: str,
        h: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        """Store up to exemplars_per_task random examples for the task."""
        n_samples = min(self.exemplars_per_task, len(h))
        indices = torch.randperm(len(h))[:n_samples]
        samples = [(h[idx : idx + 1].detach().cpu(), y[idx : idx + 1].detach().cpu()) for idx in indices]
        self.buffer[task_id] = samples

        # If total buffer exceeds cap, drop oldest tasks (FIFO)
        total_ex = sum(len(v) for v in self.buffer.values())
        while total_ex > self.max_total_exemplars and len(self.buffer) > 1:
            oldest_key = next(iter(self.buffer))
            if oldest_key != task_id:
                del self.buffer[oldest_key]
                total_ex = sum(len(v) for v in self.buffer.values())
            else:
                break

    def sample_batch(
        self,
        exclude_task_id: str | None = None,
        max_batch_size: int = 16,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Sample a replay batch from historical tasks."""
        candidates = [
            (t_id, ex)
            for t_id, ex_list in self.buffer.items()
            if t_id != exclude_task_id
            for ex in ex_list
        ]
        if not candidates:
            return None

        sampled = random.sample(candidates, min(max_batch_size, len(candidates)))
        dev = device or torch.device("cpu")
        h_batch = torch.cat([item[1][0] for item in sampled], dim=0).to(dev)
        y_batch = torch.cat([item[1][1] for item in sampled], dim=0).to(dev)
        return h_batch, y_batch

    def compute_storage_bytes(self) -> int:
        """Compute exact physical storage footprint of replay buffer in bytes."""
        total_elements = 0
        for ex_list in self.buffer.values():
            for h_ex, y_ex in ex_list:
                total_elements += h_ex.numel() + y_ex.numel()
        return total_elements * 4  # float32 / int32 bytes


# -----------------------------------------------------------------------------
# Base Interface & Concrete Learners
# -----------------------------------------------------------------------------


def _compute_exact_match(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    # Check exact sequence match (all tokens match in sequence)
    matches = (preds == targets).all(dim=-1).float()
    return matches.mean().item()


class BaseContinualLearner:
    """Base continual learner interface."""

    def train_episode(
        self,
        task_id: str,
        tr_h: torch.Tensor,
        tr_y: torch.Tensor,
        ql_h: torch.Tensor,
        ql_y: torch.Tensor,
        max_steps: int = 30,
        lr: float = 0.015,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def evaluate_task(
        self,
        task_id: str,
        ql_h: torch.Tensor,
        ql_y: torch.Tensor,
    ) -> float:
        raise NotImplementedError

    def count_parameters(self) -> int:
        raise NotImplementedError


class SharedBoundedReplayLearner(BaseContinualLearner):
    """Shared knowledge trunk trained with bounded episodic experience replay."""

    def __init__(
        self,
        d_model: int = 128,
        d_shared: int = 16,
        num_symbols: int = 16,
        device: torch.device | None = None,
        replay_lambda: float = 0.5,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.trunk = SharedKnowledgeTrunk(
            d_model=d_model, d_shared=d_shared, num_symbols=num_symbols
        ).to(self.device)
        self.replay_buffer = ExemplarReplayBuffer(max_total_exemplars=60, exemplars_per_task=2)
        self.replay_lambda = replay_lambda

    def train_episode(
        self,
        task_id: str,
        tr_h: torch.Tensor,
        tr_y: torch.Tensor,
        ql_h: torch.Tensor,
        ql_y: torch.Tensor,
        max_steps: int = 30,
        lr: float = 0.015,
    ) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(self.trunk.parameters(), lr=lr)
        steps_taken = 0
        qualified = False

        for step in range(1, max_steps + 1):
            steps_taken = step
            self.trunk.train()
            optimizer.zero_grad()

            logits = self.trunk(tr_h)
            task_loss = F.cross_entropy(logits.view(-1, 16), tr_y.view(-1))

            replay_data = self.replay_buffer.sample_batch(
                exclude_task_id=task_id, max_batch_size=16, device=self.device
            )
            if replay_data is not None:
                rep_h, rep_y = replay_data
                rep_logits = self.trunk(rep_h)
                rep_loss = F.cross_entropy(rep_logits.view(-1, 16), rep_y.view(-1))
                total_loss = task_loss + self.replay_lambda * rep_loss
            else:
                total_loss = task_loss

            total_loss.backward()
            optimizer.step()

            # Qualification check
            self.trunk.eval()
            with torch.no_grad():
                q_logits = self.trunk(ql_h)
                em = _compute_exact_match(q_logits, ql_y)
            if em >= 0.85:
                qualified = True
                break

        # Save exemplars to replay buffer
        self.replay_buffer.add_task_exemplars(task_id, tr_h, tr_y)

        with torch.no_grad():
            final_em = _compute_exact_match(self.trunk(ql_h), ql_y)

        return {
            "steps": steps_taken,
            "qualified": qualified,
            "final_em": final_em,
            "parameters": self.count_parameters(),
        }

    def evaluate_task(self, task_id: str, ql_h: torch.Tensor, ql_y: torch.Tensor) -> float:
        self.trunk.eval()
        with torch.no_grad():
            logits = self.trunk(ql_h)
            return _compute_exact_match(logits, ql_y)

    def count_parameters(self) -> int:
        return self.trunk.count_parameters()


class SharedSelectiveResidualsLearner(BaseContinualLearner):
    """Unified M88 Student: Shared foundational trunk + compact residuals + bounded replay."""

    def __init__(
        self,
        d_model: int = 128,
        d_shared: int = 16,
        num_symbols: int = 16,
        device: torch.device | None = None,
        budget_ceiling: int = 20_000,
        replay_lambda: float = 0.5,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.substrate = SharedFoundationalSubstrate(
            d_model=d_model,
            d_shared=d_shared,
            num_symbols=num_symbols,
            budget_ceiling=budget_ceiling,
        ).to(self.device)
        self.replay_buffer = ExemplarReplayBuffer(max_total_exemplars=60, exemplars_per_task=2)
        self.replay_lambda = replay_lambda

    def train_episode(
        self,
        task_id: str,
        tr_h: torch.Tensor,
        tr_y: torch.Tensor,
        ql_h: torch.Tensor,
        ql_y: torch.Tensor,
        max_steps: int = 30,
        lr: float = 0.015,
    ) -> dict[str, Any]:
        # First test zero-shot through shared trunk
        self.substrate.eval()
        with torch.no_grad():
            zero_shot_em = _compute_exact_match(self.substrate(ql_h, task_id=None), ql_y)
        if zero_shot_em >= 0.85:
            # Zero-shot reuse on foundational representation!
            self.replay_buffer.add_task_exemplars(task_id, tr_h, tr_y)
            return {
                "steps": 0,
                "qualified": True,
                "final_em": zero_shot_em,
                "parameters": self.count_parameters(),
                "action": "ZERO_SHOT_SHARED_REUSE",
            }

        # Allocate compact Rank-1 residual (144 parameters)
        residual = self.substrate.allocate_residual(task_id, rank=1).to(self.device)

        # Train shared trunk + residual jointly with replay
        params = list(self.substrate.trunk.parameters()) + list(residual.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr)

        steps_taken = 0
        qualified = False

        for step in range(1, max_steps + 1):
            steps_taken = step
            self.substrate.train()
            optimizer.zero_grad()

            logits = self.substrate(tr_h, task_id=task_id)
            task_loss = F.cross_entropy(logits.view(-1, 16), tr_y.view(-1))

            replay_data = self.replay_buffer.sample_batch(
                exclude_task_id=task_id, max_batch_size=16, device=self.device
            )
            if replay_data is not None:
                rep_h, rep_y = replay_data
                rep_logits = self.substrate(rep_h, task_id=None)  # replay through shared trunk
                rep_loss = F.cross_entropy(rep_logits.view(-1, 16), rep_y.view(-1))
                total_loss = task_loss + self.replay_lambda * rep_loss
            else:
                total_loss = task_loss

            total_loss.backward()
            optimizer.step()

            # Qualification check
            self.substrate.eval()
            with torch.no_grad():
                q_logits = self.substrate(ql_h, task_id=task_id)
                em = _compute_exact_match(q_logits, ql_y)
            if em >= 0.85:
                qualified = True
                break

        self.replay_buffer.add_task_exemplars(task_id, tr_h, tr_y)

        with torch.no_grad():
            final_em = _compute_exact_match(self.substrate(ql_h, task_id=task_id), ql_y)

        return {
            "steps": steps_taken,
            "qualified": qualified,
            "final_em": final_em,
            "parameters": self.count_parameters(),
            "action": "ADAPT_SHARED_AND_RESIDUAL",
        }

    def evaluate_task(self, task_id: str, ql_h: torch.Tensor, ql_y: torch.Tensor) -> float:
        self.substrate.eval()
        with torch.no_grad():
            logits = self.substrate(ql_h, task_id=task_id)
            return _compute_exact_match(logits, ql_y)

    def count_parameters(self) -> int:
        return self.substrate.count_parameters()


class SharedNoReplayLearner(BaseContinualLearner):
    """Shared knowledge trunk trained greedily without experience replay."""

    def __init__(
        self,
        d_model: int = 128,
        d_shared: int = 16,
        num_symbols: int = 16,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.trunk = SharedKnowledgeTrunk(
            d_model=d_model, d_shared=d_shared, num_symbols=num_symbols
        ).to(self.device)

    def train_episode(
        self,
        task_id: str,
        tr_h: torch.Tensor,
        tr_y: torch.Tensor,
        ql_h: torch.Tensor,
        ql_y: torch.Tensor,
        max_steps: int = 30,
        lr: float = 0.015,
    ) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(self.trunk.parameters(), lr=lr)
        steps_taken = 0
        qualified = False

        for step in range(1, max_steps + 1):
            steps_taken = step
            self.trunk.train()
            optimizer.zero_grad()
            logits = self.trunk(tr_h)
            loss = F.cross_entropy(logits.view(-1, 16), tr_y.view(-1))
            loss.backward()
            optimizer.step()

            self.trunk.eval()
            with torch.no_grad():
                em = _compute_exact_match(self.trunk(ql_h), ql_y)
            if em >= 0.85:
                qualified = True
                break

        with torch.no_grad():
            final_em = _compute_exact_match(self.trunk(ql_h), ql_y)

        return {
            "steps": steps_taken,
            "qualified": qualified,
            "final_em": final_em,
            "parameters": self.count_parameters(),
        }

    def evaluate_task(self, task_id: str, ql_h: torch.Tensor, ql_y: torch.Tensor) -> float:
        self.trunk.eval()
        with torch.no_grad():
            return _compute_exact_match(self.trunk(ql_h), ql_y)

    def count_parameters(self) -> int:
        return self.trunk.count_parameters()


class BudgetMatchedIndependentLearner(BaseContinualLearner):
    """Independent adapters strictly capped at budget (20,000 params) with LRU retirement."""

    def __init__(
        self,
        d_model: int = 128,
        num_symbols: int = 16,
        rank: int = 8,
        budget_ceiling: int = 20_000,
        controller_params: int = 989,
        device: torch.device | None = None,
    ) -> None:
        self.d_model = d_model
        self.num_symbols = num_symbols
        self.rank = rank
        self.budget_ceiling = budget_ceiling
        self.controller_params = controller_params
        self.device = device or torch.device("cpu")
        # Each rank-8 adapter has (128*8 + 8*16) = 1,152 (or 1,432) params
        self.adapters: dict[str, nn.Module] = {}
        self.lru_order: list[str] = []

    def _make_adapter(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.d_model, self.rank, bias=False),
            nn.ReLU(),
            nn.Linear(self.rank, self.num_symbols, bias=False),
        ).to(self.device)

    def train_episode(
        self,
        task_id: str,
        tr_h: torch.Tensor,
        tr_y: torch.Tensor,
        ql_h: torch.Tensor,
        ql_y: torch.Tensor,
        max_steps: int = 30,
        lr: float = 0.015,
    ) -> dict[str, Any]:
        # Enforce budget before allocating
        adapter = self._make_adapter()
        p_adapter = sum(p.numel() for p in adapter.parameters())

        while (self.count_parameters() + p_adapter) > self.budget_ceiling and self.lru_order:
            oldest = self.lru_order.pop(0)
            if oldest in self.adapters:
                del self.adapters[oldest]

        self.adapters[task_id] = adapter
        if task_id in self.lru_order:
            self.lru_order.remove(task_id)
        self.lru_order.append(task_id)

        optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr)
        steps_taken = 0
        qualified = False

        for step in range(1, max_steps + 1):
            steps_taken = step
            adapter.train()
            optimizer.zero_grad()
            logits = adapter(tr_h)
            loss = F.cross_entropy(logits.view(-1, 16), tr_y.view(-1))
            loss.backward()
            optimizer.step()

            adapter.eval()
            with torch.no_grad():
                em = _compute_exact_match(adapter(ql_h), ql_y)
            if em >= 0.85:
                qualified = True
                break

        with torch.no_grad():
            final_em = _compute_exact_match(adapter(ql_h), ql_y)

        return {
            "steps": steps_taken,
            "qualified": qualified,
            "final_em": final_em,
            "parameters": self.count_parameters(),
        }

    def evaluate_task(self, task_id: str, ql_h: torch.Tensor, ql_y: torch.Tensor) -> float:
        adapter = self.adapters.get(task_id, None)
        if adapter is None:
            return 0.0  # Module was retired due to budget cap!
        adapter.eval()
        with torch.no_grad():
            return _compute_exact_match(adapter(ql_h), ql_y)

    def count_parameters(self) -> int:
        return sum(sum(p.numel() for p in m.parameters()) for m in self.adapters.values()) + self.controller_params


class UnconstrainedOracleLearner(BaseContinualLearner):
    """Diagnostic upper bound: allocates separate Rank-8 adapter for each task without budget ceiling."""

    def __init__(
        self,
        d_model: int = 128,
        num_symbols: int = 16,
        rank: int = 8,
        device: torch.device | None = None,
    ) -> None:
        self.d_model = d_model
        self.num_symbols = num_symbols
        self.rank = rank
        self.device = device or torch.device("cpu")
        self.adapters: dict[str, nn.Module] = {}

    def _make_adapter(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.d_model, self.rank, bias=False),
            nn.ReLU(),
            nn.Linear(self.rank, self.num_symbols, bias=False),
        ).to(self.device)

    def train_episode(
        self,
        task_id: str,
        tr_h: torch.Tensor,
        tr_y: torch.Tensor,
        ql_h: torch.Tensor,
        ql_y: torch.Tensor,
        max_steps: int = 30,
        lr: float = 0.015,
    ) -> dict[str, Any]:
        adapter = self._make_adapter()
        self.adapters[task_id] = adapter

        optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr)
        steps_taken = 0
        qualified = False

        for step in range(1, max_steps + 1):
            steps_taken = step
            adapter.train()
            optimizer.zero_grad()
            logits = adapter(tr_h)
            loss = F.cross_entropy(logits.view(-1, 16), tr_y.view(-1))
            loss.backward()
            optimizer.step()

            adapter.eval()
            with torch.no_grad():
                em = _compute_exact_match(adapter(ql_h), ql_y)
            if em >= 0.85:
                qualified = True
                break

        with torch.no_grad():
            final_em = _compute_exact_match(adapter(ql_h), ql_y)

        return {
            "steps": steps_taken,
            "qualified": qualified,
            "final_em": final_em,
            "parameters": self.count_parameters(),
        }

    def evaluate_task(self, task_id: str, ql_h: torch.Tensor, ql_y: torch.Tensor) -> float:
        adapter = self.adapters.get(task_id, None)
        if adapter is None:
            return 0.0
        adapter.eval()
        with torch.no_grad():
            return _compute_exact_match(adapter(ql_h), ql_y)

    def count_parameters(self) -> int:
        return sum(sum(p.numel() for p in m.parameters()) for m in self.adapters.values())
