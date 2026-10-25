"""Milestone M84 Phase 3: Dynamic Sparse Neural Adapter Training Engine.

Implements controlled causal training across 7 comparative topology strategies:
1. ORIGINAL_DENSE: Full unpruned baseline (100% active connections).
2. RANDOM_PRUNING_REGROWTH: Stochastic prune/regrow control without gradient/magnitude guidance.
3. STATIC_MAGNITUDE_PRUNING: Traditional one-shot magnitude pruning without connection regrowth.
4. RIGL_GRADIENT_REGROWTH: Dynamic sparse training via gradient-salient reactivation + magnitude death.
5. FIXED_SPARSE: Static random sparse topology fixed from initialization.
6. LEARNED_TOPOLOGY_CONTROLLER: M84 autonomous policy (<1,000 params) driving dynamic topology.
7. DIAGNOSTIC_ORACLE: Evaluates candidate topology actions on validation loss to select best move.

Enforces:
- Deterministic mask enforcement after every gradient update.
- Bit-exact rollback support and hash logging.
- Exact tracking of active connections, parameter counts, loss curves, and steps to qualification.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.topology_adaptation.sparse_adapter import (
    DynamicSparseSequenceAdapter,
    PruningCriterion,
    RegrowthCriterion,
    TopologyStats,
)
from adaptive_mini_gpt.topology_adaptation.topology_controller import (
    LearnedTopologyController,
    TopologyAction,
    TopologyState,
)

logger = logging.getLogger(__name__)


class TopologyTrainingStrategy(StrEnum):
    """Comparative experimental arms for Milestone M84."""

    ORIGINAL_DENSE = "ORIGINAL_DENSE"
    RANDOM_PRUNING_REGROWTH = "RANDOM_PRUNING_REGROWTH"
    STATIC_MAGNITUDE_PRUNING = "STATIC_MAGNITUDE_PRUNING"
    RIGL_GRADIENT_REGROWTH = "RIGL_GRADIENT_REGROWTH"
    FIXED_SPARSE = "FIXED_SPARSE"
    LEARNED_TOPOLOGY_CONTROLLER = "LEARNED_TOPOLOGY_CONTROLLER"
    DIAGNOSTIC_ORACLE = "DIAGNOSTIC_ORACLE"


@dataclass
class TrainingOutcome:
    """Detailed results and telemetry from adapter training."""

    strategy: TopologyTrainingStrategy
    final_loss: float
    final_em: float
    steps_to_threshold: int | None
    total_steps: int
    initial_active_connections: int
    final_active_connections: int
    final_sparsity: float
    initial_topology_hash: str
    final_topology_hash: str
    history_loss: list[float] = field(default_factory=list)
    history_em: list[float] = field(default_factory=list)
    actions_taken: list[str] = field(default_factory=list)
    rollback_performed: bool = False


class SparseAdapterTrainer:
    """Training engine supporting dynamic rewiring and learned topology adaptation."""

    @staticmethod
    def evaluate(adapter: DynamicSparseSequenceAdapter, h: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
        """Compute cross-entropy loss and exact match accuracy."""
        adapter.eval()
        with torch.no_grad():
            logits = adapter(h)
            loss = F.cross_entropy(logits.view(-1, adapter.num_symbols), y.view(-1)).item()
            preds = logits.argmax(dim=-1)
            # Exact match across the sequence
            em = (preds == y).all(dim=-1).float().mean().item()
        return loss, em

    @staticmethod
    def _extract_topology_state(
        adapter: DynamicSparseSequenceAdapter,
        loss_history: list[float],
        logits: torch.Tensor,
        target_sparsity: float,
    ) -> TopologyState:
        """Extract diagnostic non-leaking state features for the topology controller."""
        stats = adapter.get_topology_stats()
        current_sparsity = stats.sparsity_ratio

        # Mean absolute weight across down and up projections
        down_w = adapter.down_proj.weight.detach().abs().mean().item()
        up_w = adapter.up_proj.weight.detach().abs().mean().item()
        mean_weight = (down_w + up_w) / 2.0

        # Gradient norm across active weights
        grad_norms = []
        if adapter.down_proj.weight.grad is not None:
            grad_norms.append(adapter.down_proj.weight.grad.norm().item())
        if adapter.up_proj.weight.grad is not None:
            grad_norms.append(adapter.up_proj.weight.grad.norm().item())
        grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0

        # Loss slope over recent steps
        if len(loss_history) >= 4:
            loss_slope = float((loss_history[-1] - loss_history[-4]) / 4.0)
        elif len(loss_history) >= 2:
            loss_slope = float(loss_history[-1] - loss_history[-2])
        else:
            loss_slope = 0.0

        # Prediction entropy on output distribution
        probs = F.softmax(logits.detach().view(-1, adapter.num_symbols), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item()

        # Budget pressure: positive when current sparsity is lower than target
        budget_pressure = max(0.0, float((target_sparsity - current_sparsity) / max(0.01, target_sparsity)))

        return TopologyState(
            current_sparsity=float(current_sparsity),
            mean_weight_magnitude=float(mean_weight),
            gradient_norm=float(grad_norm),
            loss_slope=float(loss_slope),
            prediction_entropy=float(entropy),
            budget_pressure=float(budget_pressure),
        )

    @classmethod
    def train_adapter(
        cls,
        adapter: DynamicSparseSequenceAdapter,
        train_h: torch.Tensor,
        train_y: torch.Tensor,
        qual_h: torch.Tensor,
        qual_y: torch.Tensor,
        strategy: TopologyTrainingStrategy,
        target_sparsity: float = 0.30,
        total_steps: int = 50,
        lr: float = 0.015,
        rewire_freq: int = 10,
        rewire_end_step: int = 35,
        controller: LearnedTopologyController | None = None,
        seed: int = 42,
        em_threshold: float = 0.85,
    ) -> TrainingOutcome:
        """Train an adapter under the specified topology adaptation strategy."""
        torch.manual_seed(seed)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)

        init_stats = adapter.get_topology_stats()
        initial_hash = adapter.compute_topology_hash()
        initial_active = init_stats.active_connections

        # Initialize fixed sparsity if specified
        if strategy == TopologyTrainingStrategy.FIXED_SPARSE:
            adapter.prune_connections(prune_ratio=target_sparsity, criterion=PruningCriterion.RANDOM, seed=seed)
            adapter.enforce_masks()
        elif strategy == TopologyTrainingStrategy.STATIC_MAGNITUDE_PRUNING:
            # For static magnitude pruning, prune initial small weights immediately to target sparsity
            adapter.prune_connections(prune_ratio=target_sparsity, criterion=PruningCriterion.MAGNITUDE)
            adapter.enforce_masks()

        history_loss: list[float] = []
        history_em: list[float] = []
        actions_taken: list[str] = []
        steps_to_threshold: int | None = None
        rollback_performed = False

        for step in range(1, total_steps + 1):
            adapter.train()
            optimizer.zero_grad()

            logits = adapter(train_h)
            loss = F.cross_entropy(logits.view(-1, adapter.num_symbols), train_y.view(-1))
            loss.backward()

            # Dynamic Topology Intervention Point
            should_intervene = (
                step % rewire_freq == 0
                and step <= rewire_end_step
                and strategy not in (
                    TopologyTrainingStrategy.ORIGINAL_DENSE,
                    TopologyTrainingStrategy.FIXED_SPARSE,
                    TopologyTrainingStrategy.STATIC_MAGNITUDE_PRUNING,
                )
            )

            if should_intervene:
                # Save rollback snapshot before topology modification
                pre_topology_checkpoint = adapter.save_topology_state()

                if strategy == TopologyTrainingStrategy.RIGL_GRADIENT_REGROWTH:
                    current_sparsity = adapter.get_topology_stats().sparsity_ratio
                    if current_sparsity < target_sparsity:
                        # Prune towards target sparsity first
                        needed_prune = (target_sparsity - current_sparsity) / (1.0 - current_sparsity)
                        adapter.prune_connections(prune_ratio=needed_prune, criterion=PruningCriterion.MAGNITUDE)
                        actions_taken.append(f"step_{step}:rigl_target_prune")
                    else:
                        # RigL cosine annealing rewire ratio
                        rewire_ratio = 0.15 * (1.0 + math.cos(math.pi * step / rewire_end_step)) / 2.0
                        pruned_cnt, regrown_cnt = adapter.rewire_topology(
                            rewire_fraction=max(0.05, rewire_ratio),
                            prune_criterion=PruningCriterion.MAGNITUDE,
                            regrow_criterion=RegrowthCriterion.GRADIENT_SALIENT,
                            seed=seed + step,
                        )
                        actions_taken.append(f"step_{step}:rigl_rewire({pruned_cnt}p,{regrown_cnt}r)")

                elif strategy == TopologyTrainingStrategy.RANDOM_PRUNING_REGROWTH:
                    curr_sp = adapter.get_topology_stats().sparsity_ratio
                    if curr_sp < target_sparsity:
                        needed_ratio = (target_sparsity - curr_sp) / max(0.01, 1.0 - curr_sp)
                        adapter.prune_connections(prune_ratio=min(0.20, needed_ratio), criterion=PruningCriterion.RANDOM, seed=seed+step)
                        actions_taken.append(f"step_{step}:random_target_prune")
                    else:
                        pruned_cnt, regrown_cnt = adapter.rewire_topology(
                            rewire_fraction=0.10,
                            prune_criterion=PruningCriterion.RANDOM,
                            regrow_criterion=RegrowthCriterion.RANDOM,
                            seed=seed + step,
                        )
                        actions_taken.append(f"step_{step}:random_rewire({pruned_cnt}p,{regrown_cnt}r)")

                elif strategy == TopologyTrainingStrategy.LEARNED_TOPOLOGY_CONTROLLER:
                    if controller is None:
                        raise ValueError("LearnedTopologyController must be supplied for LEARNED_TOPOLOGY_CONTROLLER")

                    state = cls._extract_topology_state(adapter, history_loss, logits, target_sparsity)
                    action = controller.select_action(state, deterministic=True)

                    if action == TopologyAction.MAINTAIN_TOPOLOGY:
                        actions_taken.append(f"step_{step}:MAINTAIN")
                    elif action == TopologyAction.PRUNE_LEAST_IMPORTANT:
                        curr_sp = adapter.get_topology_stats().sparsity_ratio
                        if curr_sp < target_sparsity:
                            needed_ratio = (target_sparsity - curr_sp) / max(0.01, 1.0 - curr_sp)
                            pruned = adapter.prune_connections(prune_ratio=min(0.20, needed_ratio), criterion=PruningCriterion.MAGNITUDE)
                            actions_taken.append(f"step_{step}:PRUNE_MAGNITUDE({pruned})")
                        else:
                            actions_taken.append(f"step_{step}:PRUNE_SKIPPED_AT_TARGET")
                    elif action == TopologyAction.REGROW_GRADIENT_SALIENT:
                        regrown = adapter.regrow_connections(regrow_count=15, criterion=RegrowthCriterion.GRADIENT_SALIENT)
                        actions_taken.append(f"step_{step}:REGROW_GRADIENT({regrown})")
                    elif action == TopologyAction.BALANCED_REWIRE:
                        p, r = adapter.rewire_topology(
                            rewire_fraction=0.10,
                            prune_criterion=PruningCriterion.MAGNITUDE,
                            regrow_criterion=RegrowthCriterion.GRADIENT_SALIENT,
                            seed=seed + step,
                        )
                        actions_taken.append(f"step_{step}:BALANCED_REWIRE({p}p,{r}r)")

                elif strategy == TopologyTrainingStrategy.DIAGNOSTIC_ORACLE:
                    # Evaluate candidates: maintain vs magnitude prune vs gradient rewire
                    candidate_actions = [
                        ("maintain", lambda: None),
                        ("prune_10", lambda: adapter.prune_connections(0.10, PruningCriterion.MAGNITUDE)),
                        ("rewire_10", lambda: adapter.rewire_topology(0.10, PruningCriterion.MAGNITUDE, RegrowthCriterion.GRADIENT_SALIENT, seed=seed+step)),
                    ]
                    best_action = "maintain"
                    best_loss = float("inf")

                    for action_name, action_fn in candidate_actions:
                        adapter.load_topology_state(pre_topology_checkpoint)
                        action_fn()
                        adapter.enforce_masks()
                        eval_loss, _ = cls.evaluate(adapter, qual_h, qual_y)
                        if eval_loss < best_loss:
                            best_loss = eval_loss
                            best_action = action_name

                    # Apply best candidate
                    adapter.load_topology_state(pre_topology_checkpoint)
                    if best_action == "prune_10":
                        adapter.prune_connections(0.10, PruningCriterion.MAGNITUDE)
                    elif best_action == "rewire_10":
                        adapter.rewire_topology(0.10, PruningCriterion.MAGNITUDE, RegrowthCriterion.GRADIENT_SALIENT, seed=seed+step)
                    actions_taken.append(f"step_{step}:ORACLE_{best_action}")

                # Ensure zeroed weights on newly deactivated connections
                adapter.enforce_masks()

            # Optimizer Step
            optimizer.step()
            adapter.enforce_masks()

            # Qualification Evaluation
            eval_loss, eval_em = cls.evaluate(adapter, qual_h, qual_y)
            history_loss.append(eval_loss)
            history_em.append(eval_em)

            if steps_to_threshold is None and eval_em >= em_threshold:
                steps_to_threshold = step

        final_stats = adapter.get_topology_stats()
        final_hash = adapter.compute_topology_hash()
        final_loss, final_em = cls.evaluate(adapter, qual_h, qual_y)

        return TrainingOutcome(
            strategy=strategy,
            final_loss=final_loss,
            final_em=final_em,
            steps_to_threshold=steps_to_threshold,
            total_steps=total_steps,
            initial_active_connections=initial_active,
            final_active_connections=final_stats.active_connections,
            final_sparsity=final_stats.sparsity_ratio,
            initial_topology_hash=initial_hash,
            final_topology_hash=final_hash,
            history_loss=history_loss,
            history_em=history_em,
            actions_taken=actions_taken,
            rollback_performed=rollback_performed,
        )
