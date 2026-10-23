"""Phase 4: Controlled Optimization-Budget Benchmark Engine.

Compares 6 adaptation arms under matched maximum resource envelopes:
  - Arm A: Fixed Short Budget (10 optimization steps)
  - Arm B: Fixed Medium Budget (20 optimization steps)
  - Arm C: Fixed Max Budget (40 optimization steps)
  - Arm D: Deterministic Adaptive Budget Controller (Dynamic 10-40 steps)
  - Arm E: Heuristic / Learned Adaptive Controller
  - Arm F: M57 Bounded Expansion Control (20 steps + expansion to d=32)

Operates strictly under M53 combined supervision (pointwise CE + minimal pair loss).
Tracks actual optimizer steps, wall-clock GPU duration, unused compute fraction,
and compute-adjusted learning efficiency.
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter
from adaptive_mini_gpt.guided_experiment.environment import AutonomousEnvironment, MinimalPairControl
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.reliable_transfer.composition_models import ResidualSkillAdapter
from adaptive_mini_gpt.capacity_evolution.evolvable_adapter import EvolvableResidualSkillAdapter
from adaptive_mini_gpt.adaptive_optimization.budget_controller import (
    AdaptiveBudgetController,
    BudgetAction,
    ControllerBudgetDecision,
    HeuristicLearnedBudgetController,
)
from adaptive_mini_gpt.adaptive_optimization.task_environments import (
    ObservableTaskComplexity,
    compute_observable_complexity,
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizationStepMetrics:
    """Evaluation metrics at a specific adaptation practice budget N."""
    step_n: int
    macro_f1: float
    accuracy: float
    paired_correctness: float
    unknown_f1: float


@dataclass
class ArmOptimizationEvaluation:
    """Complete adaptation trajectory and resource profile for one experimental arm."""
    arm_name: str
    target_env_id: str
    trainable_parameters: int
    total_optimizer_steps_taken: int
    max_budget_steps_allowed: int
    fraction_unused_compute: float
    gpu_duration_seconds: float
    jumpstart_f1: float
    final_f1: float
    final_accuracy: float
    final_paired_correctness: float
    learning_curve_auc: float
    compute_adjusted_efficiency: float
    trajectory: list[OptimizationStepMetrics] = field(default_factory=list)
    controller_decisions: list[ControllerBudgetDecision] = field(default_factory=list)


class AdaptiveOptimizationBenchmarkEngine:
    """Executes fair compute-matched adaptation comparisons across all 6 arms."""

    CLASS_LABELS = ["YES", "NO", "UNKNOWN"]
    PRACTICE_SCHEDULE = [0, 2, 4, 8, 16, 32]

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: Any,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def _extract_hidden(self, text: str) -> torch.Tensor:
        toks = self.tokenizer.encode(text)
        tok_tensor = torch.tensor([toks], dtype=torch.long, device=self.device)
        b, t = tok_tensor.size()
        tok_emb = self.base_model.wte(tok_tensor)
        pos = torch.arange(0, t, dtype=torch.long, device=self.device)
        x = self.base_model.drop(tok_emb + self.base_model.wpe(pos))
        for block in self.base_model.blocks:
            x, _ = block(x)
        return self.base_model.ln_f(x)

    def evaluate_target_module(
        self,
        module: nn.Module,
        env: AutonomousEnvironment,
    ) -> tuple[float, float, float, float]:
        """Compute (macro_f1, accuracy, paired_correctness, unknown_f1)."""
        module.eval()
        self.base_model.eval()

        predictions: list[str] = []
        ground_truth: list[str] = []

        with torch.no_grad():
            for q in env.held_out_test_queries:
                h = self._extract_hidden(q.question_text)
                logits = module(h)
                pred_idx = int(torch.argmax(logits, dim=-1).item())
                predictions.append(self.CLASS_LABELS[pred_idx])
                ground_truth.append(q.expected_answer)

        # Accuracy
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        accuracy = float(correct / len(ground_truth)) if ground_truth else 0.0

        # Macro F1
        f1s = []
        for lab in self.CLASS_LABELS:
            tp = sum(1 for p, g in zip(predictions, ground_truth) if p == lab and g == lab)
            fp = sum(1 for p, g in zip(predictions, ground_truth) if p == lab and g != lab)
            fn = sum(1 for p, g in zip(predictions, ground_truth) if p != lab and g == lab)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
        macro_f1 = float(np.mean(f1s))

        # Unknown F1
        unknown_f1 = f1s[2]

        # Paired Correctness
        pair_dict = {q.query_id: p for q, p in zip(env.held_out_test_queries, predictions)}
        pair_correct = 0
        total_pairs = len(env.held_out_minimal_pairs)
        for pair in env.held_out_minimal_pairs:
            p_pos = pair_dict.get(pair.positive_query_id)
            p_neg = pair_dict.get(pair.negative_query_id)
            if p_pos == pair.expected_positive_answer and p_neg == pair.expected_negative_answer:
                pair_correct += 1
        paired_correctness = float(pair_correct / total_pairs) if total_pairs > 0 else 0.0

        return macro_f1, accuracy, paired_correctness, unknown_f1

    def run_training_steps(
        self,
        module: nn.Module,
        optimizer: optim.Optimizer,
        env: AutonomousEnvironment,
        num_practice_examples: int,
        steps_to_run: int,
        criterion: nn.CrossEntropyLoss,
    ) -> list[float]:
        """Execute a block of gradient optimization steps under M53 Combined Supervision."""
        if num_practice_examples == 0 or steps_to_run == 0:
            return [1.0]

        module.train()
        train_queries = env.training_practice_queries[:num_practice_examples]
        losses: list[float] = []

        for step in range(steps_to_run):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)

            # Pointwise Supervision
            for q in train_queries:
                h = self._extract_hidden(q.question_text)
                logits = module(h)
                target_idx = torch.tensor([self.CLASS_LABELS.index(q.expected_answer)], device=self.device)
                total_loss = total_loss + criterion(logits, target_idx)

            pointwise_loss = total_loss / len(train_queries)

            # Structural Minimal-Pair Contrastive Loss
            pair_loss = torch.tensor(0.0, device=self.device)
            active_pairs = [p for p in env.held_out_minimal_pairs if p.positive_query_id in [q.query_id for q in train_queries]]
            if active_pairs:
                for pair in active_pairs:
                    h_pos = self._extract_hidden(pair.positive_query_text)
                    h_neg = self._extract_hidden(pair.negative_query_text)
                    l_pos = module(h_pos)
                    l_neg = module(h_neg)
                    # Contrastive margin loss
                    pair_loss = pair_loss + F.relu(1.0 - (l_pos[0, 0] - l_neg[0, 1]))
                pair_loss = pair_loss / len(active_pairs)

            combined_loss = pointwise_loss + 0.5 * pair_loss
            combined_loss.backward()
            optimizer.step()
            losses.append(float(combined_loss.item()))

        return losses

    def run_arm_adaptation(
        self,
        arm_name: str,
        source_adapter: PersistentCandidateAdapter,
        target_env: AutonomousEnvironment,
        max_budget_steps: int = 40,
    ) -> ArmOptimizationEvaluation:
        """Run full adaptation trajectory across practice schedule for a given arm."""
        t_start = time.perf_counter()
        criterion = nn.CrossEntropyLoss()
        complexity = compute_observable_complexity(target_env)

        # 1. Instantiate Module Architecture
        is_fresh = (arm_name == "fresh_baseline")
        is_expansion = (arm_name == "m57_expansion_control")

        if is_fresh:
            adapter_candidate: nn.Module = PersistentCandidateAdapter(
                d_model=self.base_model.config.d_model,
                bottleneck_dim=16,
                num_classes=3,
            ).to(self.device)
        elif is_expansion:
            adapter_candidate = EvolvableResidualSkillAdapter(
                source_adapter=source_adapter,
                d_model=self.base_model.config.d_model,
                initial_bottleneck_dim=16,
            ).to(self.device)
        else:
            # All other arms use standard compact ResidualSkillAdapter (d=16, 2,371 params)
            adapter_candidate = ResidualSkillAdapter(
                source_adapter=source_adapter,
                d_model=self.base_model.config.d_model,
                residual_bottleneck_dim=16,
            ).to(self.device)

        trainable_params = sum(p.numel() for p in adapter_candidate.parameters() if p.requires_grad)

        optimizer = optim.AdamW(
            [p for p in adapter_candidate.parameters() if p.requires_grad],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        controller = AdaptiveBudgetController(
            block_step_size=10,
            max_budget_steps=max_budget_steps,
        )
        heuristic_controller = HeuristicLearnedBudgetController(controller)

        trajectory: list[OptimizationStepMetrics] = []
        controller_decisions: list[ControllerBudgetDecision] = []
        loss_history: list[float] = [1.0]
        total_steps_taken = 0
        has_reset = False

        # Evaluate Jumpstart (N = 0)
        init_f1, init_acc, init_pair, init_unk = self.evaluate_target_module(adapter_candidate, target_env)
        trajectory.append(OptimizationStepMetrics(0, init_f1, init_acc, init_pair, init_unk))

        # Iterate over practice example budgets N
        for n_idx, n_examples in enumerate(self.PRACTICE_SCHEDULE[1:], start=1):
            # Determine steps to allocate based on arm
            if arm_name == "fixed_short_10":
                steps_for_block = 10 if n_idx == 1 else 0
            elif arm_name == "fixed_medium_20":
                steps_for_block = 10 if n_idx in (1, 2) else 0
            elif arm_name == "fixed_max_40":
                steps_for_block = 8  # 5 rounds * 8 = 40 total
            elif arm_name == "deterministic_adaptive":
                dec = controller.evaluate_budget_action(
                    current_steps=total_steps_taken,
                    val_loss_history=loss_history,
                    complexity=complexity,
                    has_reset=has_reset,
                )
                controller_decisions.append(dec)
                if dec.action == BudgetAction.RESET and not has_reset:
                    # Reset to fresh adapter
                    adapter_candidate = PersistentCandidateAdapter(
                        d_model=self.base_model.config.d_model,
                        bottleneck_dim=16,
                        num_classes=3,
                    ).to(self.device)
                    optimizer = optim.AdamW(
                        adapter_candidate.parameters(),
                        lr=self.learning_rate,
                        weight_decay=self.weight_decay,
                    )
                    has_reset = True
                    steps_for_block = dec.block_steps_allocated
                elif dec.action == BudgetAction.TRAIN:
                    steps_for_block = dec.block_steps_allocated
                else:  # KEEP
                    steps_for_block = 0
            elif arm_name == "heuristic_adaptive":
                dec = heuristic_controller.evaluate_budget_action(
                    current_steps=total_steps_taken,
                    val_loss_history=loss_history,
                    complexity=complexity,
                    has_reset=has_reset,
                )
                controller_decisions.append(dec)
                if dec.action == BudgetAction.RESET and not has_reset:
                    adapter_candidate = PersistentCandidateAdapter(
                        d_model=self.base_model.config.d_model,
                        bottleneck_dim=16,
                        num_classes=3,
                    ).to(self.device)
                    optimizer = optim.AdamW(
                        adapter_candidate.parameters(),
                        lr=self.learning_rate,
                        weight_decay=self.weight_decay,
                    )
                    has_reset = True
                    steps_for_block = dec.block_steps_allocated
                elif dec.action == BudgetAction.TRAIN:
                    steps_for_block = dec.block_steps_allocated
                else:
                    steps_for_block = 0
            elif arm_name == "m57_expansion_control":
                if n_idx == 3:  # Expand at round 3 (N=8)
                    adapter_candidate.expand_capacity(target_bottleneck_dim=32)
                    optimizer = optim.AdamW(
                        [p for p in adapter_candidate.parameters() if p.requires_grad],
                        lr=self.learning_rate,
                        weight_decay=self.weight_decay,
                    )
                    trainable_params = sum(p.numel() for p in adapter_candidate.parameters() if p.requires_grad)
                steps_for_block = 4  # 5 * 4 = 20 steps
            else:
                steps_for_block = 4

            # Enforce max budget cap
            allowed_steps = min(steps_for_block, max_budget_steps - total_steps_taken)
            if allowed_steps > 0:
                step_losses = self.run_training_steps(
                    module=adapter_candidate,
                    optimizer=optimizer,
                    env=target_env,
                    num_practice_examples=n_examples,
                    steps_to_run=allowed_steps,
                    criterion=criterion,
                )
                loss_history.extend(step_losses)
                total_steps_taken += allowed_steps

            f1, acc, pair, unk = self.evaluate_target_module(adapter_candidate, target_env)
            trajectory.append(OptimizationStepMetrics(n_examples, f1, acc, pair, unk))

        t_end = time.perf_counter()
        gpu_duration = t_end - t_start

        # Calculate AUC over practice trajectory
        f1_values = [m.macro_f1 for m in trajectory]
        auc = float(np.mean(f1_values))

        unused_fraction = max(0.0, float(1.0 - (total_steps_taken / max_budget_steps)))
        compute_ratio = max(0.1, float(total_steps_taken / max_budget_steps))
        compute_efficiency = float(auc / compute_ratio)

        final_m = trajectory[-1]
        return ArmOptimizationEvaluation(
            arm_name=arm_name,
            target_env_id=target_env.env_id,
            trainable_parameters=trainable_params,
            total_optimizer_steps_taken=total_steps_taken,
            max_budget_steps_allowed=max_budget_steps,
            fraction_unused_compute=unused_fraction,
            gpu_duration_seconds=gpu_duration,
            jumpstart_f1=init_f1,
            final_f1=final_m.macro_f1,
            final_accuracy=final_m.accuracy,
            final_paired_correctness=final_m.paired_correctness,
            learning_curve_auc=auc,
            compute_adjusted_efficiency=compute_efficiency,
            trajectory=trajectory,
            controller_decisions=controller_decisions,
        )
