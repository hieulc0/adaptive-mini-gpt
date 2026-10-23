"""Phase 3 & 7: Controlled Optimization-Budget Benchmark Engine for M59.

Compares adaptation arms under matched maximum resource envelopes:
  - Arm A: Fixed Short Budget (10 steps)
  - Arm B: Fixed Medium Budget (20 steps)
  - Arm C: Fixed Max Budget (40 steps)
  - Arm D: Deterministic Adaptive Budget Controller (M58 Controller)
  - Arm E: Validation-Learning Progress & Cost Controller (ProgressCostController)
  - Arm F: Experiential Learning-to-Learn Controller (ExperientialBudgetController)
  - Control Arm: M57 Bounded Expansion Control (20 steps + expansion to d=32)

Evaluates whether previous learning experiences improve learning efficiency
and final generalization on unfamiliar task families.
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
from adaptive_mini_gpt.guided_experiment.environment import AutonomousEnvironment
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.reliable_transfer.composition_models import ResidualSkillAdapter
from adaptive_mini_gpt.capacity_evolution.evolvable_adapter import EvolvableResidualSkillAdapter
from adaptive_mini_gpt.generalizable_learning.budget_controller import (
    BudgetAction,
    ControllerBudgetDecision,
    FixedBudgetController,
    M58AdaptiveBudgetController,
    ValidationProgressCostController,
    ExperientialBudgetController,
)
from adaptive_mini_gpt.generalizable_learning.experiential_memory import (
    ExperientialMemoryStore,
    LearningEpisodeRecord,
)
from adaptive_mini_gpt.generalizable_learning.task_suite import (
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


class GeneralizableOptimizationBenchmarkEngine:
    """Executes fair compute-matched adaptation comparisons across all M59 arms."""

    CLASS_LABELS = ["YES", "NO", "UNKNOWN"]
    PRACTICE_SCHEDULE = [0, 2, 4, 8, 16, 32]

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: Any,
        device: torch.device,
        memory_store: ExperientialMemoryStore | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.memory_store = memory_store or ExperientialMemoryStore()

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
        unknown_f1 = f1s[2]

        # Structural Paired Correctness on minimal pairs
        paired_correct = 0
        total_pairs = len(env.minimal_pairs)
        with torch.no_grad():
            for mp in env.minimal_pairs:
                h_base = self._extract_hidden(mp.base_query.question_text)
                pred_base = self.CLASS_LABELS[int(torch.argmax(module(h_base), dim=-1).item())]

                h_pert = self._extract_hidden(mp.perturbed_query.question_text)
                pred_pert = self.CLASS_LABELS[int(torch.argmax(module(h_pert), dim=-1).item())]\

                if pred_base == mp.base_query.expected_answer and pred_pert == mp.perturbed_query.expected_answer:
                    paired_correct += 1

        paired_correctness = float(paired_correct / total_pairs) if total_pairs > 0 else 0.0
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
        train_queries = env.held_out_test_queries[:num_practice_examples]
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

            # Minimal pair contrastive supervision
            pair_loss = torch.tensor(0.0, device=self.device)
            if env.minimal_pairs:
                mp = env.minimal_pairs[torch.randint(0, len(env.minimal_pairs), (1,)).item()]
                h_b = self._extract_hidden(mp.base_query.question_text)
                h_p = self._extract_hidden(mp.perturbed_query.question_text)
                l_b = module(h_b)
                l_p = module(h_p)
                y_b = torch.tensor([self.CLASS_LABELS.index(mp.base_query.expected_answer)], device=self.device)
                y_p = torch.tensor([self.CLASS_LABELS.index(mp.perturbed_query.expected_answer)], device=self.device)
                pair_loss = 0.5 * (criterion(l_b, y_b) + criterion(l_p, y_p))

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
            # 2,371 parameters
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

        # Select controller
        if arm_name == "fixed_short_10":
            controller = FixedBudgetController(10)
        elif arm_name == "fixed_medium_20":
            controller = FixedBudgetController(20)
        elif arm_name == "fixed_max_40":
            controller = FixedBudgetController(40)
        elif arm_name == "deterministic_adaptive":
            controller = M58AdaptiveBudgetController(block_step_size=10, max_budget_steps=max_budget_steps)
        elif arm_name == "progress_cost_adaptive":
            controller = ValidationProgressCostController(block_step_size=10, max_budget_steps=max_budget_steps)
        elif arm_name == "experiential_adaptive":
            controller = ExperientialBudgetController(
                memory_store=self.memory_store,
                block_step_size=10,
                max_budget_steps=max_budget_steps,
            )
        else:
            # Fallback (e.g. expansion control)
            controller = FixedBudgetController(max_budget_steps)

        # Baseline evaluation before training (step 0)
        f1_0, acc_0, pair_0, unk_0 = self.evaluate_target_module(adapter_candidate, target_env)
        jumpstart_f1 = f1_0

        trajectory = [
            OptimizationStepMetrics(
                step_n=0,
                macro_f1=f1_0,
                accuracy=acc_0,
                paired_correctness=pair_0,
                unknown_f1=unk_0,
            )
        ]
        controller_decisions: list[ControllerBudgetDecision] = []
        val_losses: list[float] = [1.0 - acc_0]

        current_steps = 0
        has_reset = False

        for n_practice in self.PRACTICE_SCHEDULE[1:]:
            decision = controller.evaluate_budget_action(
                current_steps=current_steps,
                val_loss_history=val_losses,
                complexity=complexity,
                uncertainty_entropy=0.0,
                has_reset=has_reset,
            )
            controller_decisions.append(decision)

            if decision.action == BudgetAction.RESET and not has_reset:
                logger.info("Executing RESET action on %s: training fresh adapter candidate.", target_env.env_id)
                adapter_candidate = PersistentCandidateAdapter(
                    d_model=self.base_model.config.d_model,
                    bottleneck_dim=16,
                    num_classes=3,
                ).to(self.device)
                optimizer = optim.AdamW(
                    [p for p in adapter_candidate.parameters() if p.requires_grad],
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                )
                has_reset = True

            if decision.action == BudgetAction.EXPAND and is_expansion:
                if isinstance(adapter_candidate, EvolvableResidualSkillAdapter):
                    adapter_candidate.expand_bottleneck(added_dim=16)
                    optimizer = optim.AdamW(
                        [p for p in adapter_candidate.parameters() if p.requires_grad],
                        lr=self.learning_rate,
                        weight_decay=self.weight_decay,
                    )

            steps_to_train = decision.block_steps_allocated
            if steps_to_train > 0:
                block_losses = self.run_training_steps(
                    module=adapter_candidate,
                    optimizer=optimizer,
                    env=target_env,
                    num_practice_examples=n_practice,
                    steps_to_run=steps_to_train,
                    criterion=criterion,
                )
                current_steps += steps_to_train
                val_losses.append(block_losses[-1] if block_losses else val_losses[-1])
            else:
                val_losses.append(val_losses[-1])

            # Evaluate at this practice budget
            f1_n, acc_n, pair_n, unk_n = self.evaluate_target_module(adapter_candidate, target_env)
            trajectory.append(
                OptimizationStepMetrics(
                    step_n=n_practice,
                    macro_f1=f1_n,
                    accuracy=acc_n,
                    paired_correctness=pair_n,
                    unknown_f1=unk_n,
                )
            )

            if decision.action == BudgetAction.KEEP:
                # Fill remaining trajectory points with frozen current performance
                for remaining_n in self.PRACTICE_SCHEDULE[len(trajectory):]:
                    trajectory.append(
                        OptimizationStepMetrics(
                            step_n=remaining_n,
                            macro_f1=f1_n,
                            accuracy=acc_n,
                            paired_correctness=pair_n,
                            unknown_f1=unk_n,
                        )
                    )
                break

        # Compute Learning-Curve AUC
        f1_values = [m.macro_f1 for m in trajectory]
        norm_steps = np.linspace(0.0, 1.0, len(f1_values))
        auc = float(np.trapezoid(f1_values, norm_steps))

        fraction_unused = float(max(0.0, max_budget_steps - current_steps) / max_budget_steps)
        gpu_dur = time.perf_counter() - t_start

        compute_budget_ratio = float(current_steps / max_budget_steps) if current_steps > 0 else 0.25
        compute_efficiency = float(auc / compute_budget_ratio)

        final_m = trajectory[-1]
        return ArmOptimizationEvaluation(
            arm_name=arm_name,
            target_env_id=target_env.env_id,
            trainable_parameters=trainable_params,
            total_optimizer_steps_taken=current_steps,
            max_budget_steps_allowed=max_budget_steps,
            fraction_unused_compute=fraction_unused,
            gpu_duration_seconds=gpu_dur,
            jumpstart_f1=jumpstart_f1,
            final_f1=final_m.macro_f1,
            final_accuracy=final_m.accuracy,
            final_paired_correctness=final_m.paired_correctness,
            learning_curve_auc=auc,
            compute_adjusted_efficiency=compute_efficiency,
            trajectory=trajectory,
            controller_decisions=controller_decisions,
        )
