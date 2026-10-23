"""Phase 3, 4, 6: Controlled Capacity Evolution Benchmark Engine.

Compares 6 adaptation arms:
  Arm A: Fresh Adapter Baseline (4,483 params)
  Arm B: Fixed Small Residual (d_res=16, 2,371 params)
  Arm C: Larger Residual Fixed Capacity (d_res=32, 4,483 params)
  Arm D: Compute-Matched Small Residual (d_res=16, 2x optimization budget)
  Arm E: Adaptive Capacity Evolution (Controller: KEEP / EXPAND / RESET)
  Arm F: Incompatible Source + Capacity Expansion (Negative Control)
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
from adaptive_mini_gpt.capacity_evolution.capacity_controller import (
    AdaptiveCapacityController,
    CapacityAction,
    ControllerDecision,
)
from adaptive_mini_gpt.capacity_evolution.evolvable_adapter import EvolvableResidualSkillAdapter
from adaptive_mini_gpt.capacity_evolution.task_complexity import (
    ObservableTaskComplexity,
    compute_observable_complexity,
)

logger = logging.getLogger(__name__)


@dataclass
class AdaptationStepMetrics:
    """Evaluation metrics at a specific adaptation practice budget N."""
    step_n: int
    macro_f1: float
    accuracy: float
    paired_correctness: float
    unknown_f1: float


@dataclass
class ArmEvolutionEvaluation:
    """Complete adaptation trajectory and resource profile for one experimental arm."""
    arm_name: str
    target_env_id: str
    trainable_parameters: int
    frozen_parameters: int
    total_parameters: int
    jumpstart_f1: float
    final_f1: float
    final_paired_correctness: float
    auc_macro_f1: float
    adaptation_steps: list[AdaptationStepMetrics]
    training_duration_seconds: float
    peak_memory_mb: float
    controller_decision: ControllerDecision | None = None


@dataclass
class MultiTaskEvolutionBenchmarkSummary:
    """Comprehensive summary across all environments and arms."""
    task_results: dict[str, dict[str, ArmEvolutionEvaluation]]
    complexity_profiles: dict[str, ObservableTaskComplexity]


class ControlledCapacityEvolutionEngine:
    """Executes controlled multi-arm capacity evolution benchmarks using M53 Combined Supervision."""

    CLASS_LABELS = ["YES", "NO", "UNKNOWN"]
    ADAPTATION_BUDGETS = [0, 2, 4, 8, 16, 32]

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: Any,
        device: torch.device,
        controller: AdaptiveCapacityController | None = None,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.controller = controller or AdaptiveCapacityController()
        self.label_to_idx = {l: i for i, l in enumerate(self.CLASS_LABELS)}

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

    def evaluate_environment(self, module: nn.Module, env: AutonomousEnvironment) -> tuple[float, float, float, float]:
        """Compute neural-only Macro F1, accuracy, paired correctness, and UNKNOWN F1."""
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

        # Macro F1 & Accuracy
        f1s = []
        for label in self.CLASS_LABELS:
            tp = sum(1 for p, g in zip(predictions, ground_truth) if p == label and g == label)
            fp = sum(1 for p, g in zip(predictions, ground_truth) if p == label and g != label)
            fn = sum(1 for p, g in zip(predictions, ground_truth) if p != label and g == label)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)

        macro_f1 = float(np.mean(f1s))
        accuracy = float(sum(1 for p, g in zip(predictions, ground_truth) if p == g) / max(1, len(ground_truth)))
        unk_idx = self.CLASS_LABELS.index("UNKNOWN")
        unk_f1 = f1s[unk_idx]

        # Structural Paired Correctness on minimal pairs
        paired_correct = 0
        total_pairs = len(env.minimal_pairs)
        with torch.no_grad():
            for mp in env.minimal_pairs:
                h_base = self._extract_hidden(mp.base_query.question_text)
                pred_base = self.CLASS_LABELS[int(torch.argmax(module(h_base), dim=-1).item())]

                h_pert = self._extract_hidden(mp.perturbed_query.question_text)
                pred_pert = self.CLASS_LABELS[int(torch.argmax(module(h_pert), dim=-1).item())]

                if pred_base == mp.base_query.expected_answer and pred_pert == mp.perturbed_query.expected_answer:
                    paired_correct += 1

        paired_correctness = float(paired_correct / total_pairs) if total_pairs > 0 else 0.0
        return macro_f1, accuracy, paired_correctness, unk_f1

    def _train_combined_supervision(
        self,
        module: nn.Module,
        practice_queries: list[Any],
        minimal_pairs: list[MinimalPairControl],
        epochs: int = 15,
        lr: float = 1e-3,
    ) -> float:
        """Train candidate adapter using M53 Combined Supervision."""
        if not practice_queries:
            return 0.0

        trainable_params = [p for p in module.parameters() if p.requires_grad]
        if not trainable_params:
            return 0.0

        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
        ce_loss_fn = nn.CrossEntropyLoss()
        module.train()

        final_loss = 0.0
        for _ in range(epochs):
            total_loss = torch.tensor(0.0, device=self.device)
            # 1. Pointwise observation CE
            for q in practice_queries:
                optimizer.zero_grad()
                h = self._extract_hidden(q.question_text)
                logits = module(h)
                y = torch.tensor([self.label_to_idx[q.expected_answer]], device=self.device)
                pt_loss = ce_loss_fn(logits, y)

                # 2. Structural minimal-pair contrastive loss
                pair_loss = torch.tensor(0.0, device=self.device)
                if minimal_pairs:
                    mp = minimal_pairs[torch.randint(0, len(minimal_pairs), (1,)).item()]
                    h_b = self._extract_hidden(mp.base_query.question_text)
                    h_p = self._extract_hidden(mp.perturbed_query.question_text)
                    l_b = module(h_b)
                    l_p = module(h_p)
                    y_b = torch.tensor([self.label_to_idx[mp.base_query.expected_answer]], device=self.device)
                    y_p = torch.tensor([self.label_to_idx[mp.perturbed_query.expected_answer]], device=self.device)
                    pair_loss = 0.5 * (ce_loss_fn(l_b, y_b) + ce_loss_fn(l_p, y_p))

                loss = pt_loss + 0.5 * pair_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.detach()

            final_loss = float((total_loss / max(1, len(practice_queries))).item())

        return final_loss

    def evaluate_adaptation_arm(
        self,
        arm_name: str,
        target_env: AutonomousEnvironment,
        compatible_source_adapter: PersistentCandidateAdapter,
        incompatible_source_adapter: PersistentCandidateAdapter,
        seed: int = 42,
    ) -> ArmEvolutionEvaluation:
        """Evaluate a specific experimental arm over adaptation budgets N in [0, 2, 4, 8, 16, 32]."""
        torch.manual_seed(seed)
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        complexity = compute_observable_complexity(target_env)
        decision_trace: ControllerDecision | None = None

        # Instantiate module according to arm specification
        if arm_name == "fresh":
            module = PersistentCandidateAdapter(
                d_model=self.base_model.config.d_model,
                bottleneck_dim=32,
                num_classes=3,
            ).to(self.device)
            epochs_per_step = 15

        elif arm_name == "fixed_small_residual":
            # Baseline M56: d_res = 16 (2,371 parameters)
            module = EvolvableResidualSkillAdapter(
                source_adapter=compatible_source_adapter,
                d_model=self.base_model.config.d_model,
                initial_bottleneck_dim=16,
            ).to(self.device)
            epochs_per_step = 15

        elif arm_name == "fixed_large_residual":
            # Arm C: larger residual fixed capacity (d_res = 32, 4,483 parameters)
            module = EvolvableResidualSkillAdapter(
                source_adapter=compatible_source_adapter,
                d_model=self.base_model.config.d_model,
                initial_bottleneck_dim=32,
            ).to(self.device)
            epochs_per_step = 15

        elif arm_name == "compute_matched_small":
            # Arm D: small residual (d_res = 16) trained with 2x epochs to match compute of Arm C
            module = EvolvableResidualSkillAdapter(
                source_adapter=compatible_source_adapter,
                d_model=self.base_model.config.d_model,
                initial_bottleneck_dim=16,
            ).to(self.device)
            epochs_per_step = 30  # Double training budget

        elif arm_name == "incompatible_expanded":
            # Arm F: Incompatible source + expanded residual (d_res = 32)
            module = EvolvableResidualSkillAdapter(
                source_adapter=incompatible_source_adapter,
                d_model=self.base_model.config.d_model,
                initial_bottleneck_dim=32,
            ).to(self.device)
            epochs_per_step = 15

        elif arm_name == "adaptive_controller":
            # Arm E: Adaptive residual capacity evolution
            module = EvolvableResidualSkillAdapter(
                source_adapter=compatible_source_adapter,
                d_model=self.base_model.config.d_model,
                initial_bottleneck_dim=16,
            ).to(self.device)
            epochs_per_step = 15

        else:
            raise ValueError(f"Unknown experimental arm: {arm_name}")

        # Compute initial zero-shot metrics (N=0)
        param_counts = module.count_parameters() if hasattr(module, "count_parameters") else {
            "total": sum(p.numel() for p in module.parameters()),
            "trainable": sum(p.numel() for p in module.parameters() if p.requires_grad),
            "frozen": sum(p.numel() for p in module.parameters() if not p.requires_grad),
        }

        m_f1_0, acc_0, pair_0, unk_0 = self.evaluate_environment(module, target_env)
        step_metrics: list[AdaptationStepMetrics] = [
            AdaptationStepMetrics(step_n=0, macro_f1=m_f1_0, accuracy=acc_0, paired_correctness=pair_0, unknown_f1=unk_0)
        ]

        val_loss_0 = 1.0 - m_f1_0  # Proxy validation signal
        current_val_loss = val_loss_0

        # Adaptation training loop over practice budgets
        for i, budget in enumerate(self.ADAPTATION_BUDGETS[1:], start=1):
            practice_pool = target_env.held_out_test_queries[:budget]

            # Controller intervention point at N=4 or N=8 for Arm E
            if arm_name == "adaptive_controller" and budget in (4, 8) and decision_trace is None:
                decision = self.controller.evaluate_action(
                    step_n=budget,
                    val_loss_initial=val_loss_0,
                    val_loss_current=current_val_loss,
                    complexity=complexity,
                )
                decision_trace = decision
                logger.info(
                    "Controller Action at N=%d on %s: %s (Reason: %s)",
                    budget, target_env.env_id, decision.action.value, decision.rationale,
                )

                if decision.action == CapacityAction.EXPAND:
                    # Execute reversible expansion from 16 to 32 dimensions
                    module.expand_capacity(target_bottleneck_dim=32)
                elif decision.action == CapacityAction.RESET:
                    # Switch to fresh adapter
                    module = PersistentCandidateAdapter(
                        d_model=self.base_model.config.d_model,
                        bottleneck_dim=32,
                        num_classes=3,
                    ).to(self.device)

            # Train on cumulative verified examples
            train_loss = self._train_combined_supervision(
                module=module,
                practice_queries=practice_pool,
                minimal_pairs=target_env.minimal_pairs,
                epochs=epochs_per_step,
            )
            current_val_loss = train_loss

            # Evaluate neural-only performance post-adaptation step
            m_f1, acc, pair, unk = self.evaluate_environment(module, target_env)
            step_metrics.append(
                AdaptationStepMetrics(step_n=budget, macro_f1=m_f1, accuracy=acc, paired_correctness=pair, unknown_f1=unk)
            )

        duration = time.perf_counter() - t0
        peak_mem = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available() else 0.0
        )

        # Compute normalized Learning Curve AUC via trapezoidal integration
        ns = np.array(self.ADAPTATION_BUDGETS, dtype=float)
        f1s = np.array([m.macro_f1 for m in step_metrics], dtype=float)
        auc_f1 = float(np.trapezoid(f1s, ns) / (ns[-1] - ns[0]))

        # Final parameter count reflects any expansion
        final_param_counts = module.count_parameters() if hasattr(module, "count_parameters") else param_counts

        return ArmEvolutionEvaluation(
            arm_name=arm_name,
            target_env_id=target_env.env_id,
            trainable_parameters=final_param_counts["trainable"],
            frozen_parameters=final_param_counts["frozen"],
            total_parameters=final_param_counts["total"],
            jumpstart_f1=step_metrics[0].macro_f1,
            final_f1=step_metrics[-1].macro_f1,
            final_paired_correctness=step_metrics[-1].paired_correctness,
            auc_macro_f1=auc_f1,
            adaptation_steps=step_metrics,
            training_duration_seconds=duration,
            peak_memory_mb=peak_mem,
            controller_decision=decision_trace,
        )
