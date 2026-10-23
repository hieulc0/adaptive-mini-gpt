"""Phase 3, 4 & 5: Controlled Multi-Budget Skill Composition and Learning Efficiency Benchmark."""
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
from adaptive_mini_gpt.learning_driven_experiment.minimal_pairs import MinimalPairEngine
from adaptive_mini_gpt.model.modular_adapter import compute_base_model_hash
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.skill_composition.composite_adapter import (
    GatedResidualSkillAdapter,
    ResidualSkillAdapter,
)

logger = logging.getLogger(__name__)


@dataclass
class CompositionBudgetStep:
    """Evaluation metrics at a specific adaptation example budget N."""
    budget_examples: int
    accuracy: float
    macro_f1: float
    paired_correctness: float
    unknown_precision: float
    unknown_recall: float


@dataclass
class CompositionArmTrajectory:
    """Adaptation trajectory and parameter footprint for a skill composition strategy."""
    arm_name: str
    composition_type: str  # "fresh", "fine_tuning", "frozen_residual", "gated_residual", "incompatible_residual"
    total_parameters: int
    trainable_parameters: int
    trajectory: list[CompositionBudgetStep]
    auc_macro_f1: float
    jumpstart_f1: float
    final_f1: float
    final_accuracy: float
    final_paired_correctness: float
    examples_to_threshold_50_f1: int | None
    training_duration_seconds: float


@dataclass
class MultiWorldCompositionReport:
    """Comprehensive comparative evaluation of skill composition across multiple target worlds."""
    target_task_id: str
    arms: dict[str, CompositionArmTrajectory]
    residual_vs_fresh_auc_gain: float  # AUC_residual - AUC_fresh
    gated_vs_fresh_auc_gain: float     # AUC_gated - AUC_fresh
    residual_vs_finetune_auc_gain: float  # AUC_residual - AUC_finetune
    gated_vs_finetune_auc_gain: float     # AUC_gated - AUC_finetune
    residual_vs_incompatible_gap: float   # AUC_residual - AUC_incompatible
    persistent_transfer_advantage_demonstrated: bool
    base_parameter_hash_invariant: bool


class CompositionTransferEngine:
    """Evaluates representation-preserving skill composition across practice budgets."""

    CLASS_LABELS = ["YES", "NO", "UNKNOWN"]
    BUDGETS = [0, 2, 4, 8, 16, 32]

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: Any,
        device: torch.device,
        seed: int = 42,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.seed = seed
        self.base_hash_initial = compute_base_model_hash(base_model)

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

    def evaluate_model(
        self,
        model_module: nn.Module,
        env: AutonomousEnvironment,
    ) -> CompositionBudgetStep:
        """Evaluate accuracy, macro F1, paired correctness, and UNKNOWN metrics."""
        model_module.eval()
        self.base_model.eval()

        predictions: list[str] = []
        ground_truth: list[str] = []

        with torch.no_grad():
            for q in env.held_out_test_queries:
                h = self._extract_hidden(q.question_text)
                logits = model_module(h)
                pred_idx = int(torch.argmax(logits, dim=-1).item())
                predictions.append(self.CLASS_LABELS[pred_idx])
                ground_truth.append(q.expected_answer)

        acc = sum(1 for p, g in zip(predictions, ground_truth) if p == g) / max(1, len(ground_truth))

        f1s: list[float] = []
        unk_prec = 0.0
        unk_rec = 0.0
        for label in self.CLASS_LABELS:
            tp = sum(1 for p, g in zip(predictions, ground_truth) if p == label and g == label)
            fp = sum(1 for p, g in zip(predictions, ground_truth) if p == label and g != label)
            fn = sum(1 for p, g in zip(predictions, ground_truth) if p != label and g == label)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
            if label == "UNKNOWN":
                unk_prec = prec
                unk_rec = rec

        macro_f1 = float(np.mean(f1s))

        def predict_fn(text: str) -> str:
            with torch.no_grad():
                h = self._extract_hidden(text)
                logits = model_module(h)
                return self.CLASS_LABELS[int(torch.argmax(logits, dim=-1).item())]

        pair_res = MinimalPairEngine.evaluate_paired_correctness(predict_fn, env.minimal_pairs)

        return CompositionBudgetStep(
            budget_examples=0,
            accuracy=float(acc),
            macro_f1=macro_f1,
            paired_correctness=pair_res.paired_correctness_rate,
            unknown_precision=unk_prec,
            unknown_recall=unk_rec,
        )

    def train_adapter_combined(
        self,
        model_module: nn.Module,
        env: AutonomousEnvironment,
        examples: list[tuple[str, str, str]],
        epochs: int = 12,
        lr: float = 3e-3,
    ) -> None:
        """Train trainable parameters of model_module using combined pointwise + paired supervision."""
        if not examples:
            return

        trainable_params = [p for p in model_module.parameters() if p.requires_grad]
        if not trainable_params:
            return

        optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()
        model_module.train()

        training_pairs = env.minimal_pairs[:min(4, len(env.minimal_pairs))] if env.minimal_pairs else []

        for _ in range(epochs):
            # 1. Pointwise loss
            for u, v, ans in examples:
                target = torch.tensor([self.CLASS_LABELS.index(ans)], dtype=torch.long, device=self.device)
                prompt = f"Does {u} {env.relation_name} {v}?"
                with torch.no_grad():
                    h = self._extract_hidden(prompt)
                logits = model_module(h)
                loss = criterion(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 2. Structural paired contrastive loss
            for pair in training_pairs:
                with torch.no_grad():
                    h_b = self._extract_hidden(pair.base_query.question_text)
                    h_p = self._extract_hidden(pair.perturbed_query.question_text)
                l_b = model_module(h_b)
                l_p = model_module(h_p)
                y_b = torch.tensor([self.CLASS_LABELS.index(pair.base_query.expected_answer)], dtype=torch.long, device=self.device)
                y_p = torch.tensor([self.CLASS_LABELS.index(pair.perturbed_query.expected_answer)], dtype=torch.long, device=self.device)
                loss_pair = 0.5 * (criterion(l_b, y_b) + criterion(l_p, y_p))
                optimizer.zero_grad()
                loss_pair.backward()
                optimizer.step()

    def run_arm_trajectory(
        self,
        arm_name: str,
        composition_type: str,
        factory_fn: Any,
        env: AutonomousEnvironment,
        all_candidate_examples: list[tuple[str, str, str]],
    ) -> CompositionArmTrajectory:
        """Execute adaptation curve across N in [0, 2, 4, 8, 16, 32] for a specific arm."""
        initial_module = factory_fn().to(self.device)
        total_p = sum(p.numel() for p in initial_module.parameters())
        trainable_p = sum(p.numel() for p in initial_module.parameters() if p.requires_grad)

        trajectory: list[CompositionBudgetStep] = []
        t0 = time.perf_counter()

        # Step 0: Zero-shot evaluation
        step0 = self.evaluate_model(initial_module, env)
        step0.budget_examples = 0
        trajectory.append(step0)

        # Progression across practice budgets
        for b in self.BUDGETS[1:]:
            subset = all_candidate_examples[:b]
            module_copy = factory_fn().to(self.device)
            self.train_adapter_combined(module_copy, env, subset, epochs=12)
            eval_step = self.evaluate_model(module_copy, env)
            eval_step.budget_examples = b
            trajectory.append(eval_step)

        duration = time.perf_counter() - t0

        x_vals = [s.budget_examples for s in trajectory]
        y_vals = [s.macro_f1 for s in trajectory]
        auc = float(np.trapz(y_vals, x_vals) / max(1, x_vals[-1]))

        thresh_steps = [s.budget_examples for s in trajectory if s.macro_f1 >= 0.50]
        ex_to_thresh = thresh_steps[0] if thresh_steps else None

        return CompositionArmTrajectory(
            arm_name=arm_name,
            composition_type=composition_type,
            total_parameters=total_p,
            trainable_parameters=trainable_p,
            trajectory=trajectory,
            auc_macro_f1=auc,
            jumpstart_f1=trajectory[0].macro_f1,
            final_f1=trajectory[-1].macro_f1,
            final_accuracy=trajectory[-1].accuracy,
            final_paired_correctness=trajectory[-1].paired_correctness,
            examples_to_threshold_50_f1=ex_to_thresh,
            training_duration_seconds=duration,
        )

    def run_multi_arm_composition(
        self,
        target_task_id: str,
        env: AutonomousEnvironment,
        source_adapter: PersistentCandidateAdapter,
        incompatible_adapter: PersistentCandidateAdapter,
    ) -> MultiWorldCompositionReport:
        """Execute 5-arm comparative composition study on an unfamiliar target environment."""
        entities = env.entities
        pairs = [(u, v) for u in entities for v in entities if u != v]
        rng = np.random.RandomState(self.seed)
        rng.shuffle(pairs)
        candidate_examples = [
            (u, v, env.evaluate_ground_truth(u, v))
            for u, v in pairs[:max(self.BUDGETS)]
        ]

        # 5 Arm Factory Functions
        arms: dict[str, CompositionArmTrajectory] = {}

        # Arm A: Fresh Baseline
        arms["fresh"] = self.run_arm_trajectory(
            arm_name="fresh",
            composition_type="fresh",
            factory_fn=lambda: PersistentCandidateAdapter(d_model=128, bottleneck_dim=32, num_classes=3),
            env=env,
            all_candidate_examples=candidate_examples,
        )

        # Arm B: Fine-Tuning (M54 baseline)
        arms["fine_tuning"] = self.run_arm_trajectory(
            arm_name="fine_tuning",
            composition_type="fine_tuning",
            factory_fn=lambda: copy.deepcopy(source_adapter),
            env=env,
            all_candidate_examples=candidate_examples,
        )

        # Arm C: Frozen Source + Residual
        arms["frozen_residual"] = self.run_arm_trajectory(
            arm_name="frozen_residual",
            composition_type="frozen_residual",
            factory_fn=lambda: ResidualSkillAdapter(
                source_adapter=source_adapter,
                d_model=128,
                residual_bottleneck_dim=16,
                num_classes=3,
            ),
            env=env,
            all_candidate_examples=candidate_examples,
        )

        # Arm D: Frozen Source + Gated Residual
        arms["gated_residual"] = self.run_arm_trajectory(
            arm_name="gated_residual",
            composition_type="gated_residual",
            factory_fn=lambda: GatedResidualSkillAdapter(
                source_adapter=source_adapter,
                d_model=128,
                residual_bottleneck_dim=16,
                num_classes=3,
            ),
            env=env,
            all_candidate_examples=candidate_examples,
        )

        # Arm E: Incompatible Frozen Source + Residual (Negative Control)
        arms["incompatible_control"] = self.run_arm_trajectory(
            arm_name="incompatible_control",
            composition_type="incompatible_residual",
            factory_fn=lambda: ResidualSkillAdapter(
                source_adapter=incompatible_adapter,
                d_model=128,
                residual_bottleneck_dim=16,
                num_classes=3,
            ),
            env=env,
            all_candidate_examples=candidate_examples,
        )

        auc_fresh = arms["fresh"].auc_macro_f1
        auc_finetune = arms["fine_tuning"].auc_macro_f1
        auc_res = arms["frozen_residual"].auc_macro_f1
        auc_gated = arms["gated_residual"].auc_macro_f1
        auc_incomp = arms["incompatible_control"].auc_macro_f1

        res_vs_fresh = auc_res - auc_fresh
        gated_vs_fresh = auc_gated - auc_fresh
        res_vs_finetune = auc_res - auc_finetune
        gated_vs_finetune = auc_gated - auc_finetune
        res_vs_incomp = auc_res - auc_incomp

        is_advantage = (res_vs_fresh > 0.01 or gated_vs_fresh > 0.01) and (res_vs_finetune > 0.01 or gated_vs_finetune > 0.01)

        current_base_hash = compute_base_model_hash(self.base_model)
        hash_ok = (current_base_hash == self.base_hash_initial)

        return MultiWorldCompositionReport(
            target_task_id=target_task_id,
            arms=arms,
            residual_vs_fresh_auc_gain=float(res_vs_fresh),
            gated_vs_fresh_auc_gain=float(gated_vs_fresh),
            residual_vs_finetune_auc_gain=float(res_vs_finetune),
            gated_vs_finetune_auc_gain=float(gated_vs_finetune),
            residual_vs_incompatible_gap=float(res_vs_incomp),
            persistent_transfer_advantage_demonstrated=is_advantage,
            base_parameter_hash_invariant=hash_ok,
        )
