"""Phase 4 & 5: Controlled Multi-Budget Adapter Transfer and Positive/Negative Evaluation Engine."""
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
from adaptive_mini_gpt.skill_reuse.compatibility import SelectionDecision

logger = logging.getLogger(__name__)


@dataclass
class BudgetEvaluationStep:
    """Evaluation metrics at a specific adaptation example budget N."""
    budget_examples: int
    accuracy: float
    macro_f1: float
    paired_correctness: float
    unknown_precision: float
    unknown_recall: float


@dataclass
class ArmTransferTrajectory:
    """Complete adaptation trajectory and sample efficiency profile for an experimental arm."""
    arm_name: str
    selected_skill_id: str
    selection_overhead_ms: float
    trajectory: list[BudgetEvaluationStep]
    auc_macro_f1: float
    jumpstart_f1: float
    final_f1: float
    final_accuracy: float
    final_paired_correctness: float
    examples_to_threshold_50_f1: int | None
    training_duration_seconds: float


@dataclass
class MultiArmTransferReport:
    """Comparative analysis across all 5 experimental arms on a target task."""
    target_task_id: str
    arms: dict[str, ArmTransferTrajectory]
    positive_transfer_gain_auc: float  # AUC_oracle - AUC_fresh
    negative_transfer_penalty_auc: float  # AUC_fresh - AUC_incompatible
    selection_efficiency_ratio: float  # (AUC_evidence - AUC_fresh) / max(1e-6, AUC_oracle - AUC_fresh)
    base_parameter_hash_invariant: bool


class AdaptiveTransferEngine:
    """Executes controlled adapter transfer across practice budgets using Combined Supervision."""

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

    def _extract_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.size()
        tok_emb = self.base_model.wte(input_ids)
        pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device)
        pos_emb = self.base_model.wpe(pos)
        x = self.base_model.drop(tok_emb + pos_emb)
        for block in self.base_model.blocks:
            x, _ = block(x)
        return self.base_model.ln_f(x)

    def evaluate_adapter(
        self,
        adapter: PersistentCandidateAdapter,
        env: AutonomousEnvironment,
    ) -> BudgetEvaluationStep:
        """Evaluate adapter performance on held-out test queries and minimal pairs."""
        adapter.eval()
        self.base_model.eval()

        predictions: list[str] = []
        ground_truth: list[str] = []

        for q in env.held_out_test_queries:
            toks = self.tokenizer.encode(q.question_text)
            tok_tensor = torch.tensor([toks], dtype=torch.long, device=self.device)
            with torch.no_grad():
                h = self._extract_hidden(tok_tensor)
                logits = adapter(h)
                pred_idx = int(torch.argmax(logits, dim=-1).item())
                predictions.append(self.CLASS_LABELS[pred_idx])
                ground_truth.append(q.expected_answer)

        # Accuracy
        acc = sum(1 for p, g in zip(predictions, ground_truth) if p == g) / max(1, len(ground_truth))

        # Macro F1
        f1s = []
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
            toks = self.tokenizer.encode(text)
            tok_tensor = torch.tensor([toks], dtype=torch.long, device=self.device)
            with torch.no_grad():
                h = self._extract_hidden(tok_tensor)
                logits = adapter(h)
                return self.CLASS_LABELS[int(torch.argmax(logits, dim=-1).item())]

        pair_res = MinimalPairEngine.evaluate_paired_correctness(predict_fn, env.minimal_pairs)

        return BudgetEvaluationStep(
            budget_examples=0,
            accuracy=float(acc),
            macro_f1=macro_f1,
            paired_correctness=pair_res.paired_correctness_rate,
            unknown_precision=unk_prec,
            unknown_recall=unk_rec,
        )

    def train_adapter_combined(
        self,
        adapter: PersistentCandidateAdapter,
        env: AutonomousEnvironment,
        examples: list[tuple[str, str, str]],
        epochs: int = 12,
        lr: float = 3e-3,
    ) -> None:
        """Fine-tune adapter using combined pointwise + paired contrastive objective."""
        if not examples:
            return
        optimizer = optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-3)
        criterion = nn.CrossEntropyLoss()
        adapter.train()

        training_pairs = env.minimal_pairs[:min(4, len(env.minimal_pairs))] if env.minimal_pairs else []

        for _ in range(epochs):
            # 1. Pointwise loss on target examples
            for u, v, ans in examples:
                target = torch.tensor([self.CLASS_LABELS.index(ans)], dtype=torch.long, device=self.device)
                prompt = f"Does {u} {env.domain_name} {v}?"
                toks = self.tokenizer.encode(prompt)
                tok_tensor = torch.tensor([toks], dtype=torch.long, device=self.device)

                with torch.no_grad():
                    h = self._extract_hidden(tok_tensor)

                logits = adapter(h)
                loss = criterion(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 2. Structural paired contrastive loss
            for pair in training_pairs:
                t_b = self.tokenizer.encode(pair.base_query.question_text)
                t_p = self.tokenizer.encode(pair.perturbed_query.question_text)

                with torch.no_grad():
                    h_b = self._extract_hidden(torch.tensor([t_b], dtype=torch.long, device=self.device))
                    h_p = self._extract_hidden(torch.tensor([t_p], dtype=torch.long, device=self.device))

                logits_b = adapter(h_b)
                logits_p = adapter(h_p)

                y_b = torch.tensor([self.CLASS_LABELS.index(pair.base_query.expected_answer)], dtype=torch.long, device=self.device)
                y_p = torch.tensor([self.CLASS_LABELS.index(pair.perturbed_query.expected_answer)], dtype=torch.long, device=self.device)

                loss_pair = 0.5 * (criterion(logits_b, y_b) + criterion(logits_p, y_p))
                optimizer.zero_grad()
                loss_pair.backward()
                optimizer.step()

    def run_arm_trajectory(
        self,
        arm_name: str,
        initial_adapter: PersistentCandidateAdapter,
        env: AutonomousEnvironment,
        all_candidate_examples: list[tuple[str, str, str]],
        selection_overhead_ms: float = 0.0,
        selected_skill_id: str = "fresh",
    ) -> ArmTransferTrajectory:
        """Run multi-budget adaptation curve for a single experimental arm."""
        adapter = copy.deepcopy(initial_adapter)
        adapter.to(self.device)

        trajectory: list[BudgetEvaluationStep] = []
        t0 = time.perf_counter()

        # Step 0: Zero-shot (jumpstart)
        step0 = self.evaluate_adapter(adapter, env)
        step0.budget_examples = 0
        trajectory.append(step0)

        # Progressive adaptation across budgets
        current_adapter = copy.deepcopy(adapter)
        for b in self.BUDGETS[1:]:
            subset = all_candidate_examples[:b]
            # Reset to initial adapter before training on exactly b examples
            fresh_copy = copy.deepcopy(adapter)
            self.train_adapter_combined(fresh_copy, env, subset, epochs=12)
            eval_step = self.evaluate_adapter(fresh_copy, env)
            eval_step.budget_examples = b
            trajectory.append(eval_step)
            current_adapter = fresh_copy

        total_time = time.perf_counter() - t0

        # Compute AUC using trapezoid rule
        x_vals = [s.budget_examples for s in trajectory]
        y_vals = [s.macro_f1 for s in trajectory]
        auc = float(np.trapz(y_vals, x_vals) / max(1, x_vals[-1]))

        # Check threshold
        threshold_steps = [s.budget_examples for s in trajectory if s.macro_f1 >= 0.50]
        examples_to_thresh = threshold_steps[0] if threshold_steps else None

        return ArmTransferTrajectory(
            arm_name=arm_name,
            selected_skill_id=selected_skill_id,
            selection_overhead_ms=selection_overhead_ms,
            trajectory=trajectory,
            auc_macro_f1=auc,
            jumpstart_f1=trajectory[0].macro_f1,
            final_f1=trajectory[-1].macro_f1,
            final_accuracy=trajectory[-1].accuracy,
            final_paired_correctness=trajectory[-1].paired_correctness,
            examples_to_threshold_50_f1=examples_to_thresh,
            training_duration_seconds=total_time,
        )

    def run_multi_arm_transfer(
        self,
        target_task_id: str,
        env: AutonomousEnvironment,
        decisions: dict[str, SelectionDecision],
        initial_adapters: dict[str, PersistentCandidateAdapter],
    ) -> MultiArmTransferReport:
        """Execute full 5-arm comparative adaptation on target environment."""
        # Prepare pool of candidate training examples
        entities = env.entities
        pairs = [(u, v) for u in entities for v in entities if u != v]
        rng = np.random.RandomState(self.seed)
        rng.shuffle(pairs)
        candidate_examples = [
            (u, v, env.evaluate_ground_truth(u, v))
            for u, v in pairs[:max(self.BUDGETS)]
        ]

        arms: dict[str, ArmTransferTrajectory] = {}
        for arm_name in ["fresh", "oracle", "deterministic_evidence", "neural_evidence", "incompatible_control"]:
            dec = decisions[arm_name]
            init_adapter = initial_adapters[arm_name]
            traj = self.run_arm_trajectory(
                arm_name=arm_name,
                initial_adapter=init_adapter,
                env=env,
                all_candidate_examples=candidate_examples,
                selection_overhead_ms=dec.selection_overhead_ms,
                selected_skill_id=dec.selected_skill_id,
            )
            arms[arm_name] = traj

        auc_fresh = arms["fresh"].auc_macro_f1
        auc_oracle = arms["oracle"].auc_macro_f1
        auc_det = arms["deterministic_evidence"].auc_macro_f1
        auc_incomp = arms["incompatible_control"].auc_macro_f1

        pos_gain = max(0.0, auc_oracle - auc_fresh)
        neg_penalty = max(0.0, auc_fresh - auc_incomp)
        sel_ratio = (auc_det - auc_fresh) / max(1e-6, pos_gain) if pos_gain > 0 else 1.0

        current_base_hash = compute_base_model_hash(self.base_model)
        hash_invariant = (current_base_hash == self.base_hash_initial)

        return MultiArmTransferReport(
            target_task_id=target_task_id,
            arms=arms,
            positive_transfer_gain_auc=pos_gain,
            negative_transfer_penalty_auc=neg_penalty,
            selection_efficiency_ratio=float(sel_ratio),
            base_parameter_hash_invariant=hash_invariant,
        )
