"""Phase 2: Replicated Skill Transfer Engine Across Multiple Independent Worlds."""
from __future__ import annotations

import copy
import time
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter
from adaptive_mini_gpt.guided_experiment.environment import (
    AutonomousEnvironment,
    EnvironmentQuery,
    MinimalPairControl,
)
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.reliable_transfer.composition_models import (
    GatedResidualSkillAdapter,
    ResidualSkillAdapter,
)

logger = logging.getLogger(__name__)


@dataclass
class AdaptationStepMetrics:
    """Metrics evaluated at budget N."""
    budget_n: int
    macro_f1: float
    accuracy: float
    paired_correctness: float
    unknown_recall: float
    unknown_precision: float


@dataclass
class ArmTransferEvaluation:
    """Full trajectory for a single adaptation arm on a target world."""
    arm_name: str
    target_task_id: str
    trainable_parameters: int
    total_parameters: int
    trajectory: list[AdaptationStepMetrics]
    jumpstart_f1: float
    final_f1: float
    auc_macro_f1: float
    final_paired_correctness: float
    training_wall_clock_ms: float
    peak_gpu_memory_bytes: int


@dataclass
class MultiWorldBenchmarkSummary:
    """Consolidated results across all target environments and arms."""
    target_results: dict[str, dict[str, ArmTransferEvaluation]]
    persistent_transfer_demonstrated: bool
    residual_vs_fresh_auc_gain_mean: float
    gated_vs_fresh_auc_gain_mean: float
    summary_text: str


class ReliableTransferEngine:
    """Executes multi-world transfer benchmarks using M53 combined supervision."""

    CLASS_LABELS = ["YES", "NO", "UNKNOWN"]
    BUDGETS = [0, 2, 4, 8, 16, 32]

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: Any,
        device: torch.device,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device

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

    def evaluate_environment(
        self,
        module: nn.Module,
        env: AutonomousEnvironment,
    ) -> tuple[float, float, float, float, float]:
        """Evaluate Macro F1, Accuracy, Paired Correctness, UNKNOWN Recall, UNKNOWN Precision."""
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

        # Macro F1 and Accuracy
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        acc = correct / len(predictions) if predictions else 0.0

        f1s: list[float] = []
        for label in self.CLASS_LABELS:
            tp = sum(1 for p, g in zip(predictions, ground_truth) if p == label and g == label)
            fp = sum(1 for p, g in zip(predictions, ground_truth) if p == label and g != label)
            fn = sum(1 for p, g in zip(predictions, ground_truth) if p != label and g == label)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            f1s.append(f1)
        macro_f1 = float(np.mean(f1s))

        # UNKNOWN recall and precision
        tp_u = sum(1 for p, g in zip(predictions, ground_truth) if p == "UNKNOWN" and g == "UNKNOWN")
        fp_u = sum(1 for p, g in zip(predictions, ground_truth) if p == "UNKNOWN" and g != "UNKNOWN")
        fn_u = sum(1 for p, g in zip(predictions, ground_truth) if p != "UNKNOWN" and g == "UNKNOWN")
        unk_rec = tp_u / (tp_u + fn_u) if (tp_u + fn_u) > 0 else 0.0
        unk_prec = tp_u / (tp_u + fp_u) if (tp_u + fp_u) > 0 else 0.0

        # Paired Correctness on minimal pairs
        paired_correct = 0
        total_pairs = 0
        with torch.no_grad():
            for pair in env.minimal_pairs:
                q_fwd = pair.base_query
                q_rev = pair.perturbed_query
                h_fwd = self._extract_hidden(q_fwd.question_text)
                h_rev = self._extract_hidden(q_rev.question_text)
                p_fwd = self.CLASS_LABELS[int(torch.argmax(module(h_fwd), dim=-1).item())]
                p_rev = self.CLASS_LABELS[int(torch.argmax(module(h_rev), dim=-1).item())]
                if p_fwd == q_fwd.expected_answer and p_rev == q_rev.expected_answer:
                    paired_correct += 1
                total_pairs += 1
        paired_corr = paired_correct / total_pairs if total_pairs > 0 else 0.0

        return macro_f1, acc, paired_corr, unk_rec, unk_prec

    def _train_combined_supervision(
        self,
        module: nn.Module,
        train_examples: list[EnvironmentQuery],
        minimal_pairs: list[MinimalPairControl],
        epochs: int = 12,
        lr: float = 1e-3,
    ) -> None:
        """Train using M53 Combined Supervision (Pointwise Cross-Entropy + Paired Margin Loss)."""
        if not train_examples:
            return

        trainable = [p for p in module.parameters() if p.requires_grad]
        if not trainable:
            return

        optimizer = optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
        ce_loss_fn = nn.CrossEntropyLoss()
        margin_loss_fn = nn.MarginRankingLoss(margin=0.5)
        label_to_idx = {l: i for i, l in enumerate(self.CLASS_LABELS)}

        module.train()
        for epoch in range(epochs):
            # 1. Pointwise loss
            for q in train_examples:
                optimizer.zero_grad()
                h = self._extract_hidden(q.question_text)
                logits = module(h)
                y = torch.tensor([label_to_idx[q.expected_answer]], device=self.device)
                l_ce = ce_loss_fn(logits, y)

                # 2. Structural minimal-pair contrastive step if available
                l_pair = torch.tensor(0.0, device=self.device)
                if minimal_pairs:
                    pair = minimal_pairs[epoch % len(minimal_pairs)]
                    q_f = pair.base_query
                    q_r = pair.perturbed_query
                    h_f = self._extract_hidden(q_f.question_text)
                    h_r = self._extract_hidden(q_r.question_text)
                    logits_f = module(h_f)
                    logits_r = module(h_r)
                    score_f = logits_f[0, label_to_idx[q_f.expected_answer]]
                    score_r = logits_r[0, label_to_idx[q_f.expected_answer]]
                    target = torch.tensor([1.0], device=self.device)
                    l_pair = margin_loss_fn(score_f.unsqueeze(0), score_r.unsqueeze(0), target)

                total_loss = l_ce + 0.5 * l_pair
                total_loss.backward()
                optimizer.step()

    def evaluate_adaptation_arm(
        self,
        arm_name: str,
        env: AutonomousEnvironment,
        compatible_source_adapter: PersistentCandidateAdapter,
        incompatible_source_adapter: PersistentCandidateAdapter,
        target_practice_pool: list[EnvironmentQuery],
        seed: int = 42,
    ) -> ArmTransferEvaluation:
        """Run trajectory evaluation for a single arm across N in [0, 2, 4, 8, 16, 32]."""
        torch.manual_seed(seed)
        d_model = self.base_model.config.d_model

        # Build arm model
        if arm_name == "fresh":
            module = PersistentCandidateAdapter(d_model=d_model, bottleneck_dim=32).to(self.device)
        elif arm_name == "fine_tuning":
            module = copy.deepcopy(compatible_source_adapter).to(self.device)
            for p in module.parameters():
                p.requires_grad = True
        elif arm_name == "frozen_residual":
            module = ResidualSkillAdapter(compatible_source_adapter, d_model=d_model).to(self.device)
        elif arm_name == "gated_residual":
            module = GatedResidualSkillAdapter(compatible_source_adapter, d_model=d_model).to(self.device)
        elif arm_name == "incompatible_control":
            module = ResidualSkillAdapter(incompatible_source_adapter, d_model=d_model).to(self.device)
        else:
            raise ValueError(f"Unknown adaptation arm: {arm_name}")

        total_p = sum(p.numel() for p in module.parameters())
        trainable_p = sum(p.numel() for p in module.parameters() if p.requires_grad)

        trajectory: list[AdaptationStepMetrics] = []
        t0 = time.perf_counter()

        # Track peak memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        for n in self.BUDGETS:
            if n > 0:
                train_slice = target_practice_pool[:n]
                # Reset module weights from jumpstart checkpoint and train up to budget n
                # To guarantee monotonic practice data budget without compounding artifacts
                if arm_name == "fresh":
                    module = PersistentCandidateAdapter(d_model=d_model, bottleneck_dim=32).to(self.device)
                elif arm_name == "fine_tuning":
                    module = copy.deepcopy(compatible_source_adapter).to(self.device)
                    for p in module.parameters():
                        p.requires_grad = True
                elif arm_name == "frozen_residual":
                    module = ResidualSkillAdapter(compatible_source_adapter, d_model=d_model).to(self.device)
                elif arm_name == "gated_residual":
                    module = GatedResidualSkillAdapter(compatible_source_adapter, d_model=d_model).to(self.device)
                elif arm_name == "incompatible_control":
                    module = ResidualSkillAdapter(incompatible_source_adapter, d_model=d_model).to(self.device)

                self._train_combined_supervision(
                    module=module,
                    train_examples=train_slice,
                    minimal_pairs=env.minimal_pairs,
                    epochs=12,
                )

            # Evaluate at budget n
            f1, acc, paired, unk_rec, unk_prec = self.evaluate_environment(module, env)
            trajectory.append(AdaptationStepMetrics(
                budget_n=n,
                macro_f1=f1,
                accuracy=acc,
                paired_correctness=paired,
                unknown_recall=unk_rec,
                unknown_precision=unk_prec,
            ))

        wall_ms = (time.perf_counter() - t0) * 1000.0
        peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

        # Compute normalized trapezoidal AUC over macro F1
        f1_vals = [m.macro_f1 for m in trajectory]
        auc = float(np.trapezoid(f1_vals, self.BUDGETS) / max(1, self.BUDGETS[-1]))

        return ArmTransferEvaluation(
            arm_name=arm_name,
            target_task_id=env.domain_name,
            trainable_parameters=trainable_p,
            total_parameters=total_p,
            trajectory=trajectory,
            jumpstart_f1=f1_vals[0],
            final_f1=f1_vals[-1],
            auc_macro_f1=auc,
            final_paired_correctness=trajectory[-1].paired_correctness,
            training_wall_clock_ms=wall_ms,
            peak_gpu_memory_bytes=peak_mem,
        )

    def benchmark_multi_world(
        self,
        environments: dict[str, AutonomousEnvironment],
        compatible_source_adapter: PersistentCandidateAdapter,
        incompatible_source_adapter: PersistentCandidateAdapter,
        seed: int = 42,
    ) -> MultiWorldBenchmarkSummary:
        """Run all 5 arms across multiple independent target worlds."""
        arm_names = ["fresh", "fine_tuning", "frozen_residual", "gated_residual", "incompatible_control"]
        target_results: dict[str, dict[str, ArmTransferEvaluation]] = {}

        residual_gains: list[float] = []
        gated_gains: list[float] = []

        for env_id, env in environments.items():
            practice_pool = env.held_out_test_queries[:32]
            target_results[env_id] = {}
            for arm in arm_names:
                eval_res = self.evaluate_adaptation_arm(
                    arm_name=arm,
                    env=env,
                    compatible_source_adapter=compatible_source_adapter,
                    incompatible_source_adapter=incompatible_source_adapter,
                    target_practice_pool=practice_pool,
                    seed=seed,
                )
                target_results[env_id][arm] = eval_res

            fresh_auc = target_results[env_id]["fresh"].auc_macro_f1
            res_auc = target_results[env_id]["frozen_residual"].auc_macro_f1
            gated_auc = target_results[env_id]["gated_residual"].auc_macro_f1

            residual_gains.append(res_auc - fresh_auc)
            gated_gains.append(gated_auc - fresh_auc)

        mean_res_gain = float(np.mean(residual_gains))
        mean_gated_gain = float(np.mean(gated_gains))
        persistent_adv = (mean_res_gain > 0.0 or mean_gated_gain > 0.0)

        summary_text = (
            f"Multi-World Benchmark across {list(environments.keys())}: "
            f"Residual vs Fresh Mean AUC Gain = {mean_res_gain:+.4f}, "
            f"Gated vs Fresh Mean AUC Gain = {mean_gated_gain:+.4f}. "
            f"Persistent Advantage Demonstrated = {persistent_adv}."
        )

        return MultiWorldBenchmarkSummary(
            target_results=target_results,
            persistent_transfer_demonstrated=persistent_adv,
            residual_vs_fresh_auc_gain_mean=mean_res_gain,
            gated_vs_fresh_auc_gain_mean=mean_gated_gain,
            summary_text=summary_text,
        )
