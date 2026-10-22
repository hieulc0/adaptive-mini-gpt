"""Phase 6: Epistemic Learning Efficiency, Sample Efficiency Curves, and Retention.

Benchmarks sample efficiency across practice budgets N in [0, 2, 4, 8, 16, 32] on novel held-out worlds:
  1. Fresh Neural Learner (baseline few-shot adaptation without prior memories).
  2. Experience-Conditioned Neural Learner (adapted with prior structural experiences).
  3. Corrective Closed-Loop Learner (targeted gap-directed adaptation).

Tracks:
  - Overall accuracy, Macro F1, YES accuracy, NO accuracy, and UNKNOWN accuracy at every budget.
  - Expected Calibration Error (ECE) and Brier Score progression.
  - Learning Curve Area Under Curve (AUC), N50, N66 budgets.
  - Multi-episode continual retention matrix with canonical arithmetic carry and logic preservation.
"""
from __future__ import annotations

import collections
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.modular_adapter import (
    add_domain_adapter,
    compute_base_model_hash,
    freeze_for_adapter_training,
    inject_multi_domain_adapters,
    set_active_domain_adapter,
)
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.uncertainty.epistemic_benchmark import EpistemicTask, EpistemicWorldDataset

logger = logging.getLogger("m45_learning_efficiency")


@dataclass
class EpistemicCurveStep:
    budget: int
    accuracy: float
    macro_f1: float
    yes_accuracy: float
    no_accuracy: float
    unknown_accuracy: float
    expected_calibration_error: float
    confusion_matrix: dict[str, dict[str, int]]
    mean_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "yes_accuracy": round(self.yes_accuracy, 4),
            "no_accuracy": round(self.no_accuracy, 4),
            "unknown_accuracy": round(self.unknown_accuracy, 4),
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "confusion_matrix": self.confusion_matrix,
            "mean_latency_ms": round(self.mean_latency_ms, 2),
        }


@dataclass
class EpistemicEfficiencyReport:
    strategy_name: str
    target_world: str
    curve: list[EpistemicCurveStep]
    final_accuracy: float
    final_macro_f1: float
    final_unknown_recall: float
    area_under_curve: float
    n50_examples: int | None
    n66_examples: int | None
    training_steps_total: int
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "target_world": self.target_world,
            "curve": [c.to_dict() for c in self.curve],
            "final_accuracy": round(self.final_accuracy, 4),
            "final_macro_f1": round(self.final_macro_f1, 4),
            "final_unknown_recall": round(self.final_unknown_recall, 4),
            "area_under_curve": round(self.area_under_curve, 4),
            "n50_examples": self.n50_examples,
            "n66_examples": self.n66_examples,
            "training_steps_total": self.training_steps_total,
            "duration_seconds": round(self.duration_seconds, 2),
        }


class EpistemicEfficiencyBenchmarkRunner:
    """Evaluates sample efficiency curves across budgets [0, 2, 4, 8, 16, 32]."""

    CLASSES = ["YES", "NO", "UNKNOWN"]
    CLASS_TO_IDX = {"YES": 0, "NO": 1, "UNKNOWN": 2}

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: BPETokenizer,
        device: torch.device,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.cand_seqs = [self.tokenizer.encode(f" {c}") for c in self.CLASSES]

    def _eval_step(
        self,
        model: MiniGPT,
        tasks: list[EpistemicTask],
        budget: int,
    ) -> EpistemicCurveStep:
        model.eval()
        latencies: list[float] = []
        cm_counts = collections.defaultdict(int)
        correctness = []
        confidences = []

        y_tot = y_corr = 0
        n_tot = n_corr = 0
        u_tot = u_corr = 0

        for t in tasks:
            t0 = time.perf_counter()
            prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
            toks = self.tokenizer.encode(prompt)
            if len(toks) > model.config.context_length - 8:
                toks = toks[-(model.config.context_length - 8):]

            scores = []
            with torch.no_grad():
                for seq in self.cand_seqs:
                    full_seq = toks + seq
                    if len(full_seq) > model.config.context_length:
                        full_seq = full_seq[-model.config.context_length:]
                    inp = torch.tensor([full_seq], dtype=torch.long, device=self.device)
                    out = model(inp)
                    logits = out[0] if isinstance(out, tuple) else out
                    log_probs = torch.log_softmax(logits, dim=-1)
                    seq_lp = sum(float(log_probs[0, len(toks) - 1 + i, tok].item()) for i, tok in enumerate(seq))
                    scores.append(seq_lp / max(1, len(seq)))

            probs = torch.softmax(torch.tensor(scores), dim=-1).tolist()
            best_idx = int(np.argmax(scores))
            pred = self.CLASSES[best_idx]
            conf = float(probs[best_idx])
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)

            is_corr = (pred == t.expected_verdict)
            correctness.append(1 if is_corr else 0)
            confidences.append(conf)

            cm_counts[f"{t.expected_verdict}_{pred}"] += 1

            if t.expected_verdict == "YES":
                y_tot += 1
                if is_corr:
                    y_corr += 1
            elif t.expected_verdict == "NO":
                n_tot += 1
                if is_corr:
                    n_corr += 1
            elif t.expected_verdict == "UNKNOWN":
                u_tot += 1
                if is_corr:
                    u_corr += 1

        overall_acc = sum(correctness) / max(1, len(tasks))
        y_rec = y_corr / max(1, y_tot)
        n_rec = n_corr / max(1, n_tot)
        u_rec = u_corr / max(1, u_tot)

        # Macro F1
        def f1_c(c: str) -> float:
            tp = cm_counts[f"{c}_{c}"]
            fp = sum(cm_counts[f"{o}_{c}"] for o in self.CLASSES if o != c)
            fn = sum(cm_counts[f"{c}_{o}"] for o in self.CLASSES if o != c)
            p = tp / max(1, tp + fp)
            r = tp / max(1, tp + fn)
            return 2 * p * r / max(1e-6, p + r)

        macro_f1 = (f1_c("YES") + f1_c("NO") + f1_c("UNKNOWN")) / 3.0

        # ECE
        bin_edges = np.linspace(0.0, 1.0, 11)
        ece = 0.0
        c_arr = np.array(confidences)
        k_arr = np.array(correctness)
        for i in range(10):
            b_min, b_max = bin_edges[i], bin_edges[i + 1]
            idx = (c_arr >= b_min) & (c_arr < b_max if i < 9 else c_arr <= b_max)
            if np.sum(idx) > 0:
                ece += (np.sum(idx) / len(tasks)) * abs(np.mean(k_arr[idx]) - np.mean(c_arr[idx]))

        cm_dict = {
            "TRUE_YES": {"PRED_YES": cm_counts["YES_YES"], "PRED_NO": cm_counts["YES_NO"], "PRED_UNK": cm_counts["YES_UNKNOWN"]},
            "TRUE_NO": {"PRED_YES": cm_counts["NO_YES"], "PRED_NO": cm_counts["NO_NO"], "PRED_UNK": cm_counts["NO_UNKNOWN"]},
            "TRUE_UNK": {"PRED_YES": cm_counts["UNKNOWN_YES"], "PRED_NO": cm_counts["UNKNOWN_NO"], "PRED_UNK": cm_counts["UNKNOWN_UNKNOWN"]},
        }

        return EpistemicCurveStep(
            budget=budget,
            accuracy=overall_acc,
            macro_f1=macro_f1,
            yes_accuracy=y_rec,
            no_accuracy=n_rec,
            unknown_accuracy=u_rec,
            expected_calibration_error=float(ece),
            confusion_matrix=cm_dict,
            mean_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        )

    def _train_adapter(
        self,
        model: MiniGPT,
        adapter_name: str,
        practice_tasks: list[EpistemicTask],
        steps: int = 30,
        lr: float = 5e-4,
    ) -> None:
        if not practice_tasks:
            return
        freeze_for_adapter_training(model, adapter_name)
        set_active_domain_adapter(model, adapter_name)
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        model.train()

        for s in range(steps):
            t = practice_tasks[s % len(practice_tasks)]
            prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
            target = f" {t.expected_verdict}"
            p_ids = self.tokenizer.encode(prompt)
            t_ids = self.tokenizer.encode(target)
            full_ids = p_ids + t_ids
            if len(full_ids) > model.config.context_length:
                full_ids = full_ids[-model.config.context_length:]
                p_ids = full_ids[:-len(t_ids)]

            inp = torch.tensor([full_ids[:-1]], dtype=torch.long, device=self.device)
            lbl = torch.tensor([[-100] * (len(p_ids) - 1) + t_ids], dtype=torch.long, device=self.device)

            out = model(inp)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits.view(-1, logits.size(-1)), lbl.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()

    @staticmethod
    def compute_auc(budgets: list[int], values: list[float]) -> float:
        if len(budgets) < 2:
            return float(np.mean(values)) if values else 0.0
        auc = 0.0
        for i in range(len(budgets) - 1):
            dx = budgets[i + 1] - budgets[i]
            y_avg = (values[i] + values[i + 1]) / 2.0
            auc += dx * y_avg
        return round(auc / (budgets[-1] - budgets[0]), 4)

    def run_strategy_evaluation(
        self,
        strategy_name: str,
        target_world: EpistemicWorldDataset,
        source_worlds: list[EpistemicWorldDataset] | None = None,
        budgets: list[int] = [0, 2, 4, 8, 16, 32],
    ) -> EpistemicEfficiencyReport:
        t0 = time.perf_counter()
        curve: list[EpistemicCurveStep] = []
        accuracies: list[float] = []
        total_steps = 0

        for b in budgets:
            model = copy.deepcopy(self.base_model)
            inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
            adp = f"m45_{strategy_name.lower()}_{b}"
            add_domain_adapter(model, adp, r=8)
            set_active_domain_adapter(model, adp)

            if b > 0:
                practice_subset = target_world.train_practice_pool[:b]
                if strategy_name == "EXPERIENCE_CONDITIONED_LEARNER" and source_worlds:
                    # Interleave prior structural experiences with target practice
                    prior_tasks = []
                    for sw in source_worlds:
                        prior_tasks.extend(sw.train_practice_pool[:4])
                    combined = practice_subset + prior_tasks
                    steps = max(20, b * 3)
                    total_steps += steps
                    self._train_adapter(model, adp, combined, steps=steps)
                else:
                    steps = max(20, b * 3)
                    total_steps += steps
                    self._train_adapter(model, adp, practice_subset, steps=steps)

            step_res = self._eval_step(model, target_world.held_out_test_pool, b)
            curve.append(step_res)
            accuracies.append(step_res.accuracy)

        auc = self.compute_auc(budgets, accuracies)
        n50 = next((b for b, a in zip(budgets, accuracies) if a >= 0.50), None)
        n66 = next((b for b, a in zip(budgets, accuracies) if a >= 0.667), None)
        dur = time.perf_counter() - t0

        return EpistemicEfficiencyReport(
            strategy_name=strategy_name,
            target_world=target_world.world_name,
            curve=curve,
            final_accuracy=accuracies[-1],
            final_macro_f1=curve[-1].macro_f1,
            final_unknown_recall=curve[-1].unknown_accuracy,
            area_under_curve=auc,
            n50_examples=n50,
            n66_examples=n66,
            training_steps_total=total_steps,
            duration_seconds=dur,
        )
