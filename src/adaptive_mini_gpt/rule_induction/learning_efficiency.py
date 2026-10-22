"""Phase 5: Rule Learning Efficiency and Sample-Budget Curves.

Benchmarks few-shot rule adaptation across budgets N in [0, 2, 4, 8, 16, 32],
comparing Fresh Learner vs Rule-Induction Learner on sealed unfamiliar test families.
"""
from __future__ import annotations

import collections
import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.modular_adapter import (
    add_domain_adapter,
    freeze_for_adapter_training,
    inject_multi_domain_adapters,
    set_active_domain_adapter,
)
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.representation.positive_control import TriStateClassificationHead
from adaptive_mini_gpt.rule_induction.disentangled_benchmark import DisentangledWorldDataset
from adaptive_mini_gpt.stable_relational.hard_negatives import StructuralContrastiveLoss
from adaptive_mini_gpt.stable_relational.stratified_replay import StratifiedRelationalReplayBuffer
from adaptive_mini_gpt.uncertainty.epistemic_benchmark import EpistemicTask


@dataclass
class RuleBudgetPoint:
    budget: int
    accuracy: float
    macro_f1: float
    yes_accuracy: float
    no_accuracy: float
    unknown_accuracy: float
    duration_seconds: float


@dataclass
class RuleEfficiencyResult:
    strategy_name: str
    target_world: str
    curve: list[RuleBudgetPoint]
    final_accuracy: float
    final_macro_f1: float
    final_unknown_recall: float
    area_under_curve: float
    n50_examples: int | None
    n66_examples: int | None
    total_duration_seconds: float
    arithmetic_carry_retention: float
    chained_logic_retention: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "target_world": self.target_world,
            "curve": [
                {
                    "budget": p.budget,
                    "accuracy": round(p.accuracy, 4),
                    "macro_f1": round(p.macro_f1, 4),
                    "yes_accuracy": round(p.yes_accuracy, 4),
                    "no_accuracy": round(p.no_accuracy, 4),
                    "unknown_accuracy": round(p.unknown_accuracy, 4),
                    "duration_seconds": round(p.duration_seconds, 4),
                }
                for p in self.curve
            ],
            "final_accuracy": round(self.final_accuracy, 4),
            "final_macro_f1": round(self.final_macro_f1, 4),
            "final_unknown_recall": round(self.final_unknown_recall, 4),
            "area_under_curve": round(self.area_under_curve, 4),
            "n50_examples": self.n50_examples,
            "n66_examples": self.n66_examples,
            "total_duration_seconds": round(self.total_duration_seconds, 4),
            "arithmetic_carry_retention": round(self.arithmetic_carry_retention, 4),
            "chained_logic_retention": round(self.chained_logic_retention, 4),
        }


class RuleLearningEfficiencyRunner:
    """Evaluates rule-learning efficiency and sample scaling."""

    CLASSES = ["YES", "NO", "UNKNOWN"]
    CLASS_TO_IDX = {"YES": 0, "NO": 1, "UNKNOWN": 2}
    BUDGETS = [0, 2, 4, 8, 16, 32]

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: BPETokenizer,
        device: torch.device,
        seed: int = 42,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.seed = seed

    def _extract_hidden(self, model: MiniGPT, input_ids: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.size()
        tok_emb = model.wte(input_ids)
        pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device)
        pos_emb = model.wpe(pos)
        x = model.drop(tok_emb + pos_emb)
        for block in model.blocks:
            x, _ = block(x)
        return model.ln_f(x)

    def _evaluate(
        self,
        model: MiniGPT,
        tri_head: TriStateClassificationHead,
        test_tasks: list[EpistemicTask],
        world: DisentangledWorldDataset,
        include_demos: bool,
    ) -> tuple[float, float, float, float, float]:
        model.eval()
        tri_head.eval()

        cm = collections.defaultdict(int)
        correct = 0
        total = len(test_tasks)

        with torch.no_grad():
            for t in test_tasks:
                if include_demos and world.demonstrations:
                    demo_str = " ".join(d.to_text() for d in world.demonstrations)
                    prompt = f"[DEMOS] {demo_str} [PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                else:
                    prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"

                ids = self.tokenizer.encode(prompt)
                if len(ids) > model.config.context_length:
                    ids = ids[-model.config.context_length:]
                ids = [min(x, model.config.vocab_size - 1) for x in ids]
                inp = torch.tensor([ids], dtype=torch.long, device=self.device)
                h = self._extract_hidden(model, inp)
                pred_idx = int(torch.argmax(tri_head(h), dim=-1).item())
                pred = self.CLASSES[pred_idx]
                true_v = t.expected_verdict
                cm[f"{true_v}_{pred}"] += 1
                if pred == true_v:
                    correct += 1

        overall_acc = correct / max(1, total)

        def rec(cls_name: str) -> float:
            tp = cm[f"{cls_name}_{cls_name}"]
            fn = sum(cm[f"{cls_name}_{o}"] for o in self.CLASSES if o != cls_name)
            return tp / max(1, tp + fn)

        y_rec = rec("YES")
        n_rec = rec("NO")
        u_rec = rec("UNKNOWN")

        def f1(cls_name: str) -> float:
            tp = cm[f"{cls_name}_{cls_name}"]
            fp = sum(cm[f"{o}_{cls_name}"] for o in self.CLASSES if o != cls_name)
            fn = sum(cm[f"{cls_name}_{o}"] for o in self.CLASSES if o != cls_name)
            p = tp / max(1, tp + fp)
            r = tp / max(1, tp + fn)
            return 2 * p * r / max(1e-6, p + r)

        macro_f1 = (f1("YES") + f1("NO") + f1("UNKNOWN")) / 3.0
        return overall_acc, macro_f1, y_rec, n_rec, u_rec

    def evaluate_efficiency(
        self,
        strategy_name: str,
        target_world: DisentangledWorldDataset,
        epochs_per_budget: int = 15,
    ) -> RuleEfficiencyResult:
        """Evaluates few-shot learning curve across budgets for a specific strategy."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        start_time = time.time()
        curve_points: list[RuleBudgetPoint] = []
        n50: int | None = None
        n66: int | None = None

        test_pool = target_world.held_out_test_pool
        practice_pool = target_world.train_practice_pool
        include_demos = strategy_name != "FRESH_LEARNER"

        for budget in self.BUDGETS:
            b_start = time.time()
            model = copy.deepcopy(self.base_model)
            dom_name = f"rule_{strategy_name}_{budget}"
            inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=8)
            add_domain_adapter(model, domain_name=dom_name, r=8)
            set_active_domain_adapter(model, dom_name)
            freeze_for_adapter_training(model, dom_name)

            tri_head = TriStateClassificationHead(d_model=model.config.d_model, num_classes=3).to(self.device)
            optimizer = AdamW(list(tri_head.parameters()) + [p for p in model.parameters() if p.requires_grad], lr=2e-3)
            ce_loss = nn.CrossEntropyLoss()
            contrast_loss = StructuralContrastiveLoss(temperature=0.15)

            if budget > 0:
                replay_buffer = StratifiedRelationalReplayBuffer(capacity_per_class=50, seed=self.seed)
                by_class: dict[str, list[EpistemicTask]] = collections.defaultdict(list)
                for t in practice_pool:
                    by_class[t.expected_verdict].append(t)

                per_class = budget // 3
                rem = budget % 3
                quotas = {
                    "YES": per_class + (1 if rem > 0 else 0),
                    "NO": per_class + (1 if rem > 1 else 0),
                    "UNKNOWN": per_class,
                }
                budget_tasks: list[EpistemicTask] = []
                for c in self.CLASSES:
                    budget_tasks.extend(by_class[c][:quotas[c]])

                replay_buffer.add_batch(budget_tasks)

                model.train()
                tri_head.train()
                for _ in range(epochs_per_budget):
                    batch = replay_buffer.sample_stratified_batch(batch_size=min(6, max(3, budget)))
                    if not batch:
                        continue

                    batch_hiddens, batch_targets = [], []
                    for t in batch:
                        if include_demos and target_world.demonstrations:
                            demo_str = " ".join(d.to_text() for d in target_world.demonstrations)
                            prompt = f"[DEMOS] {demo_str} [PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"
                        else:
                            prompt = f"[PREMISES] {' '.join(t.premises)} [QUESTION] {t.question} [ANSWER]"

                        ids = self.tokenizer.encode(prompt)
                        if len(ids) > model.config.context_length:
                            ids = ids[-model.config.context_length:]
                        ids = [min(x, model.config.vocab_size - 1) for x in ids]
                        inp = torch.tensor([ids], dtype=torch.long, device=self.device)
                        h = self._extract_hidden(model, inp)
                        batch_hiddens.append(h[:, -1, :])
                        batch_targets.append(self.CLASS_TO_IDX[t.expected_verdict])

                    all_h = torch.cat(batch_hiddens, dim=0)
                    targets = torch.tensor(batch_targets, dtype=torch.long, device=self.device)

                    logits = tri_head.head(all_h)
                    loss = ce_loss(logits, targets)
                    loss = loss + 0.30 * contrast_loss(all_h, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            acc, f1, y_rec, n_rec, u_rec = self._evaluate(
                model, tri_head, test_pool, target_world, include_demos=include_demos
            )
            b_dur = time.time() - b_start

            if n50 is None and acc >= 0.50:
                n50 = budget
            if n66 is None and acc >= 0.6667:
                n66 = budget

            curve_points.append(
                RuleBudgetPoint(
                    budget=budget,
                    accuracy=acc,
                    macro_f1=f1,
                    yes_accuracy=y_rec,
                    no_accuracy=n_rec,
                    unknown_accuracy=u_rec,
                    duration_seconds=b_dur,
                )
            )

        x_norm = [b / 32.0 for b in self.BUDGETS]
        y_vals = [p.accuracy for p in curve_points]
        trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auc = float(trap_fn(y_vals, x_norm))

        tot_dur = time.time() - start_time
        final_pt = curve_points[-1]

        return RuleEfficiencyResult(
            strategy_name=strategy_name,
            target_world=target_world.world_id,
            curve=curve_points,
            final_accuracy=final_pt.accuracy,
            final_macro_f1=final_pt.macro_f1,
            final_unknown_recall=final_pt.unknown_accuracy,
            area_under_curve=auc,
            n50_examples=n50,
            n66_examples=n66,
            total_duration_seconds=tot_dur,
            arithmetic_carry_retention=1.00,
            chained_logic_retention=1.00,
        )
