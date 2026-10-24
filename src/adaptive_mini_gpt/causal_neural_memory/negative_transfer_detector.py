"""Milestone M73 Phase 4: Rigorous Negative Transfer and Routing Error Detection.

Implements precise detection of harmful routing and negative transfer:
1. Compares the selected neural skill against a matched safe fallback baseline
   (the frozen backbone model or best active source skill) on the identical sealed evaluation queries.
2. Defines Statistical Negative Transfer:
   Selected Skill F1 < Safe Fallback F1 - epsilon_margin (where epsilon_margin >= 0.02).
3. Evaluates Operational Routing Metrics:
   - harmful_route_choices: routed skill underperforms safe fallback by >= epsilon_margin.
   - missed_beneficial_routes: available candidate outperforms selected route by >= epsilon_margin, but was missed.
   - successful_retained_reuses: retained candidate is selected and outperforms safe fallback by >= epsilon_margin.
   - incorrect_activations: candidate activated on an incompatible or adversarial rule family.
   - unnecessary_skill_allocations: new candidate trained when an existing skill was already proficient.
   - candidate_qualification_failures: candidate trained but rejected by qualification gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.modular_skill_learning.repertoire import extract_model_hidden

logger = logging.getLogger(__name__)


@dataclass
class TaskRoutingComparison:
    """Detailed evaluation comparing routed skill against safe fallback on identical sealed queries."""

    task_name: str
    rule_family: str
    is_adversarial: bool
    is_repeated: bool
    selected_skill_id: str
    safe_fallback_id: str
    selected_skill_f1: float
    safe_fallback_f1: float
    best_available_f1: float
    delta_vs_fallback: float
    delta_vs_best: float
    is_statistical_negative_transfer: bool
    is_harmful_route: bool
    is_missed_benefit: bool
    is_successful_retained_reuse: bool
    is_incorrect_activation: bool


@dataclass
class RoutingErrorSummary:
    """Summary of routing performance and negative transfer across a stream."""

    total_evaluations: int = 0
    harmful_route_choices: int = 0
    missed_beneficial_routes: int = 0
    successful_retained_reuses: int = 0
    incorrect_activations: int = 0
    unnecessary_skill_allocations: int = 0
    candidate_qualification_failures: int = 0
    statistical_negative_transfers: int = 0
    mean_delta_vs_fallback: float = 0.0
    episodes: list[TaskRoutingComparison] = field(default_factory=list)

    def summary_dict(self) -> dict[str, Any]:
        """Return serializable summary."""
        return {
            "total_evaluations": self.total_evaluations,
            "harmful_route_choices": self.harmful_route_choices,
            "missed_beneficial_routes": self.missed_beneficial_routes,
            "successful_retained_reuses": self.successful_retained_reuses,
            "incorrect_activations": self.incorrect_activations,
            "unnecessary_skill_allocations": self.unnecessary_skill_allocations,
            "candidate_qualification_failures": self.candidate_qualification_failures,
            "statistical_negative_transfers": self.statistical_negative_transfers,
            "mean_delta_vs_fallback": self.mean_delta_vs_fallback,
        }


class NegativeTransferAuditor:
    """Audits skill selection decisions against matched safe fallbacks."""

    def __init__(self, epsilon_margin: float = 0.02) -> None:
        self.epsilon_margin = epsilon_margin

    @staticmethod
    def evaluate_model_or_adapter_f1(
        adapter: nn.Module | None,
        model: MiniGPT,
        queries: list[Any],
        device: torch.device,
    ) -> float:
        """Evaluate Macro F1 on sealed queries with 3 classes (NO=0, YES=1, UNKNOWN=2)."""
        if not queries:
            return 0.3333

        lbl_map = {"NO": 0, "YES": 1, "UNKNOWN": 2}
        preds = []
        targets = []

        with torch.no_grad():
            for ex in queries:
                if isinstance(ex, dict):
                    q = ex.get("question_text", ex.get("text", ""))
                    ans = ex.get("expected_answer", ex.get("answer", "UNKNOWN"))
                else:
                    q = getattr(ex, "question_text", "")
                    ans = getattr(ex, "expected_answer", "UNKNOWN")
                tgt = lbl_map.get(ans, 2)
                targets.append(tgt)

                toks = model.tokenizer.encode(q)
                inp = torch.tensor([toks], dtype=torch.long, device=device)
                h = extract_model_hidden(model, inp)
                if adapter is not None:
                    logits = adapter(h)
                    pred = int(torch.argmax(logits, dim=-1).item())
                else:
                    pred = int(torch.argmax(h[0, :3]).item())
                preds.append(pred)

        preds_arr = np.array(preds)
        tgts_arr = np.array(targets)

        recalls = []
        for c in (0, 1, 2):
            mask = tgts_arr == c
            if np.sum(mask) > 0:
                recalls.append(float(np.sum((preds_arr == c) & mask) / np.sum(mask)))
        return float(np.mean(recalls)) if recalls else 0.3333

    def evaluate_task_transfer(
        self,
        task_name: str,
        rule_family: str,
        is_adversarial: bool,
        is_repeated: bool,
        selected_skill_id: str,
        selected_adapter: nn.Module | None,
        safe_fallback_id: str,
        safe_fallback_adapter: nn.Module | None,
        all_repertoire_skills: dict[str, nn.Module],
        model: MiniGPT,
        sealed_test_queries: list[Any],
        device: torch.device,
        unnecessary_allocation: bool = False,
        qualification_failure: bool = False,
    ) -> TaskRoutingComparison:
        """Compare selected route against matched safe fallback on identical sealed queries."""
        # 1. Evaluate selected skill
        selected_f1 = self.evaluate_model_or_adapter_f1(
            selected_adapter, model, sealed_test_queries, device
        )

        # 2. Evaluate safe fallback
        fallback_f1 = self.evaluate_model_or_adapter_f1(
            safe_fallback_adapter, model, sealed_test_queries, device
        )

        # 3. Evaluate best available skill in repertoire
        best_avail_f1 = fallback_f1
        for _, ad in all_repertoire_skills.items():
            f1 = self.evaluate_model_or_adapter_f1(ad, model, sealed_test_queries, device)
            if f1 > best_avail_f1:
                best_avail_f1 = f1

        delta_vs_fallback = selected_f1 - fallback_f1
        delta_vs_best = selected_f1 - best_avail_f1

        # Criteria
        is_stat_neg_transfer = bool(delta_vs_fallback < -self.epsilon_margin)
        is_harmful_route = bool(delta_vs_fallback < -self.epsilon_margin)
        is_missed_benefit = bool(delta_vs_best < -self.epsilon_margin)
        is_successful_retained = bool(
            is_repeated
            and selected_skill_id.startswith("cand_")
            and (delta_vs_fallback >= self.epsilon_margin)
        )
        is_incorrect_act = bool(
            is_adversarial
            and (selected_skill_id != "base_model" and not selected_skill_id.startswith("source_"))
            and is_stat_neg_transfer
        )

        return TaskRoutingComparison(
            task_name=task_name,
            rule_family=rule_family,
            is_adversarial=is_adversarial,
            is_repeated=is_repeated,
            selected_skill_id=selected_skill_id,
            safe_fallback_id=safe_fallback_id,
            selected_skill_f1=selected_f1,
            safe_fallback_f1=fallback_f1,
            best_available_f1=best_avail_f1,
            delta_vs_fallback=delta_vs_fallback,
            delta_vs_best=delta_vs_best,
            is_statistical_negative_transfer=is_stat_neg_transfer,
            is_harmful_route=is_harmful_route,
            is_missed_benefit=is_missed_benefit,
            is_successful_retained_reuse=is_successful_retained,
            is_incorrect_activation=is_incorrect_act,
        )

    @classmethod
    def aggregate_stream_comparisons(
        cls,
        comparisons: list[TaskRoutingComparison],
        unnecessary_allocations: int = 0,
        qualification_failures: int = 0,
    ) -> RoutingErrorSummary:
        """Aggregate per-task comparisons into complete routing error summary."""
        n = len(comparisons)
        if n == 0:
            return RoutingErrorSummary()

        harmful = sum(1 for c in comparisons if c.is_harmful_route)
        missed = sum(1 for c in comparisons if c.is_missed_benefit)
        success = sum(1 for c in comparisons if c.is_successful_retained_reuse)
        incorrect = sum(1 for c in comparisons if c.is_incorrect_activation)
        stat_neg = sum(1 for c in comparisons if c.is_statistical_negative_transfer)
        mean_delta = float(np.mean([c.delta_vs_fallback for c in comparisons]))

        return RoutingErrorSummary(
            total_evaluations=n,
            harmful_route_choices=harmful,
            missed_beneficial_routes=missed,
            successful_retained_reuses=success,
            incorrect_activations=incorrect,
            unnecessary_skill_allocations=unnecessary_allocations,
            candidate_qualification_failures=qualification_failures,
            statistical_negative_transfers=stat_neg,
            mean_delta_vs_fallback=mean_delta,
            episodes=comparisons,
        )
