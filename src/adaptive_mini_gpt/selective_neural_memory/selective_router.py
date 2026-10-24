"""Phase 3: Selective Skill Routing and Negative Transfer Prevention.

Implements SelectiveSkillRouter:
1. Evaluates candidate skills against development evidence only (never sealed test labels).
2. Requires candidate to exceed selection threshold tau_select (0.45) and outperform source skills by delta_margin (+0.05).
3. If no candidate justifies routing, falls back to best active source skill or base model.
4. Precludes routing interference and cross-task degradation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.modular_skill_learning.repertoire import extract_model_hidden
from adaptive_mini_gpt.selective_neural_memory.skill_state import (
    ManagedSkillRecord,
    SelectiveSkillMemoryRegistry,
    SkillState,
)

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Detailed record of a skill routing decision on a task."""
    task_name: str
    selected_skill_id: str
    selected_state: SkillState
    probe_f1: float
    best_source_f1: float
    candidate_probe_f1: float
    is_candidate_routed: bool
    candidate_accepted_for_routing: bool
    reason: str


class SelectiveSkillRouter:
    """Evidence-grounded router preventing negative transfer via selective gating."""

    def __init__(
        self,
        tau_select: float = 0.45,
        delta_margin: float = 0.05,
    ) -> None:
        self.tau_select = tau_select
        self.delta_margin = delta_margin

    @staticmethod
    def evaluate_adapter_on_queries(
        adapter: nn.Module | None,
        model: MiniGPT,
        queries: list[Any],
        device: torch.device,
    ) -> tuple[float, float]:
        """Evaluate macro F1 and accuracy on a small set of development queries."""
        if not queries:
            return 0.3333, 0.3333

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
                tgt_idx = lbl_map.get(ans, 2)
                targets.append(tgt_idx)

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
        acc = float(np.mean(preds_arr == tgts_arr))

        recalls = []
        for c_idx in [0, 1, 2]:
            mask = (tgts_arr == c_idx)
            if np.sum(mask) > 0:
                recalls.append(float(np.sum((preds_arr == c_idx) & mask) / np.sum(mask)))
        macro_f1 = float(np.mean(recalls)) if recalls else 0.3333
        return macro_f1, acc

    def route_task(
        self,
        task_name: str,
        model: MiniGPT,
        registry: SelectiveSkillMemoryRegistry,
        dev_queries: list[Any],
        device: torch.device,
        allow_dormant_probing: bool = False,
    ) -> RoutingDecision:
        """Select best neural route on development queries with negative transfer prevention."""
        # 1. Evaluate baseline source skills (active source skills)
        active_skill_ids = registry.list_active_skills()
        best_source_id = "base_model"
        best_source_f1 = 0.0

        for sk_id in active_skill_ids:
            rec = registry.get_skill(sk_id)
            if not sk_id.startswith("cand_") and not sk_id.startswith("m70_") and not sk_id.startswith("m71_"):
                f1, _ = self.evaluate_adapter_on_queries(rec.adapter, model, dev_queries, device)
                if f1 > best_source_f1:
                    best_source_f1 = f1
                    best_source_id = sk_id

        # 2. Evaluate acquired candidate skills
        candidate_ids = [
            sk_id for sk_id in (registry.list_skills() if allow_dormant_probing else active_skill_ids)
            if sk_id.startswith("cand_") or sk_id.startswith("m70_") or sk_id.startswith("m71_")
        ]

        best_cand_id = ""
        best_cand_f1 = 0.0
        for sk_id in candidate_ids:
            rec = registry.get_skill(sk_id)
            if rec.state == SkillState.REJECTED:
                continue
            f1, _ = self.evaluate_adapter_on_queries(rec.adapter, model, dev_queries, device)
            if f1 > best_cand_f1:
                best_cand_f1 = f1
                best_cand_id = sk_id

        # 3. Decision rule: Route to candidate only if it strictly passes both thresholds
        if best_cand_id and (best_cand_f1 >= self.tau_select) and (best_cand_f1 >= best_source_f1 + self.delta_margin):
            cand_rec = registry.get_skill(best_cand_id)
            cand_rec.reuse_count += 1
            return RoutingDecision(
                task_name=task_name,
                selected_skill_id=best_cand_id,
                selected_state=cand_rec.state,
                probe_f1=best_cand_f1,
                best_source_f1=best_source_f1,
                candidate_probe_f1=best_cand_f1,
                is_candidate_routed=True,
                candidate_accepted_for_routing=True,
                reason=(
                    f"Candidate '{best_cand_id}' qualified on dev evidence "
                    f"(F1={best_cand_f1:.4f} >= {self.tau_select:.2f} and +{best_cand_f1 - best_source_f1:.4f} margin over source)."
                ),
            )

        # 4. Fallback to best source skill to prevent negative transfer
        if best_source_id != "base_model" and best_source_f1 > 0.35:
            src_rec = registry.get_skill(best_source_id)
            src_rec.reuse_count += 1
            return RoutingDecision(
                task_name=task_name,
                selected_skill_id=best_source_id,
                selected_state=src_rec.state,
                probe_f1=best_source_f1,
                best_source_f1=best_source_f1,
                candidate_probe_f1=best_cand_f1,
                is_candidate_routed=False,
                candidate_accepted_for_routing=False,
                reason=(
                    f"Routed to source skill '{best_source_id}' (F1={best_source_f1:.4f}); "
                    f"candidate '{best_cand_id}' (F1={best_cand_f1:.4f}) failed margin threshold."
                ),
            )

        # 5. Fallback to base model
        base_f1, _ = self.evaluate_adapter_on_queries(None, model, dev_queries, device)
        return RoutingDecision(
            task_name=task_name,
            selected_skill_id="base_model",
            selected_state=SkillState.ACTIVE,
            probe_f1=base_f1,
            best_source_f1=best_source_f1,
            candidate_probe_f1=best_cand_f1,
            is_candidate_routed=False,
            candidate_accepted_for_routing=False,
            reason=f"No skill met threshold; defaulted to frozen base model (F1={base_f1:.4f}).",
        )
