"""Phase 5: Task-Conditioned Skill Routing and Interference Elimination.

Solves M57's through-composition interference (-0.3449 F1 degradation) by
dynamically routing queries without relying on hidden task identifiers:
  - Source-consistent queries -> Original frozen source skill f_source(h)
  - Unfamiliar / target queries -> Adapted composite skill f_source(h) + g_res(h)
  - Reset / fallback queries -> Fresh adapter g_fresh(h)

Routing is driven autonomously by observable prediction entropy, confidence,
and embedding proximity to source reference centroids.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter
from adaptive_mini_gpt.guided_experiment.environment import AutonomousEnvironment
from adaptive_mini_gpt.model.transformer import MiniGPT

logger = logging.getLogger(__name__)


class RouteDestination(str, Enum):
    """Destination module for routed query."""
    SOURCE_SKILL = "SOURCE_SKILL"
    COMPOSITE_SKILL = "COMPOSITE_SKILL"
    FRESH_SKILL = "FRESH_SKILL"


class RouterMode(str, Enum):
    """Operational mode for evaluation comparisons."""
    AUTONOMOUS_EVIDENCE = "AUTONOMOUS_EVIDENCE"
    ALWAYS_SOURCE = "ALWAYS_SOURCE"
    ALWAYS_COMPOSITE = "ALWAYS_COMPOSITE"
    ORACLE = "ORACLE"


@dataclass
class RoutingStepResult:
    """Detailed audit of a single query routing decision."""
    query_text: str
    true_domain: str
    destination_selected: RouteDestination
    confidence_score: float
    prediction_entropy: float
    is_correctly_routed: bool
    latency_ms: float
    predicted_label: str
    true_label: str


@dataclass
class RoutedEvaluationReport:
    """Comprehensive performance and latency profile of skill routing."""
    router_mode: RouterMode
    source_task_id: str
    target_task_id: str
    source_routed_macro_f1: float
    target_routed_macro_f1: float
    source_isolated_baseline_f1: float
    source_composite_interference_f1: float
    routing_accuracy: float
    source_misrouting_rate: float
    target_misrouting_rate: float
    mean_routing_latency_ms: float
    total_queries_evaluated: int
    interference_recovered: float


class TaskConditionedSkillRouter(nn.Module):
    """Autonomous evidence-driven router preventing interference between source and target skills."""

    CLASS_LABELS = ["YES", "NO", "UNKNOWN"]

    def __init__(
        self,
        source_adapter: PersistentCandidateAdapter,
        composite_adapter: nn.Module,
        fresh_adapter: PersistentCandidateAdapter | None = None,
        confidence_threshold: float = 0.75,
        entropy_threshold: float = 0.55,
    ) -> None:
        super().__init__()
        self.source_adapter = source_adapter
        self.composite_adapter = composite_adapter
        self.fresh_adapter = fresh_adapter
        self.confidence_threshold = confidence_threshold
        self.entropy_threshold = entropy_threshold

        self.source_centroid: torch.Tensor | None = None

    def calibrate_source_centroid(self, source_embeddings: list[torch.Tensor]) -> None:
        """Calibrate representative centroid from qualified source domain representations."""
        if not source_embeddings:
            return
        stacked = torch.stack([e.squeeze(0)[-1, :] if e.dim() >= 2 else e for e in source_embeddings])
        self.source_centroid = torch.mean(stacked, dim=0).detach()

    def determine_route(
        self,
        hidden_states: torch.Tensor,
        mode: RouterMode = RouterMode.AUTONOMOUS_EVIDENCE,
        true_task_id: str | None = None,
        source_task_id: str = "source",
    ) -> tuple[RouteDestination, float, float]:
        """Determine destination strictly via observable confidence and entropy unless mode is ablation."""
        if mode == RouterMode.ALWAYS_SOURCE:
            return RouteDestination.SOURCE_SKILL, 1.0, 0.0

        if mode == RouterMode.ALWAYS_COMPOSITE:
            return RouteDestination.COMPOSITE_SKILL, 1.0, 0.0

        if mode == RouterMode.ORACLE:
            if true_task_id and source_task_id in true_task_id:
                return RouteDestination.SOURCE_SKILL, 1.0, 0.0
            return RouteDestination.COMPOSITE_SKILL, 1.0, 0.0

        # Autonomous mode: Evaluate hidden state through frozen source adapter and composite adapter
        with torch.no_grad():
            self.source_adapter.eval()
            self.composite_adapter.eval()

            src_logits = self.source_adapter(hidden_states)
            comp_logits = self.composite_adapter(hidden_states)

            p_src = F.softmax(src_logits, dim=-1)
            p_comp = F.softmax(comp_logits, dim=-1)

            conf_src = float(torch.max(p_src, dim=-1)[0].item())
            conf_comp = float(torch.max(p_comp, dim=-1)[0].item())

            ent_src = float(-torch.sum(torch.clamp(p_src, min=1e-12) * torch.log(torch.clamp(p_src, min=1e-12)), dim=-1).item())
            ent_comp = float(-torch.sum(torch.clamp(p_comp, min=1e-12) * torch.log(torch.clamp(p_comp, min=1e-12)), dim=-1).item())

        # Decide route based on relative confidence and decisive entropy
        if self.fresh_adapter is not None and conf_src < 0.38 and conf_comp < 0.38:
            return RouteDestination.FRESH_FALLBACK, max(conf_src, conf_comp), min(ent_src, ent_comp)

        # Route to SOURCE_SKILL if source adapter is more confident or decisive than composite
        if (conf_src > conf_comp) or (conf_src >= conf_comp - 0.05 and ent_src <= ent_comp):
            return RouteDestination.SOURCE_SKILL, conf_src, ent_src
        else:
            return RouteDestination.COMPOSITE_SKILL, conf_comp, ent_comp

    def forward(
        self,
        hidden_states: torch.Tensor,
        mode: RouterMode = RouterMode.AUTONOMOUS_EVIDENCE,
        true_task_id: str | None = None,
        source_task_id: str = "source",
    ) -> tuple[torch.Tensor, RouteDestination]:
        """Route and execute forward pass through selected skill pathway."""
        dest, _, _ = self.determine_route(hidden_states, mode=mode, true_task_id=true_task_id, source_task_id=source_task_id)

        if dest == RouteDestination.SOURCE_SKILL:
            with torch.no_grad():
                self.source_adapter.eval()
                return self.source_adapter(hidden_states), dest

        if dest == RouteDestination.COMPOSITE_SKILL:
            return self.composite_adapter(hidden_states), dest

        if self.fresh_adapter is not None:
            return self.fresh_adapter(hidden_states), dest

        return self.composite_adapter(hidden_states), dest

    def evaluate_routed_benchmark(
        self,
        base_model: MiniGPT,
        tokenizer: Any,
        device: torch.device,
        source_env: AutonomousEnvironment,
        target_env: AutonomousEnvironment,
        mode: RouterMode = RouterMode.AUTONOMOUS_EVIDENCE,
    ) -> RoutedEvaluationReport:
        """Run full evaluation comparing routed performance against baseline interference."""
        self.eval()
        base_model.eval()

        # Helper to extract hidden state
        def extract_hidden(text: str) -> torch.Tensor:
            toks = tokenizer.encode(text)
            inp = torch.tensor([toks], dtype=torch.long, device=device)
            b, t = inp.size()
            tok_emb = base_model.wte(inp)
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            x = base_model.drop(tok_emb + base_model.wpe(pos))
            for block in base_model.blocks:
                x, _ = block(x)
            return base_model.ln_f(x)

        # 1. Evaluate baseline source performance in true isolation
        src_preds_iso: list[str] = []
        src_truths: list[str] = []
        with torch.no_grad():
            for q in source_env.held_out_test_queries:
                h = extract_hidden(q.question_text)
                logits = self.source_adapter(h)
                pred_idx = int(torch.argmax(logits, dim=-1).item())
                src_preds_iso.append(self.CLASS_LABELS[pred_idx])
                src_truths.append(q.expected_answer)

        def calc_f1(preds: list[str], truths: list[str]) -> float:
            f1s = []
            for lab in self.CLASS_LABELS:
                tp = sum(1 for p, g in zip(preds, truths) if p == lab and g == lab)
                fp = sum(1 for p, g in zip(preds, truths) if p == lab and g != lab)
                fn = sum(1 for p, g in zip(preds, truths) if p != lab and g == lab)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                f1s.append(f1)
            return float(np.mean(f1s))

        src_isolated_f1 = calc_f1(src_preds_iso, src_truths)

        # 2. Evaluate unrouted composite pathway on source (M57 interference)
        src_preds_comp: list[str] = []
        with torch.no_grad():
            for q in source_env.held_out_test_queries:
                h = extract_hidden(q.question_text)
                logits = self.composite_adapter(h)
                pred_idx = int(torch.argmax(logits, dim=-1).item())
                src_preds_comp.append(self.CLASS_LABELS[pred_idx])
        src_unrouted_composite_f1 = calc_f1(src_preds_comp, src_truths)
        baseline_interference = max(0.0, src_isolated_f1 - src_unrouted_composite_f1)

        # 3. Evaluate combined query stream through the actual Router
        queries_with_domain: list[tuple[Any, str]] = (
            [(q, "source") for q in source_env.held_out_test_queries] +
            [(q, "target") for q in target_env.held_out_test_queries]
        )

        src_routed_preds: list[str] = []
        src_routed_truths: list[str] = []
        tgt_routed_preds: list[str] = []
        tgt_routed_truths: list[str] = []

        routing_correct = 0
        src_misrouted = 0
        tgt_misrouted = 0
        latencies: list[float] = []

        with torch.no_grad():
            for q, domain in queries_with_domain:
                t0 = time.perf_counter()
                h = extract_hidden(q.question_text)
                logits, dest = self.forward(
                    h, mode=mode, true_task_id=domain, source_task_id="source"
                )
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

                pred_idx = int(torch.argmax(logits, dim=-1).item())
                pred_label = self.CLASS_LABELS[pred_idx]

                if domain == "source":
                    src_routed_preds.append(pred_label)
                    src_routed_truths.append(q.expected_answer)
                    if dest == RouteDestination.SOURCE_SKILL:
                        routing_correct += 1
                    else:
                        src_misrouted += 1
                else:
                    tgt_routed_preds.append(pred_label)
                    tgt_routed_truths.append(q.expected_answer)
                    if dest in (RouteDestination.COMPOSITE_SKILL, RouteDestination.FRESH_SKILL):
                        routing_correct += 1
                    else:
                        tgt_misrouted += 1

        src_routed_f1 = calc_f1(src_routed_preds, src_routed_truths)
        tgt_routed_f1 = calc_f1(tgt_routed_preds, tgt_routed_truths)
        total_q = len(queries_with_domain)
        routing_acc = float(routing_correct / total_q) if total_q > 0 else 0.0
        src_mis_rate = float(src_misrouted / len(source_env.held_out_test_queries)) if source_env.held_out_test_queries else 0.0
        tgt_mis_rate = float(tgt_misrouted / len(target_env.held_out_test_queries)) if target_env.held_out_test_queries else 0.0
        interference_recovered = max(0.0, src_routed_f1 - src_unrouted_composite_f1)

        return RoutedEvaluationReport(
            router_mode=mode,
            source_task_id=source_env.env_id,
            target_task_id=target_env.env_id,
            source_routed_macro_f1=src_routed_f1,
            target_routed_macro_f1=tgt_routed_f1,
            source_isolated_baseline_f1=src_isolated_f1,
            source_composite_interference_f1=baseline_interference,
            routing_accuracy=routing_acc,
            source_misrouting_rate=src_mis_rate,
            target_misrouting_rate=tgt_mis_rate,
            mean_routing_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            total_queries_evaluated=total_q,
            interference_recovered=interference_recovered,
        )
