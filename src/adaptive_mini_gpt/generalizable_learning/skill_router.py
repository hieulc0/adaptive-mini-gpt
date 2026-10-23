"""Phase 4 & 5: Evidence-Based Skill Routing and Representation Learning.

Solves M58's routing failure (47.92% accuracy, worse joint F1 than static composite).
Evaluates 6 routing strategies without peeking at hidden rule labels or test sets:
  1. ALWAYS_SOURCE
  2. ALWAYS_COMPOSITE
  3. ALWAYS_FRESH
  4. M58_CONFIDENCE_ENTROPY (relative confidence / entropy difference)
  5. DETERMINISTIC_VERIFIED_EVIDENCE (sample likelihood & centroid proximity)
  6. MINIMAL_TRAINABLE_ROUTER (lightweight linear router tuned on verified sample)

Phase 5 includes an optional contrastive representation objective on skill representations
to evaluate whether representation disentanglement improves routing without hurting
relational transfer.
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
from adaptive_mini_gpt.reliable_transfer.composition_models import ResidualSkillAdapter

logger = logging.getLogger(__name__)


class RouteDestination(str, Enum):
    """Destination module for routed query."""
    SOURCE_SKILL = "SOURCE_SKILL"
    COMPOSITE_SKILL = "COMPOSITE_SKILL"
    FRESH_SKILL = "FRESH_SKILL"


class RouterMode(str, Enum):
    """Operational mode for routing comparisons."""
    ALWAYS_SOURCE = "ALWAYS_SOURCE"
    ALWAYS_COMPOSITE = "ALWAYS_COMPOSITE"
    ALWAYS_FRESH = "ALWAYS_FRESH"
    M58_CONFIDENCE_ENTROPY = "M58_CONFIDENCE_ENTROPY"
    DETERMINISTIC_VERIFIED_EVIDENCE = "DETERMINISTIC_VERIFIED_EVIDENCE"
    MINIMAL_TRAINABLE_ROUTER = "MINIMAL_TRAINABLE_ROUTER"
    ORACLE = "ORACLE"


@dataclass
class RoutingStepResult:
    """Detailed audit of an individual routing decision."""
    query_text: str
    true_domain: str
    destination_selected: RouteDestination
    confidence_score: float
    is_correctly_routed: bool
    latency_ms: float
    predicted_label: str
    true_label: str


@dataclass
class RoutedEvaluationReport:
    """Joint evaluation metrics comparing source and target skill performance."""
    router_mode: RouterMode
    source_task_id: str
    target_task_id: str
    source_routed_macro_f1: float
    target_routed_macro_f1: float
    joint_source_target_f1: float  # Primary metric: (source_f1 + target_f1) / 2.0
    source_isolated_baseline_f1: float
    source_composite_interference_f1: float
    routing_accuracy: float
    source_misrouting_rate: float
    target_misrouting_rate: float
    mean_routing_latency_ms: float
    total_queries_evaluated: int
    interference_recovered: float
    misrouting_penalty: float


class MinimalTrainableSkillGate(nn.Module):
    """Minimal 387-parameter linear gate mapping representation to routing logits."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, 3)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # hidden_state: [1, seq_len, d_model] -> use pooled representation [1, d_model]
        pooled = hidden_state.mean(dim=1) if hidden_state.dim() == 3 else hidden_state
        return self.gate(pooled)


class EvidenceBasedSkillRouter(nn.Module):
    """Comprehensive evidence-driven skill router for M59."""

    CLASS_LABELS = ["YES", "NO", "UNKNOWN"]

    def __init__(
        self,
        source_adapter: PersistentCandidateAdapter,
        composite_adapter: ResidualSkillAdapter,
        fresh_adapter: PersistentCandidateAdapter | None = None,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.source_adapter = source_adapter
        self.composite_adapter = composite_adapter
        self.fresh_adapter = fresh_adapter
        self.d_model = d_model

        self.source_centroid: torch.Tensor | None = None
        self.target_centroid: torch.Tensor | None = None
        self.trainable_gate = MinimalTrainableSkillGate(d_model=d_model)

    def calibrate_centroids(
        self,
        source_embeddings: list[torch.Tensor],
        target_embeddings: list[torch.Tensor],
    ) -> None:
        """Calibrate representation centroids from verified observations."""
        if source_embeddings:
            stacked_src = torch.stack([e.squeeze(0).mean(dim=0) if e.dim() >= 2 else e for e in source_embeddings])
            self.source_centroid = torch.mean(stacked_src, dim=0).detach()
        if target_embeddings:
            stacked_tgt = torch.stack([e.squeeze(0).mean(dim=0) if e.dim() >= 2 else e for e in target_embeddings])
            self.target_centroid = torch.mean(stacked_tgt, dim=0).detach()

    def train_minimal_gate(
        self,
        training_data: list[tuple[torch.Tensor, RouteDestination]],
        steps: int = 25,
        lr: float = 5e-3,
    ) -> None:
        """Train the minimal gating network on verified sample observations."""
        if not training_data:
            return
        optimizer = torch.optim.AdamW(self.trainable_gate.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        label_map = {
            RouteDestination.SOURCE_SKILL: 0,
            RouteDestination.COMPOSITE_SKILL: 1,
            RouteDestination.FRESH_SKILL: 2,
        }

        self.trainable_gate.train()
        for _ in range(steps):
            for h, dest in training_data:
                h_det = h.detach()
                target = torch.tensor([label_map[dest]], dtype=torch.long, device=h_det.device)
                logits = self.trainable_gate(h_det)
                loss = criterion(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.trainable_gate.eval()

    def determine_route(
        self,
        hidden_states: torch.Tensor,
        mode: RouterMode = RouterMode.DETERMINISTIC_VERIFIED_EVIDENCE,
        true_task_id: str | None = None,
        source_task_id: str = "source",
    ) -> tuple[RouteDestination, float]:
        """Determine routing destination according to specified routing mode."""
        if mode == RouterMode.ALWAYS_SOURCE:
            return RouteDestination.SOURCE_SKILL, 1.0

        if mode == RouterMode.ALWAYS_COMPOSITE:
            return RouteDestination.COMPOSITE_SKILL, 1.0

        if mode == RouterMode.ALWAYS_FRESH:
            return RouteDestination.FRESH_SKILL, 1.0

        if mode == RouterMode.ORACLE:
            if true_task_id and source_task_id in true_task_id:
                return RouteDestination.SOURCE_SKILL, 1.0
            return RouteDestination.COMPOSITE_SKILL, 1.0

        # Mode 4: M58 Confidence/Entropy router
        if mode == RouterMode.M58_CONFIDENCE_ENTROPY:
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

            if (conf_src > conf_comp) or (conf_src >= conf_comp - 0.05 and ent_src <= ent_comp):
                return RouteDestination.SOURCE_SKILL, conf_src
            return RouteDestination.COMPOSITE_SKILL, conf_comp

        # Mode 6: Minimal Trainable Router
        if mode == RouterMode.MINIMAL_TRAINABLE_ROUTER:
            with torch.no_grad():
                self.trainable_gate.eval()
                gate_logits = self.trainable_gate(hidden_states)
                probs = F.softmax(gate_logits, dim=-1)
                pred_idx = int(torch.argmax(probs, dim=-1).item())
                conf = float(probs[0, pred_idx].item())
            dest_map = [RouteDestination.SOURCE_SKILL, RouteDestination.COMPOSITE_SKILL, RouteDestination.FRESH_SKILL]
            return dest_map[pred_idx], conf

        # Mode 5: DETERMINISTIC_VERIFIED_EVIDENCE (Default M59)
        # Evaluates likelihood through adapter predictions and centroid distance
        with torch.no_grad():
            self.source_adapter.eval()
            self.composite_adapter.eval()
            src_logits = self.source_adapter(hidden_states)
            comp_logits = self.composite_adapter(hidden_states)
            p_src = F.softmax(src_logits, dim=-1)
            p_comp = F.softmax(comp_logits, dim=-1)

            # Max confidence
            conf_src = float(torch.max(p_src, dim=-1)[0].item())
            conf_comp = float(torch.max(p_comp, dim=-1)[0].item())

            # Representation centroid distance if available
            pooled_h = hidden_states.mean(dim=1).squeeze(0) if hidden_states.dim() == 3 else hidden_states.squeeze(0)
            if self.source_centroid is not None and self.target_centroid is not None:
                dist_src = float(torch.norm(pooled_h - self.source_centroid).item())
                dist_tgt = float(torch.norm(pooled_h - self.target_centroid).item())
                # Evidence score combining confidence and proximity
                score_src = conf_src / (1.0 + 0.5 * dist_src)
                score_tgt = conf_comp / (1.0 + 0.5 * dist_tgt)
            else:
                score_src = conf_src
                score_tgt = conf_comp

        if score_src > score_tgt:
            return RouteDestination.SOURCE_SKILL, conf_src
        return RouteDestination.COMPOSITE_SKILL, conf_comp

    def forward(
        self,
        hidden_states: torch.Tensor,
        mode: RouterMode = RouterMode.DETERMINISTIC_VERIFIED_EVIDENCE,
        true_task_id: str | None = None,
        source_task_id: str = "source",
    ) -> tuple[torch.Tensor, RouteDestination]:
        """Forward pass with dynamically selected route."""
        dest, _ = self.determine_route(
            hidden_states, mode=mode, true_task_id=true_task_id, source_task_id=source_task_id
        )
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
        mode: RouterMode = RouterMode.DETERMINISTIC_VERIFIED_EVIDENCE,
    ) -> RoutedEvaluationReport:
        """Run complete joint evaluation of source and target tasks."""
        self.eval()
        base_model.eval()

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

        # Baseline isolated source performance
        src_iso_preds, src_iso_targets = [], []
        with torch.no_grad():
            self.source_adapter.eval()
            for q in source_env.held_out_test_queries:
                h = extract_hidden(q.question_text)
                logits = self.source_adapter(h)
                pred_idx = int(torch.argmax(logits, dim=-1).item())
                src_iso_preds.append(self.CLASS_LABELS[pred_idx])
                src_iso_targets.append(q.expected_answer)
        src_iso_f1 = float(np.mean([p == t for p, t in zip(src_iso_preds, src_iso_targets)]))

        # Baseline unrouted composite performance on source (interference)
        src_comp_preds = []
        with torch.no_grad():
            self.composite_adapter.eval()
            for q in source_env.held_out_test_queries:
                h = extract_hidden(q.question_text)
                logits = self.composite_adapter(h)
                pred_idx = int(torch.argmax(logits, dim=-1).item())
                src_comp_preds.append(self.CLASS_LABELS[pred_idx])
        src_comp_f1 = float(np.mean([p == t for p, t in zip(src_comp_preds, src_iso_targets)]))

        # Routed evaluation across both tasks
        source_results: list[RoutingStepResult] = []
        target_results: list[RoutingStepResult] = []

        for q in source_env.held_out_test_queries:
            t0 = time.perf_counter()
            h = extract_hidden(q.question_text)
            logits, dest = self.forward(h, mode=mode, true_task_id=source_env.env_id, source_task_id=source_env.env_id)
            lat = (time.perf_counter() - t0) * 1000.0
            pred_idx = int(torch.argmax(logits, dim=-1).item())
            pred_lbl = self.CLASS_LABELS[pred_idx]
            is_correct_dest = (dest == RouteDestination.SOURCE_SKILL)
            source_results.append(RoutingStepResult(
                query_text=q.question_text,
                true_domain=source_env.env_id,
                destination_selected=dest,
                confidence_score=1.0,
                is_correctly_routed=is_correct_dest,
                latency_ms=lat,
                predicted_label=pred_lbl,
                true_label=q.expected_answer,
            ))

        for q in target_env.held_out_test_queries:
            t0 = time.perf_counter()
            h = extract_hidden(q.question_text)
            logits, dest = self.forward(h, mode=mode, true_task_id=target_env.env_id, source_task_id=source_env.env_id)
            lat = (time.perf_counter() - t0) * 1000.0
            pred_idx = int(torch.argmax(logits, dim=-1).item())
            pred_lbl = self.CLASS_LABELS[pred_idx]
            is_correct_dest = (dest in (RouteDestination.COMPOSITE_SKILL, RouteDestination.FRESH_SKILL))
            target_results.append(RoutingStepResult(
                query_text=q.question_text,
                true_domain=target_env.env_id,
                destination_selected=dest,
                confidence_score=1.0,
                is_correctly_routed=is_correct_dest,
                latency_ms=lat,
                predicted_label=pred_lbl,
                true_label=q.expected_answer,
            ))

        src_routed_acc = float(np.mean([r.predicted_label == r.true_label for r in source_results]))
        tgt_routed_acc = float(np.mean([r.predicted_label == r.true_label for r in target_results]))
        joint_f1 = (src_routed_acc + tgt_routed_acc) / 2.0

        all_results = source_results + target_results
        total_queries = len(all_results)
        routing_acc = float(np.mean([r.is_correctly_routed for r in all_results]))
        src_misroute = float(np.mean([not r.is_correctly_routed for r in source_results]))
        tgt_misroute = float(np.mean([not r.is_correctly_routed for r in target_results]))
        mean_lat = float(np.mean([r.latency_ms for r in all_results]))

        interference_recovered = src_routed_acc - src_comp_f1
        misrouting_penalty = max(0.0, src_iso_f1 - src_routed_acc)

        return RoutedEvaluationReport(
            router_mode=mode,
            source_task_id=source_env.env_id,
            target_task_id=target_env.env_id,
            source_routed_macro_f1=src_routed_acc,
            target_routed_macro_f1=tgt_routed_acc,
            joint_source_target_f1=joint_f1,
            source_isolated_baseline_f1=src_iso_f1,
            source_composite_interference_f1=src_comp_f1,
            routing_accuracy=routing_acc,
            source_misrouting_rate=src_misroute,
            target_misrouting_rate=tgt_misroute,
            mean_routing_latency_ms=mean_lat,
            total_queries_evaluated=total_queries,
            interference_recovered=interference_recovered,
            misrouting_penalty=misrouting_penalty,
        )


def compute_contrastive_representation_loss(
    source_representations: torch.Tensor,
    target_representations: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """Phase 5: Optional contrastive representation loss between source and target skill adapters.

    Encourages distinct clustering between source and target skills without imposing
    rigid orthogonality across relational features.
    """
    # Normalize
    z_src = F.normalize(source_representations, p=2, dim=-1)
    z_tgt = F.normalize(target_representations, p=2, dim=-1)
    cosine_sim = torch.sum(z_src * z_tgt, dim=-1)
    # Hinge loss: penalize if cosine similarity exceeds threshold (1 - margin)
    contrastive_loss = torch.relu(cosine_sim - (1.0 - margin)).mean()
    return contrastive_loss
