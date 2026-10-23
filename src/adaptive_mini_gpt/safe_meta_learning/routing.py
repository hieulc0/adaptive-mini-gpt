"""Phase 5: Counterfactual Skill-Routing Utility Engine for M61.

Solves the routing discrepancy identified in M60 where nominal route-label accuracy
diverged from downstream task performance:
1. Evaluates all candidate pathways independently on every query:
   - SOURCE_SKILL
   - COMPOSITE_SKILL
   - FRESH_SKILL
2. Determines empirical ground-truth utility:
   - Which route ACTUALLY yields the correct prediction on the given query.
3. Utility-Based Expected Router:
   - Uses observed development validation performance to estimate which pathway
     has the highest empirical expected reward on the query category.
4. Offline Empirical Best-Route Reference (Upper Bound):
   - Chooses the pathway that actually produces the correct answer (or source if both correct).
5. Compares:
   - ALWAYS_SOURCE
   - ALWAYS_COMPOSITE
   - ALWAYS_FRESH
   - M58_CONFIDENCE_ENTROPY
   - DETERMINISTIC_VERIFIED_EVIDENCE
   - MINIMAL_TRAINABLE_ROUTER
   - UTILITY_EXPECTED_ROUTER (New M61 model)
   - EMPIRICAL_BEST_ROUTE_BOUND
"""
from __future__ import annotations

import copy
import enum
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.causal_meta_learning.functional_routing import (
    FunctionalQueryTrace,
    MinimalNeuralGate,
)
from adaptive_mini_gpt.generalizable_learning.skill_router import (
    RouteDestination,
    RouterMode,
)
from adaptive_mini_gpt.guided_experiment.environment import AutonomousEnvironment
from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.reliable_transfer.composition_models import ResidualSkillAdapter

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualQueryOutcome:
    """Detailed trace of an individual query evaluated counterfactually across all candidate routes."""
    query_text: str
    true_domain: str
    expected_answer: str
    source_pred: str
    composite_pred: str
    fresh_pred: str
    source_correct: bool
    composite_correct: bool
    fresh_correct: bool
    best_empirical_route: RouteDestination
    selected_route: RouteDestination
    selected_pred: str
    selected_correct: bool
    is_route_dependent: bool
    routing_latency_ms: float


@dataclass
class CounterfactualRoutingReport:
    """Quantitative routing performance report grounded in downstream task utility."""
    policy_name: str
    source_task_id: str
    target_task_id: str
    total_queries: int
    route_dependent_queries: int
    route_dependence_rate: float
    source_routed_macro_f1: float
    target_routed_macro_f1: float
    joint_source_target_f1: float
    always_source_joint_f1: float
    always_composite_joint_f1: float
    always_fresh_joint_f1: float
    empirical_best_joint_f1: float
    behavioral_gain_over_best_static: float
    nominal_routing_accuracy: float
    utility_routing_accuracy: float  # Fraction of queries where selected route matches empirical best
    mean_latency_ms: float
    superior_to_best_static_route: bool


class CounterfactualSkillRouter(nn.Module):
    """Evaluates and executes counterfactual routing based on empirical downstream task utility."""

    CLASS_LABELS = ["NO", "YES"]

    def __init__(
        self,
        d_model: int,
        source_adapter: nn.Module,
        composite_adapter: nn.Module,
        fresh_adapter: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.source_adapter = source_adapter
        self.composite_adapter = composite_adapter
        self.fresh_adapter = fresh_adapter
        self.trainable_gate = MinimalNeuralGate(d_model=d_model, num_destinations=3)
        self.source_centroid: torch.Tensor | None = None
        self.target_centroid: torch.Tensor | None = None

        # Empirical prior utility table estimated on development validation sets
        # [source_domain_utility, target_domain_utility] per route
        self.route_empirical_utilities: dict[RouteDestination, dict[str, float]] = {
            RouteDestination.SOURCE_SKILL: {"source": 0.85, "target": 0.35},
            RouteDestination.COMPOSITE_SKILL: {"source": 0.40, "target": 0.75},
            RouteDestination.FRESH_SKILL: {"source": 0.50, "target": 0.65},
        }

    def calibrate_evidence_centroids(
        self,
        source_support_hidden: list[torch.Tensor],
        target_support_hidden: list[torch.Tensor],
    ) -> None:
        """Calibrate representation centroids from support examples."""
        with torch.no_grad():
            if source_support_hidden:
                stacked_src = torch.stack([
                    h.mean(dim=1).squeeze(0) if h.dim() == 3 else h.squeeze(0)
                    for h in source_support_hidden
                ])
                self.source_centroid = stacked_src.mean(dim=0).detach()

            if target_support_hidden:
                stacked_tgt = torch.stack([
                    h.mean(dim=1).squeeze(0) if h.dim() == 3 else h.squeeze(0)
                    for h in target_support_hidden
                ])
                self.target_centroid = stacked_tgt.mean(dim=0).detach()

    def update_empirical_utilities(
        self,
        source_val_accs: dict[RouteDestination, float],
        target_val_accs: dict[RouteDestination, float],
    ) -> None:
        """Update empirical route utilities based on observed development validation performance."""
        for dest in (RouteDestination.SOURCE_SKILL, RouteDestination.COMPOSITE_SKILL, RouteDestination.FRESH_SKILL):
            if dest in source_val_accs:
                self.route_empirical_utilities[dest]["source"] = source_val_accs[dest]
            if dest in target_val_accs:
                self.route_empirical_utilities[dest]["target"] = target_val_accs[dest]

    def select_route(
        self,
        hidden_states: torch.Tensor,
        policy: str,
        true_domain: str | None = None,
        source_env_id: str = "source",
    ) -> tuple[RouteDestination, float]:
        """Select execution route under the assigned routing policy."""
        if policy == "ALWAYS_SOURCE":
            return RouteDestination.SOURCE_SKILL, 1.0

        if policy == "ALWAYS_COMPOSITE":
            return RouteDestination.COMPOSITE_SKILL, 1.0

        if policy == "ALWAYS_FRESH":
            return RouteDestination.FRESH_SKILL, 1.0

        if policy == "EMPIRICAL_BEST_ROUTE_BOUND":
            # Handled counterfactually at evaluation time
            return RouteDestination.SOURCE_SKILL, 1.0

        if policy == "MINIMAL_TRAINABLE_ROUTER":
            with torch.no_grad():
                self.trainable_gate.eval()
                gate_logits = self.trainable_gate(hidden_states)
                probs = F.softmax(gate_logits, dim=-1)
                pred_idx = int(torch.argmax(probs, dim=-1).item())
                conf = float(probs[0, pred_idx].item())
            dest_map = [RouteDestination.SOURCE_SKILL, RouteDestination.COMPOSITE_SKILL, RouteDestination.FRESH_SKILL]
            return dest_map[pred_idx], conf

        # DETERMINISTIC_VERIFIED_EVIDENCE
        with torch.no_grad():
            self.source_adapter.eval()
            self.composite_adapter.eval()
            src_logits = self.source_adapter(hidden_states)
            comp_logits = self.composite_adapter(hidden_states)
            p_src = F.softmax(src_logits, dim=-1)
            p_comp = F.softmax(comp_logits, dim=-1)
            conf_src = float(torch.max(p_src, dim=-1)[0].item())
            conf_comp = float(torch.max(p_comp, dim=-1)[0].item())

            pooled_h = hidden_states.mean(dim=1).squeeze(0) if hidden_states.dim() == 3 else hidden_states.squeeze(0)
            if self.source_centroid is not None and self.target_centroid is not None:
                dist_src = float(torch.norm(pooled_h - self.source_centroid.to(pooled_h.device)).item())
                dist_tgt = float(torch.norm(pooled_h - self.target_centroid.to(pooled_h.device)).item())
                score_src = conf_src / (1.0 + 0.5 * dist_src)
                score_tgt = conf_comp / (1.0 + 0.5 * dist_tgt)
            else:
                score_src = conf_src
                score_tgt = conf_comp

        if policy == "UTILITY_EXPECTED_ROUTER":
            # Weight confidence scores by observed empirical utility of each skill
            # P(domain=src) vs P(domain=tgt)
            p_domain_src = score_src / (score_src + score_tgt + 1e-12)
            p_domain_tgt = 1.0 - p_domain_src

            u_src = p_domain_src * self.route_empirical_utilities[RouteDestination.SOURCE_SKILL]["source"] + \
                    p_domain_tgt * self.route_empirical_utilities[RouteDestination.SOURCE_SKILL]["target"]
            u_comp = p_domain_src * self.route_empirical_utilities[RouteDestination.COMPOSITE_SKILL]["source"] + \
                     p_domain_tgt * self.route_empirical_utilities[RouteDestination.COMPOSITE_SKILL]["target"]

            if u_src >= u_comp:
                return RouteDestination.SOURCE_SKILL, float(u_src)
            return RouteDestination.COMPOSITE_SKILL, float(u_comp)

        # Default DETERMINISTIC_VERIFIED_EVIDENCE
        if score_src > score_tgt:
            return RouteDestination.SOURCE_SKILL, conf_src
        return RouteDestination.COMPOSITE_SKILL, conf_comp

    def evaluate_counterfactual_benchmark(
        self,
        base_model: MiniGPT,
        tokenizer: Any,
        device: torch.device,
        source_env: AutonomousEnvironment,
        target_env: AutonomousEnvironment,
        policy: str,
    ) -> CounterfactualRoutingReport:
        """Run counterfactual benchmark testing all execution routes on every query."""
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

        all_queries = [
            (q, source_env.env_id) for q in source_env.held_out_test_queries
        ] + [
            (q, target_env.env_id) for q in target_env.held_out_test_queries
        ]

        query_outcomes: list[CounterfactualQueryOutcome] = []
        forced_src_correct = 0
        forced_comp_correct = 0
        forced_fresh_correct = 0
        empirical_best_correct = 0

        for q, true_domain in all_queries:
            h = extract_hidden(q.question_text)
            with torch.no_grad():
                self.source_adapter.eval()
                self.composite_adapter.eval()
                src_l = self.source_adapter(h)
                comp_l = self.composite_adapter(h)

                if self.fresh_adapter is not None:
                    self.fresh_adapter.eval()
                    fresh_l = self.fresh_adapter(h)
                    fresh_pred = self.CLASS_LABELS[int(torch.argmax(fresh_l, dim=-1).item())]
                else:
                    fresh_pred = "NO"

                src_pred = self.CLASS_LABELS[int(torch.argmax(src_l, dim=-1).item())]
                comp_pred = self.CLASS_LABELS[int(torch.argmax(comp_l, dim=-1).item())]

                src_corr = (src_pred == q.expected_answer)
                comp_corr = (comp_pred == q.expected_answer)
                fresh_corr = (fresh_pred == q.expected_answer)

                if src_corr:
                    forced_src_correct += 1
                if comp_corr:
                    forced_comp_correct += 1
                if fresh_corr:
                    forced_fresh_correct += 1

                # Determine offline empirical best route
                if comp_corr and not src_corr:
                    emp_best_route = RouteDestination.COMPOSITE_SKILL
                    empirical_best_correct += 1
                elif src_corr:
                    emp_best_route = RouteDestination.SOURCE_SKILL
                    empirical_best_correct += 1
                elif fresh_corr:
                    emp_best_route = RouteDestination.FRESH_SKILL
                    empirical_best_correct += 1
                else:
                    emp_best_route = RouteDestination.SOURCE_SKILL

            # Execute Selected Policy
            t0 = time.perf_counter()
            if policy == "EMPIRICAL_BEST_ROUTE_BOUND":
                selected_dest = emp_best_route
            else:
                selected_dest, _ = self.select_route(
                    h, policy=policy, true_domain=true_domain, source_env_id=source_env.env_id
                )
            lat_ms = (time.perf_counter() - t0) * 1000.0

            if selected_dest == RouteDestination.SOURCE_SKILL:
                chosen_pred = src_pred
            elif selected_dest == RouteDestination.COMPOSITE_SKILL:
                chosen_pred = comp_pred
            else:
                chosen_pred = fresh_pred

            chosen_corr = (chosen_pred == q.expected_answer)
            is_dependent = (src_corr != comp_corr)

            query_outcomes.append(CounterfactualQueryOutcome(
                query_text=q.question_text,
                true_domain=true_domain,
                expected_answer=q.expected_answer,
                source_pred=src_pred,
                composite_pred=comp_pred,
                fresh_pred=fresh_pred,
                source_correct=src_corr,
                composite_correct=comp_corr,
                fresh_correct=fresh_corr,
                best_empirical_route=emp_best_route,
                selected_route=selected_dest,
                selected_pred=chosen_pred,
                selected_correct=chosen_corr,
                is_route_dependent=is_dependent,
                routing_latency_ms=lat_ms,
            ))

        total_q = len(query_outcomes)
        dependent_q = sum(1 for o in query_outcomes if o.is_route_dependent)

        src_outcomes = [o for o in query_outcomes if o.true_domain == source_env.env_id]
        tgt_outcomes = [o for o in query_outcomes if o.true_domain == target_env.env_id]

        src_f1 = float(np.mean([o.selected_correct for o in src_outcomes])) if src_outcomes else 0.0
        tgt_f1 = float(np.mean([o.selected_correct for o in tgt_outcomes])) if tgt_outcomes else 0.0
        joint_f1 = (src_f1 + tgt_f1) / 2.0

        forced_src_joint = float(forced_src_correct / total_q)
        forced_comp_joint = float(forced_comp_correct / total_q)
        forced_fresh_joint = float(forced_fresh_correct / total_q)
        emp_best_joint = float(empirical_best_correct / total_q)

        best_static_f1 = max(forced_src_joint, forced_comp_joint, forced_fresh_joint)
        gain_over_static = joint_f1 - best_static_f1

        nominal_acc = float(np.mean([
            (o.selected_route == RouteDestination.SOURCE_SKILL if o.true_domain == source_env.env_id
             else o.selected_route in (RouteDestination.COMPOSITE_SKILL, RouteDestination.FRESH_SKILL))
            for o in query_outcomes
        ]))

        utility_acc = float(np.mean([
            (o.selected_route == o.best_empirical_route) for o in query_outcomes
        ]))

        return CounterfactualRoutingReport(
            policy_name=policy,
            source_task_id=source_env.env_id,
            target_task_id=target_env.env_id,
            total_queries=total_q,
            route_dependent_queries=dependent_q,
            route_dependence_rate=float(dependent_q / total_q) if total_q > 0 else 0.0,
            source_routed_macro_f1=src_f1,
            target_routed_macro_f1=tgt_f1,
            joint_source_target_f1=joint_f1,
            always_source_joint_f1=forced_src_joint,
            always_composite_joint_f1=forced_comp_joint,
            always_fresh_joint_f1=forced_fresh_joint,
            empirical_best_joint_f1=emp_best_joint,
            behavioral_gain_over_best_static=gain_over_static,
            nominal_routing_accuracy=nominal_acc,
            utility_routing_accuracy=utility_acc,
            mean_latency_ms=float(np.mean([o.routing_latency_ms for o in query_outcomes])),
            superior_to_best_static_route=(gain_over_static > 0.0),
        )
