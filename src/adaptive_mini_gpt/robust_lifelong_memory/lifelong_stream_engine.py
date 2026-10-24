"""Milestone M72 Phase 4 & 5: Robust Lifelong Stream Engine with 8-Arm Causal Ablation.

Executes a 20-episode sequential stream across 8 causal ablation arms:
1. RESET: Discard newly learned adapters after each episode (resets to initial 3 source skills).
2. RETAIN_ALL: Preserve all qualified adapters and route greedily without selective confidence threshold.
3. SELECTIVE_RETAIN: Preserve qualified adapters, route selectively (tau_select >= 0.45, delta_margin >= 0.05),
   and fall back to base or active source skill when confidence is inadequate.
4. RETAIN_DISABLED: Preserve candidate weights in storage as DORMANT, but route strictly over initial source skills.
   (Isolates parameter storage cost drag from routing capability).
5. WRONG_ROUTE: Intentionally force selection of an incompatible or adversarial adapter when available
   (Direct empirical measure of negative transfer).
6. FRESH_ADAPT: Train fresh compact parameters on each episode under matched total compute (zero retention).
7. ALWAYS_REUSE: Select from the original qualified repertoire only (0 optimizer steps).
8. OFFLINE_BEST_ROUTE: Hindsight diagnostic evaluating all available repertoire adapters on test queries
   to select the empirical maximum (0 queries, 0 steps; theoretical upper bound).
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_mini_gpt.budgeted_autonomous_learning.budget_regimes import (
    BudgetConstraint,
    FairBudgetLedger,
)
from adaptive_mini_gpt.continual_self_improvement.rational_controller import (
    RationalActionType,
    RationalExperimentController,
)
from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.modular_skill_learning.repertoire import (
    Rank16ResidualAdapter,
    extract_model_hidden,
)
from adaptive_mini_gpt.robust_lifelong_memory.capacity_accounting import (
    CompleteModelCapacityTracker,
)
from adaptive_mini_gpt.robust_lifelong_memory.skill_lifecycle import (
    RobustSkillRegistry,
    SkillState,
)
from adaptive_mini_gpt.selective_neural_memory.selective_router import (
    RoutingDecision,
    SelectiveSkillRouter,
)

logger = logging.getLogger(__name__)


@dataclass
class M72EpisodeRecord:
    """Record of an episode within the 20-episode lifelong stream."""
    episode_index: int
    task_name: str
    rule_family: str
    world_id: str
    is_repeated: bool
    is_adversarial: bool
    action_taken: str
    selected_skill_id: str
    selected_state: str
    zero_shot_macro_f1: float
    adapted_macro_f1: float
    accuracy: float
    classwise_recall: dict[str, float]
    paired_correctness: float
    queries_consumed: int
    optimizer_steps_consumed: int
    active_parameters: int
    stored_parameters: int
    net_utility: float
    is_realized_transfer: bool
    is_negative_transfer: bool
    candidate_accepted: bool


@dataclass
class M72StreamResult:
    """Complete multi-episode stream result under a specific causal arm."""
    arm_name: str
    stream_id: str
    seed: int
    episodes: list[M72EpisodeRecord] = field(default_factory=list)
    mean_macro_f1: float = 0.0
    mean_net_utility: float = 0.0
    repeated_zero_shot_f1: float = 0.0
    realized_transfers: int = 0
    negative_transfers: int = 0
    final_active_skills: int = 0
    final_dormant_skills: int = 0
    final_stored_params: int = 0
    total_optimizer_steps: int = 0
    total_queries: int = 0
    total_gpu_seconds: float = 0.0


class RobustLifelongStreamEngine:
    """Engine executing the 20-episode stream across 8 causal ablation arms."""

    def __init__(
        self,
        cost_ledger: FairBudgetLedger | None = None,
        rational_controller: RationalExperimentController | None = None,
        selective_router: SelectiveSkillRouter | None = None,
    ) -> None:
        self.cost_ledger = cost_ledger or FairBudgetLedger()
        self.controller = rational_controller or RationalExperimentController()
        self.router = selective_router or SelectiveSkillRouter(tau_select=0.45, delta_margin=0.05)

    @staticmethod
    def _train_candidate_adapter(
        model: MiniGPT,
        train_env: Any,
        steps: int,
        device: torch.device,
        seed: int = 42,
    ) -> Rank16ResidualAdapter:
        """Train an isolated rank-16 candidate adapter with live gradient descent."""
        torch.manual_seed(seed)
        adapter = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3).to(device)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)

        lbl_map = {"NO": 0, "YES": 1, "UNKNOWN": 2}
        train_queries = getattr(train_env, "held_out_test_queries", getattr(train_env, "examples", []))[:16]
        if not train_queries:
            return adapter

        cached_data = []
        with torch.no_grad():
            for ex in train_queries:
                if isinstance(ex, dict):
                    q = ex.get("question_text", ex.get("text", ""))
                    tgt = lbl_map.get(ex.get("expected_answer", "UNKNOWN"), 2)
                else:
                    q = getattr(ex, "question_text", "")
                    tgt = lbl_map.get(getattr(ex, "expected_answer", "UNKNOWN"), 2)
                toks = model.tokenizer.encode(q)
                inp = torch.tensor([toks], dtype=torch.long, device=device)
                h = extract_model_hidden(model, inp)
                cached_data.append((h, tgt))

        adapter.train()
        for step in range(steps):
            idx = step % len(cached_data)
            h, tgt = cached_data[idx]
            optimizer.zero_grad()
            logits = adapter(h)
            loss = F.cross_entropy(logits, torch.tensor([tgt], device=device))
            loss.backward()
            optimizer.step()

        adapter.eval()
        return adapter

    def run_stream(
        self,
        arm_name: str,
        stream_id: str,
        seed: int,
        model: MiniGPT,
        initial_registry: RobustSkillRegistry,
        task_sequence: list[dict[str, Any]],
        budget: BudgetConstraint,
        device: torch.device,
    ) -> M72StreamResult:
        """Execute a 20-episode sequential stream under the specified causal arm."""
        registry = copy.deepcopy(initial_registry)
        episodes: list[M72EpisodeRecord] = []

        total_realized_transfers = 0
        total_negative_transfers = 0
        repeated_zero_shot_scores: list[float] = []

        total_steps = 0
        total_queries = 0
        total_gpu_time = 0.0

        for ep in task_sequence:
            t_start = time.time()
            ep_idx = ep["episode_index"]
            task_name = ep["task_name"]
            rule_fam = ep["rule_family"]
            world_id = ep["world_id"]
            is_repeated = ep["is_repeated_encounter"]
            is_adversarial = ep["is_adversarial"]
            tr_env = ep["train_env"]
            qual_env = ep["qual_env"]
            ev_env = ep["eval_env"]

            # Development queries for routing and qualification (NEVER sealed evaluation queries)
            dev_queries = getattr(qual_env, "held_out_test_queries", getattr(qual_env, "examples", []))[:8]
            # Evaluation queries for measuring held-out performance
            eval_queries = getattr(ev_env, "held_out_test_queries", getattr(ev_env, "examples", []))

            # -------------------------------------------------------------
            # 1. Routing Strategy by Causal Arm
            # -------------------------------------------------------------
            selected_sk_id = "base_model"
            selected_state = SkillState.ACTIVE
            routing_res: RoutingDecision | None = None

            if arm_name == "RESET":
                # Resets after each episode: routes only over initial source skills
                routing_res = self.router.route_task(task_name, model, registry, dev_queries, device, allow_dormant_probing=False)
                selected_sk_id = routing_res.selected_skill_id
                selected_state = routing_res.selected_state

            elif arm_name == "ALWAYS_REUSE":
                # Only routes over initial source skills (0 steps ever)
                routing_res = self.router.route_task(task_name, model, registry, dev_queries, device, allow_dormant_probing=False)
                selected_sk_id = routing_res.selected_skill_id
                selected_state = routing_res.selected_state

            elif arm_name == "RETAIN_DISABLED":
                # Acquired skills are kept DORMANT in storage; routing is restricted to initial source skills
                routing_res = self.router.route_task(task_name, model, registry, dev_queries, device, allow_dormant_probing=False)
                selected_sk_id = routing_res.selected_skill_id
                selected_state = routing_res.selected_state

            elif arm_name == "RETAIN_ALL":
                # Greedy routing over all active/acquired skills WITHOUT confidence margin
                all_skills = registry.list_skills()
                best_f1 = -1.0
                best_id = "base_model"
                for sk_id in all_skills:
                    rec = registry.get_skill(sk_id)
                    if rec.state == SkillState.REJECTED:
                        continue
                    f1, _ = self.router.evaluate_adapter_on_queries(rec.adapter, model, dev_queries, device)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_id = sk_id

                selected_sk_id = best_id
                selected_state = registry.get_skill(best_id).state if best_id in registry._records else SkillState.ACTIVE
                routing_res = RoutingDecision(
                    task_name=task_name,
                    selected_skill_id=best_id,
                    selected_state=selected_state,
                    probe_f1=max(best_f1, 0.3333),
                    best_source_f1=0.3333,
                    candidate_probe_f1=best_f1,
                    is_candidate_routed=best_id.startswith("cand_"),
                    candidate_accepted_for_routing=True,
                    reason="Greedy probe routing without selective threshold",
                )

            elif arm_name == "WRONG_ROUTE":
                # Deliberately select incompatible/adversarial adapter on adversarial tasks
                if is_adversarial:
                    # Invert route: select Transitive DAG or Rail adapter on adversarial tasks
                    cand_ids = [k for k in registry.list_skills() if "dag" in k.lower() or "rail" in k.lower()]
                    if cand_ids:
                        selected_sk_id = cand_ids[0]
                    else:
                        selected_sk_id = "base_model"
                    selected_state = SkillState.ACTIVE
                    f1, _ = self.router.evaluate_adapter_on_queries(
                        registry.get_skill(selected_sk_id).adapter if selected_sk_id in registry._records else None,
                        model,
                        dev_queries,
                        device,
                    )
                    routing_res = RoutingDecision(
                        task_name=task_name,
                        selected_skill_id=selected_sk_id,
                        selected_state=selected_state,
                        probe_f1=f1,
                        best_source_f1=0.3333,
                        candidate_probe_f1=f1,
                        is_candidate_routed=True,
                        candidate_accepted_for_routing=True,
                        reason="Forced adversarial route",
                    )
                else:
                    routing_res = self.router.route_task(task_name, model, registry, dev_queries, device, allow_dormant_probing=True)
                    selected_sk_id = routing_res.selected_skill_id
                    selected_state = routing_res.selected_state

            elif arm_name == "SELECTIVE_RETAIN":
                # Selective routing with confidence threshold (tau_select >= 0.45, delta_margin >= 0.05)
                routing_res = self.router.route_task(task_name, model, registry, dev_queries, device, allow_dormant_probing=True)
                selected_sk_id = routing_res.selected_skill_id
                selected_state = routing_res.selected_state

            elif arm_name == "FRESH_ADAPT":
                # No retention; fresh model routing
                routing_res = self.router.route_task(task_name, model, registry, dev_queries, device, allow_dormant_probing=False)
                selected_sk_id = routing_res.selected_skill_id
                selected_state = routing_res.selected_state

            else:  # OFFLINE_BEST_ROUTE
                # Hindsight diagnostic upper bound: Evaluates all available adapters on test queries
                all_skills = registry.list_skills()
                best_test_f1 = -1.0
                best_test_id = "base_model"
                for sk_id in all_skills:
                    rec = registry.get_skill(sk_id)
                    f1, _ = self.router.evaluate_adapter_on_queries(rec.adapter, model, eval_queries, device)
                    if f1 > best_test_f1:
                        best_test_f1 = f1
                        best_test_id = sk_id
                selected_sk_id = best_test_id
                selected_state = registry.get_skill(best_test_id).state if best_test_id in registry._records else SkillState.ACTIVE
                routing_res = RoutingDecision(
                    task_name=task_name,
                    selected_skill_id=best_test_id,
                    selected_state=selected_state,
                    probe_f1=best_test_f1,
                    best_source_f1=0.3333,
                    candidate_probe_f1=best_test_f1,
                    is_candidate_routed=best_test_id.startswith("cand_"),
                    candidate_accepted_for_routing=True,
                    reason="Hindsight optimal test route",
                )

            zero_shot_probe_f1 = routing_res.probe_f1 if routing_res else 0.3333

            # -------------------------------------------------------------
            # 2. Action Decision (Train Candidate vs Reuse)
            # -------------------------------------------------------------
            if arm_name == "ALWAYS_REUSE":
                action = RationalActionType.REUSE_QUALIFIED_SKILL
            elif arm_name == "FRESH_ADAPT":
                action = RationalActionType.EXPERIMENT_AND_ADAPT
            elif arm_name == "OFFLINE_BEST_ROUTE":
                action = RationalActionType.REUSE_QUALIFIED_SKILL
            else:
                decision = self.controller.evaluate_action_for_task(
                    task_name=task_name,
                    best_skill_id=selected_sk_id,
                    best_skill_probe_f1=zero_shot_probe_f1,
                    budget=budget,
                )
                action = decision.action

            is_transfer = False
            is_neg_transfer = False
            candidate_accepted = False
            queries_used = 2
            steps_used = 0

            # -------------------------------------------------------------
            # 3. Execution of Action
            # -------------------------------------------------------------
            if action == RationalActionType.EXPERIMENT_AND_ADAPT:
                steps_used = min(budget.max_steps_per_task, 40)
                queries_used = min(budget.max_training_queries, 16)
                cand_ad = self._train_candidate_adapter(model, tr_env, steps=steps_used, device=device, seed=seed + ep_idx)

                # Validate candidate on development queries (NOT test)
                val_f1, _ = self.router.evaluate_adapter_on_queries(cand_ad, model, dev_queries, device)
                accepted, gain, reason = self.controller.decide_candidate_acceptance(
                    candidate_id=f"cand_{task_name}",
                    held_out_val_f1=val_f1,
                    baseline_f1=zero_shot_probe_f1,
                )

                if accepted or arm_name == "FRESH_ADAPT":
                    candidate_accepted = True
                    cand_id = f"cand_{task_name}"
                    initial_state = SkillState.DORMANT if arm_name == "RETAIN_DISABLED" else SkillState.ACTIVE
                    registry.register_skill(
                        skill_id=cand_id,
                        rule_family=rule_fam,
                        source_env_id=task_name,
                        adapter=cand_ad,
                        macro_f1=val_f1,
                        accuracy=val_f1,
                        initial_state=initial_state,
                        created_episode=ep_idx,
                    )
                    eval_f1, eval_acc = self.router.evaluate_adapter_on_queries(cand_ad, model, eval_queries, device)
                else:
                    routed_rec = registry.get_skill(selected_sk_id) if selected_sk_id in registry._records else None
                    ad_to_eval = routed_rec.adapter if routed_rec else None
                    eval_f1, eval_acc = self.router.evaluate_adapter_on_queries(ad_to_eval, model, eval_queries, device)
            else:
                # Reusing existing skill (0 optimizer steps)
                routed_rec = registry.get_skill(selected_sk_id) if selected_sk_id in registry._records else None
                ad_to_eval = routed_rec.adapter if routed_rec else None
                eval_f1, eval_acc = self.router.evaluate_adapter_on_queries(ad_to_eval, model, eval_queries, device)

                # Check for realized positive transfer on repeated encounters
                if is_repeated and routing_res and routing_res.is_candidate_routed and eval_f1 >= 0.45:
                    is_transfer = True
                    total_realized_transfers += 1

                # Check for negative transfer on adversarial tasks
                if is_adversarial and eval_f1 < 0.3333:
                    is_neg_transfer = True
                    total_negative_transfers += 1

            if is_repeated:
                repeated_zero_shot_scores.append(eval_f1)

            # In RESET arm: discard any created candidate at end of episode
            if arm_name == "RESET" or arm_name == "FRESH_ADAPT":
                registry = copy.deepcopy(initial_registry)

            # Capacity metrics
            capacity_audit = CompleteModelCapacityTracker.audit_capacity(
                backbone_model=model,
                registry=registry,
                routed_skill_id=selected_sk_id,
            )

            total_steps += steps_used
            total_queries += queries_used
            gpu_time = max(0.005, time.time() - t_start)
            total_gpu_time += gpu_time

            outcome = self.cost_ledger.compute_outcome(
                policy_name=f"{arm_name}_{stream_id}",
                budget_tier=budget.tier,
                task_name=task_name,
                macro_f1=eval_f1,
                accuracy=eval_acc,
                queries_used=queries_used,
                steps_used=steps_used,
                stored_params=capacity_audit.total_stored_adapter_params,
                active_params=capacity_audit.active_inference_params - capacity_audit.backbone_params,
                gpu_seconds=gpu_time,
                false_allocations=1 if is_neg_transfer else 0,
                missed_reuses=0,
            )

            episodes.append(
                M72EpisodeRecord(
                    episode_index=ep_idx,
                    task_name=task_name,
                    rule_family=rule_fam,
                    world_id=world_id,
                    is_repeated=is_repeated,
                    is_adversarial=is_adversarial,
                    action_taken=action.value,
                    selected_skill_id=selected_sk_id,
                    selected_state=selected_state.value if isinstance(selected_state, SkillState) else str(selected_state),
                    zero_shot_macro_f1=zero_shot_probe_f1,
                    adapted_macro_f1=eval_f1,
                    accuracy=eval_acc,
                    classwise_recall={},
                    paired_correctness=eval_acc,
                    queries_consumed=queries_used,
                    optimizer_steps_consumed=steps_used,
                    active_parameters=capacity_audit.active_inference_params,
                    stored_parameters=capacity_audit.total_stored_model_params,
                    net_utility=outcome.amortized_utility,
                    is_realized_transfer=is_transfer,
                    is_negative_transfer=is_neg_transfer,
                    candidate_accepted=candidate_accepted,
                )
            )

        mean_f1 = float(np.mean([e.adapted_macro_f1 for e in episodes]))
        mean_utility = float(np.mean([e.net_utility for e in episodes]))
        mean_rep_f1 = float(np.mean(repeated_zero_shot_scores)) if repeated_zero_shot_scores else 0.3333

        return M72StreamResult(
            arm_name=arm_name,
            stream_id=stream_id,
            seed=seed,
            episodes=episodes,
            mean_macro_f1=mean_f1,
            mean_net_utility=mean_utility,
            repeated_zero_shot_f1=mean_rep_f1,
            realized_transfers=total_realized_transfers,
            negative_transfers=total_negative_transfers,
            final_active_skills=len(registry.list_active_skills()),
            final_dormant_skills=len(registry.list_dormant_skills()),
            final_stored_params=registry.total_stored_parameters(),
            total_optimizer_steps=total_steps,
            total_queries=total_queries,
            total_gpu_seconds=total_gpu_time,
        )
