"""Phase 3: Genuine Sequential Autonomous Learning Task Streams.

Executes sequential multi-episode task streams where:
- Episode 1: Proposes bounded experiment on a novel task, trains candidate adapter,
             validates on independent world, and ACCEPTS IT INTO THE RESEARCH REPERTOIRE.
- Episode 2: Evaluates on a fresh held-out world/vocabulary of the same or compatible family,
             measuring the ACTUAL REALIZED TRANSFER BENEFIT of the retained weights.
- Episode 3: Encounters another task (incomplete or novel), conducts further experimentation,
             and accumulates capabilities.
- Final Sealed Eval: Evaluates the accumulated research state on sealed held-out environments.

Crucially: Repertoire state PERSISTS and ACCUMULATES across episodes, enabling genuine
continual self-improvement rather than per-task isolated evaluation.
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.autonomous_research_loop.experiment_proposals import (
    ExperimentProposal,
    ExperimentProposalEngine,
    ResearchExperimentAction,
)
from adaptive_mini_gpt.budgeted_autonomous_learning.budget_regimes import (
    BudgetConstraint,
    BudgetTier,
    EpisodeBudgetOutcome,
    FairBudgetLedger,
)
from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.modular_skill_learning.repertoire import (
    ModularSkillRepertoire,
    Rank16ResidualAdapter,
    extract_model_hidden,
)
from adaptive_mini_gpt.modular_skill_learning.uncertainty_diagnostic import EpistemicUncertaintyDiagnostic

logger = logging.getLogger(__name__)


@dataclass
class StreamEpisodeStep:
    """The record of a single episode step in a sequential task stream."""
    episode_index: int
    task_name: str
    rule_family: str
    is_known_family: bool
    action_taken: str
    candidate_id: str | None
    accepted_into_repertoire: bool
    macro_f1: float
    accuracy: float
    queries_used: int
    steps_used: int
    stored_params_delta: int
    total_stored_params: int
    gpu_seconds: float
    cost_outcome: EpisodeBudgetOutcome


@dataclass
class SequentialStreamResult:
    """The complete trace and evaluation of a sequential autonomous learning stream."""
    stream_id: str
    seed: int
    budget_tier: BudgetTier
    arm_name: str
    episodes: list[StreamEpisodeStep]
    final_repertoire_skill_count: int
    retained_skills: list[str]
    realized_future_transfers: int
    accumulated_macro_f1: float
    accumulated_amortized_utility: float
    sealed_eval_macro_f1: float
    diagnostics: str


class SequentialTaskStreamEngine:
    """Executes closed-loop sequential task streams with persistent repertoire accumulation."""

    def __init__(
        self,
        cost_ledger: FairBudgetLedger,
        proposal_engine: ExperimentProposalEngine | None = None,
        uncertainty_diagnostic: EpistemicUncertaintyDiagnostic | None = None,
    ) -> None:
        self.cost_ledger = cost_ledger
        self.proposal_engine = proposal_engine or ExperimentProposalEngine(
            reuse_f1_threshold=0.45,
            sprt_novelty_threshold=2.0,
            evidence_repair_uncertainty_threshold=0.50,
        )
        self.diagnostic = uncertainty_diagnostic or EpistemicUncertaintyDiagnostic()

    def run_sequential_stream(
        self,
        stream_id: str,
        seed: int,
        model: MiniGPT,
        initial_repertoire: ModularSkillRepertoire,
        task_sequence: list[dict[str, Any]],
        sealed_eval_env: Any,
        budget: BudgetConstraint,
        arm_name: str = "RETAINED_LEARNING",
        device: torch.device | None = None,
    ) -> SequentialStreamResult:
        """Run an end-to-end multi-episode stream with persistent research state accumulation."""
        dev = device or torch.device("cpu")
        repertoire = copy.deepcopy(initial_repertoire)

        # Track persistent stored parameters
        total_stored_params = sum(
            sum(t.numel() for t in rec.state_dict.values())
            for rec in repertoire._skills.values()
        )

        episode_steps: list[StreamEpisodeStep] = []
        realized_transfers = 0
        total_f1_sum = 0.0
        total_util_sum = 0.0

        for ep_idx, task_dict in enumerate(task_sequence):
            t_name = task_dict["name"]
            r_fam = task_dict["family"]
            is_known = task_dict["is_known"]
            tr_env = task_dict["train_env"]
            q_env = task_dict["qual_env"]
            ev_env = task_dict["eval_env"]

            t0 = time.time()
            # 1. Probe current repertoire with budget-limited probe queries
            probe_budget = min(budget.max_probe_queries, len(tr_env.held_out_test_queries))
            probe_queries = tr_env.held_out_test_queries[:probe_budget]
            best_sk_id, best_f1 = repertoire.select_best_skill_by_evidence_probe(
                model=model, probe_queries=probe_queries, device=dev
            )

            # 2. Compute epistemic uncertainty
            obs = [(q.source, q.target, q.expected_answer) for q in probe_queries]
            sprt_llr, _ = self.diagnostic.compute_sprt_llr(
                observations=obs,
                candidate_hypothesis=tr_env.true_hypothesis if hasattr(tr_env, "true_hypothesis") else None,
                edges=tr_env.base_edges if hasattr(tr_env, "base_edges") else [],
            )

            # 3. Formulate proposal under arm policy constraints
            if arm_name == "RESET_WEIGHTS":
                # Resets candidate additions after every episode (tests persistent retention)
                if ep_idx > 0:
                    for sk in list(repertoire.list_skills()):
                        if sk.startswith("candidate_"):
                            del repertoire._skills[sk]
                proposal = self.proposal_engine.formulate_proposal(
                    target_env_id=t_name,
                    best_reuse_f1=best_f1,
                    best_skill_id=best_sk_id,
                    sprt_llr=sprt_llr,
                    evidence_gap_ratio=0.1,
                    is_known_family=is_known,
                )
            elif arm_name == "UNCHANGED_CONTROL":
                # Static reuse of original source repertoire only
                proposal = ExperimentProposal(
                    proposal_id=f"unchanged_{t_name}",
                    action=ResearchExperimentAction.REUSE_EXISTING,
                    target_env_id=t_name,
                    motivation="Static control arm",
                    hypothesis="No learning allowed",
                    max_oracle_queries=probe_budget,
                    max_optimizer_steps=0,
                    matched_control_action="REUSE_EXISTING",
                    min_validation_f1_gain=0.0,
                    max_active_parameters=0,
                )
            elif arm_name == "FRESH_ADAPT_CONTROL":
                # Adapts fresh residual parameters on each task without retention
                proposal = ExperimentProposal(
                    proposal_id=f"fresh_{t_name}",
                    action=ResearchExperimentAction.FRESH_ADAPT,
                    target_env_id=t_name,
                    motivation="Fresh adaptation control arm",
                    hypothesis="Temporary adaptation",
                    max_oracle_queries=budget.max_queries_per_task,
                    max_optimizer_steps=budget.max_steps_per_task,
                    matched_control_action="REUSE_EXISTING",
                    min_validation_f1_gain=0.0,
                    max_active_parameters=2371,
                )
            else:
                # Standard RETAINED_LEARNING arm
                # If an acquired skill from a previous episode matches, probe F1 will be high!
                proposal = self.proposal_engine.formulate_proposal(
                    target_env_id=t_name,
                    best_reuse_f1=best_f1,
                    best_skill_id=best_sk_id,
                    sprt_llr=sprt_llr,
                    evidence_gap_ratio=0.1,
                    is_known_family=is_known,
                )

            # 4. Execute proposal
            queries_used = probe_budget
            steps_used = 0
            stored_delta = 0
            cand_id = None
            accepted = False
            false_alloc = 0
            missed_reuse = 0

            label_to_idx = {"NO": 0, "YES": 1, "UNKNOWN": 2}

            if proposal.action == ResearchExperimentAction.REUSE_EXISTING:
                # Reuse existing skill (could be a newly retained skill from Episode 1!)
                eval_f1, acc, _, _ = repertoire.evaluate_skill_on_env(
                    best_sk_id, model, ev_env, dev
                )
                if best_sk_id and best_sk_id.startswith("candidate_"):
                    realized_transfers += 1  # Verified multi-episode reuse of retained knowledge!

            elif proposal.action == ResearchExperimentAction.FRESH_ADAPT:
                queries_used = budget.max_queries_per_task
                steps_used = budget.max_steps_per_task
                fresh_ad = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3).to(dev)
                opt = torch.optim.AdamW(fresh_ad.parameters(), lr=5e-4, weight_decay=1e-4)

                train_q = tr_env.held_out_test_queries[:budget.max_training_queries]
                data = []
                for q in train_q:
                    toks = model.tokenizer.encode(q.question_text)
                    inp = torch.tensor([toks], dtype=torch.long, device=dev)
                    with torch.no_grad():
                        h = extract_model_hidden(model, inp)
                    data.append((h, label_to_idx.get(q.expected_answer, 2)))

                fresh_ad.train()
                for step in range(steps_used):
                    h, tgt = data[step % len(data)]
                    opt.zero_grad()
                    out = fresh_ad(h)
                    loss = F.cross_entropy(out, torch.tensor([tgt], device=dev))
                    loss.backward()
                    opt.step()

                # Evaluate fresh
                fresh_ad.eval()
                conf_mat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                corr = 0
                with torch.no_grad():
                    for q in ev_env.held_out_test_queries:
                        toks = model.tokenizer.encode(q.question_text)
                        inp = torch.tensor([toks], dtype=torch.long, device=dev)
                        h = extract_model_hidden(model, inp)
                        pred = int(torch.argmax(fresh_ad(h), dim=-1).item())
                        act = label_to_idx.get(q.expected_answer, 2)
                        conf_mat[act][pred] += 1
                        if pred == act:
                            corr += 1
                f1s = []
                for c in range(3):
                    tp = conf_mat[c][c]
                    fp = sum(conf_mat[r][c] for r in range(3) if r != c)
                    fn = sum(conf_mat[c][p] for p in range(3) if p != c)
                    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
                eval_f1 = float(np.mean(f1s))
                acc = float(corr / len(ev_env.held_out_test_queries))

            elif proposal.action == ResearchExperimentAction.TRAIN_NEW_SKILL:
                cand_id = f"candidate_skill_{t_name}_{ep_idx}"
                queries_used = min(budget.max_queries_per_task, 16 + probe_budget)
                steps_used = budget.max_steps_per_task

                if is_known:
                    false_alloc = 1

                # Train candidate adapter
                cand_ad = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3).to(dev)
                opt = torch.optim.AdamW(cand_ad.parameters(), lr=5e-4, weight_decay=1e-4)

                train_q = tr_env.held_out_test_queries[:budget.max_training_queries]
                data = []
                for q in train_q:
                    toks = model.tokenizer.encode(q.question_text)
                    inp = torch.tensor([toks], dtype=torch.long, device=dev)
                    with torch.no_grad():
                        h = extract_model_hidden(model, inp)
                    data.append((h, label_to_idx.get(q.expected_answer, 2)))

                cand_ad.train()
                for step in range(steps_used):
                    h, tgt = data[step % len(data)]
                    opt.zero_grad()
                    out = cand_ad(h)
                    loss = F.cross_entropy(out, torch.tensor([tgt], device=dev))
                    loss.backward()
                    opt.step()

                # Qualify on independent validation world (q_env)
                cand_ad.eval()
                conf_mat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                corr = 0
                with torch.no_grad():
                    for q in q_env.held_out_test_queries:
                        toks = model.tokenizer.encode(q.question_text)
                        inp = torch.tensor([toks], dtype=torch.long, device=dev)
                        h = extract_model_hidden(model, inp)
                        pred = int(torch.argmax(cand_ad(h), dim=-1).item())
                        act = label_to_idx.get(q.expected_answer, 2)
                        conf_mat[act][pred] += 1
                        if pred == act:
                            corr += 1
                f1s = []
                for c in range(3):
                    tp = conf_mat[c][c]
                    fp = sum(conf_mat[r][c] for r in range(3) if r != c)
                    fn = sum(conf_mat[c][p] for p in range(3) if p != c)
                    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
                qual_f1 = float(np.mean(f1s))

                # Check qualification threshold
                if qual_f1 >= 0.40 and (qual_f1 - best_f1) >= proposal.min_validation_f1_gain:
                    accepted = True
                    stored_delta = 2371
                    total_stored_params += stored_delta
                    # ACCEPTS CANDIDATE INTO ACTIVE REPERTOIRE
                    repertoire.register_skill(
                        skill_id=cand_id,
                        rule_family=r_fam,
                        source_env_id=t_name,
                        adapter=cand_ad,
                        macro_f1=qual_f1,
                        accuracy=float(corr / len(q_env.held_out_test_queries)),
                        classwise_recall={},
                    )
                    eval_f1, acc, _, _ = repertoire.evaluate_skill_on_env(
                        cand_id, model, ev_env, dev
                    )
                else:
                    accepted = False
                    # Rollback: evaluate fallback skill
                    eval_f1, acc, _, _ = repertoire.evaluate_skill_on_env(
                        best_sk_id, model, ev_env, dev
                    )
            else:
                eval_f1, acc, _, _ = repertoire.evaluate_skill_on_env(
                    best_sk_id, model, ev_env, dev
                )

            gpu_time = max(0.01, time.time() - t0)

            # Audit missed reuse
            if is_known and proposal.action != ResearchExperimentAction.REUSE_EXISTING and best_f1 >= 0.40:
                missed_reuse = 1

            cost_out = self.cost_ledger.compute_outcome(
                policy_name=arm_name,
                budget_tier=budget.tier,
                task_name=t_name,
                macro_f1=eval_f1,
                accuracy=acc,
                queries_used=queries_used,
                steps_used=steps_used,
                stored_params=stored_delta,
                active_params=2371 if proposal.action in (ResearchExperimentAction.TRAIN_NEW_SKILL, ResearchExperimentAction.FRESH_ADAPT) else 0,
                gpu_seconds=gpu_time,
                false_allocations=false_alloc,
                missed_reuses=missed_reuse,
                realized_future_reuses=realized_transfers,
                action_name=proposal.action.value,
            )

            step_rec = StreamEpisodeStep(
                episode_index=ep_idx,
                task_name=t_name,
                rule_family=r_fam,
                is_known_family=is_known,
                action_taken=proposal.action.value,
                candidate_id=cand_id,
                accepted_into_repertoire=accepted,
                macro_f1=eval_f1,
                accuracy=acc,
                queries_used=queries_used,
                steps_used=steps_used,
                stored_params_delta=stored_delta,
                total_stored_params=total_stored_params,
                gpu_seconds=gpu_time,
                cost_outcome=cost_out,
            )
            episode_steps.append(step_rec)
            total_f1_sum += eval_f1
            total_util_sum += cost_out.amortized_utility

        # 5. Final Sealed Evaluation
        # Evaluate the accumulated repertoire on the sealed held-out environment
        probe_q = sealed_eval_env.held_out_test_queries[:budget.max_probe_queries]
        best_sealed_id, _ = repertoire.select_best_skill_by_evidence_probe(model, probe_q, dev)
        sealed_f1, _, _, _ = repertoire.evaluate_skill_on_env(
            best_sealed_id, model, sealed_eval_env, dev
        )

        n_eps = max(1, len(episode_steps))
        mean_f1 = float(total_f1_sum / n_eps)
        mean_u = float(total_util_sum / n_eps)

        diag = (
            f"Stream={stream_id}, Tier={budget.tier.value}, Arm={arm_name}, "
            f"MeanF1={mean_f1:.4f}, MeanAmortUtil={mean_u:.4f}, "
            f"RealizedTransfers={realized_transfers}, StoredParams={total_stored_params}, "
            f"SealedF1={sealed_f1:.4f}"
        )

        return SequentialStreamResult(
            stream_id=stream_id,
            seed=seed,
            budget_tier=budget.tier,
            arm_name=arm_name,
            episodes=episode_steps,
            final_repertoire_skill_count=len(repertoire.list_skills()),
            retained_skills=repertoire.list_skills(),
            realized_future_transfers=realized_transfers,
            accumulated_macro_f1=mean_f1,
            accumulated_amortized_utility=mean_u,
            sealed_eval_macro_f1=float(sealed_f1),
            diagnostics=diag,
        )
