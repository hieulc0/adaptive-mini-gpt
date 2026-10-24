"""Phase 2: Likelihood-Based Autonomous Neural Plan Discovery Engine.

Computes normalized sequence log-likelihood for candidate plans:
    S(P) = (1/N) * sum_{i=1}^N (1 / |y_i|) * sum_{t=1}^{|y_i|} log p_P(y_i,t | x_i)
and calibrated posterior probabilities over candidate plans.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch
import torch.nn.functional as F

from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.neural_plan_discovery.plan_space import CandidatePlan, PlanSpace, PlanType
from adaptive_mini_gpt.neural_skill_composition.compound_tasks import CompoundProceduralExample
from adaptive_mini_gpt.neural_skill_composition.primitive_skills import (
    PrimitiveSkillType,
    QualifiedPrimitiveRecord,
)
from adaptive_mini_gpt.neural_weight_learning.pathway_instrumentation import (
    NeuralPathwayInstrumenter,
)
from adaptive_mini_gpt.neural_weight_learning.procedural_tasks import ProceduralExample

logger = logging.getLogger(__name__)\


class PlanExecutionStatus(StrEnum):
    """Execution status returned by the likelihood planner."""

    EXECUTED = "EXECUTED"
    ABSTAINED_UNSUPPORTED = "ABSTAINED_UNSUPPORTED"
    ABSTAINED_AMBIGUOUS = "ABSTAINED_AMBIGUOUS"
    ABSTAINED_NO_EVIDENCE = "ABSTAINED_NO_EVIDENCE"


@dataclass
class LikelihoodPlanTrace:
    """Trace of candidate plan evaluation via normalized sequence log-likelihood."""

    plan: CandidatePlan
    sequence_log_likelihood: float  # S(P) = -mean_cross_entropy
    mean_cross_entropy: float
    token_accuracy: float
    exact_match: float
    posterior_prob: float = 0.0
    forward_passes: int = 0


@dataclass
class ReliablePlanSearchResult:
    """Result of reliable autonomous plan discovery."""

    task_id: str
    selected_plan: CandidatePlan
    execution_status: PlanExecutionStatus
    is_abstained: bool
    confidence_score: float  # Posterior probability or normalized confidence
    evidence_margin: float   # S(P_(1)) - S(P_(2))
    top_likelihood: float
    demonstrations_used: int
    total_forward_passes: int
    search_latency_ms: float
    candidate_traces: dict[str, LikelihoodPlanTrace] = field(default_factory=dict)
    active_evidence_requested: bool = False


class LikelihoodPlanDiscoveryEngine:
    """Reliable autonomous plan discovery engine using normalized sequence log-likelihood."""

    # Default calibrated thresholds
    # S(P) = -cross_entropy. A random guess over vocab=16 is -log(16) ~= -2.77
    # An unfamiliar task has loss ~ 2.4 - 3.5 (S(P) ~ -2.4 to -3.5)
    # A supported task has loss ~ 0.35 - 1.2 (S(P) ~ -0.35 to -1.20)
    DEFAULT_MIN_LOG_LIKELIHOOD: float = -1.65  # Equivalent to CE <= 1.65
    DEFAULT_MIN_EVIDENCE_MARGIN: float = 0.08  # S(P_(1)) - S(P_(2)) >= 0.08 in log space
    DEFAULT_MIN_TOKEN_ACCURACY: float = 0.50

    @classmethod
    def execute_single_primitive(
        cls,
        model: MiniGPT,
        adapter_rec: QualifiedPrimitiveRecord,
        inputs: list[tuple[int, ...]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute a single neural primitive adapter on token sequences."""
        dummy_examples = [
            ProceduralExample(
                example_id=f"probe_{i}",
                rule_family="PROBE",
                input_tokens=seq,
                target_tokens=seq,
                input_text=" ".join(f"S{s}" for s in seq),
                target_text=" ".join(f"S{s}" for s in seq),
            )
            for i, seq in enumerate(inputs)
        ]

        h, _ = NeuralPathwayInstrumenter.extract_symbol_hidden_states(model, dummy_examples, device)
        adapter = adapter_rec.adapter
        adapter.eval()
        with torch.no_grad():
            logits = adapter(h)
            preds = torch.argmax(logits, dim=-1)

        return logits, preds

    @classmethod
    def evaluate_candidate_plan_likelihood(
        cls,
        model: MiniGPT,
        plan: CandidatePlan,
        demonstrations: list[CompoundProceduralExample],
        primitives: dict[PrimitiveSkillType, QualifiedPrimitiveRecord],
        device: torch.device,
    ) -> LikelihoodPlanTrace:
        """Compute exact normalized sequence log-likelihood for a candidate plan on demonstrations.

        S(P) = (1/N) * sum_{i=1}^N (1 / |y_i|) * sum_{t=1}^{|y_i|} log p_P(y_i,t | x_i)
        """
        if plan.is_empty or len(demonstrations) == 0:
            return LikelihoodPlanTrace(
                plan=plan,
                sequence_log_likelihood=-99.9,
                mean_cross_entropy=99.9,
                token_accuracy=0.0,
                exact_match=0.0,
                forward_passes=0,
            )

        inputs = [d.input_tokens for d in demonstrations]
        targets = torch.tensor(
            [list(d.target_tokens) for d in demonstrations], dtype=torch.long, device=device
        )
        total_passes = 0

        # Stage 1 execution
        prim1_type = PrimitiveSkillType(plan.stages[0])
        prim1_rec = primitives[prim1_type]
        logits1, preds1 = cls.execute_single_primitive(model, prim1_rec, inputs, device)
        total_passes += len(inputs)

        if plan.depth == 1:
            final_logits = logits1
            final_preds = preds1
        elif plan.depth == 2:
            # Stage 2 receives predictions of stage 1
            stage1_outputs = [tuple(p.tolist()) for p in preds1]
            prim2_type = PrimitiveSkillType(plan.stages[1])
            prim2_rec = primitives[prim2_type]
            logits2, preds2 = cls.execute_single_primitive(model, prim2_rec, stage1_outputs, device)
            total_passes += len(inputs)
            final_logits = logits2
            final_preds = preds2
        else:
            raise ValueError(f"Depth {plan.depth} not supported in bounded M78 engine")

        # Compute log-softmax
        log_probs = F.log_softmax(final_logits, dim=-1)  # [N, seq_len, vocab_size]
        # Gather target log-likelihoods
        target_log_probs = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)  # [N, seq_len]
        # Per-sequence mean log likelihood
        seq_log_likelihood = target_log_probs.mean(dim=-1)  # [N]
        mean_seq_ll = float(seq_log_likelihood.mean().item())

        token_acc = float((final_preds == targets).float().mean().item())
        seq_em = float((final_preds == targets).all(dim=-1).float().mean().item())
        ce_loss = float(F.cross_entropy(final_logits.view(-1, 16), targets.view(-1)).item())

        return LikelihoodPlanTrace(
            plan=plan,
            sequence_log_likelihood=round(mean_seq_ll, 4),
            mean_cross_entropy=round(ce_loss, 4),
            token_accuracy=round(token_acc, 4),
            exact_match=round(seq_em, 4),
            forward_passes=total_passes,
        )

    @classmethod
    def discover_plan(
        cls,
        model: MiniGPT,
        task_id: str,
        demonstrations: list[CompoundProceduralExample],
        primitives: dict[PrimitiveSkillType, QualifiedPrimitiveRecord],
        device: torch.device,
        candidate_plans: list[CandidatePlan] | None = None,
        min_log_likelihood: float = DEFAULT_MIN_LOG_LIKELIHOOD,
        min_evidence_margin: float = DEFAULT_MIN_EVIDENCE_MARGIN,
        min_token_accuracy: float = DEFAULT_MIN_TOKEN_ACCURACY,
        temperature: float = 1.0,
    ) -> ReliablePlanSearchResult:
        """Search candidate plans using sequence log-likelihood and calibrated confidence."""
        start_time = time.perf_counter()

        if candidate_plans is None:
            candidate_plans = PlanSpace.generate_candidate_plans(
                available_primitives=list(primitives.keys()),
                max_depth=2,
                include_abstain=True,
            )

        abstain_plan = next(p for p in candidate_plans if p.plan_type == PlanType.ABSTAIN)
        executable_plans = [p for p in candidate_plans if not p.is_empty]

        # Case 1: Zero demonstrations provided (N=0) -> cannot infer plan without observations
        if len(demonstrations) == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ReliablePlanSearchResult(
                task_id=task_id,
                selected_plan=abstain_plan,
                execution_status=PlanExecutionStatus.ABSTAINED_NO_EVIDENCE,
                is_abstained=True,
                confidence_score=0.0,
                evidence_margin=0.0,
                top_likelihood=-99.9,
                demonstrations_used=0,
                total_forward_passes=0,
                search_latency_ms=round(elapsed_ms, 2),
                candidate_traces={},
                active_evidence_requested=True,
            )

        traces: dict[str, LikelihoodPlanTrace] = {}
        total_forward_passes = 0

        for plan in executable_plans:
            trace = cls.evaluate_candidate_plan_likelihood(
                model=model,
                plan=plan,
                demonstrations=demonstrations,
                primitives=primitives,
                device=device,
            )
            traces[plan.plan_id] = trace
            total_forward_passes += trace.forward_passes

        # Rank candidate plans by sequence log-likelihood descending (higher is better)
        sorted_traces = sorted(
            traces.values(),
            key=lambda t: (t.sequence_log_likelihood, t.token_accuracy, t.exact_match),
            reverse=True,
        )

        best_trace = sorted_traces[0]
        runner_up_trace = sorted_traces[1] if len(sorted_traces) > 1 else None

        best_ll = best_trace.sequence_log_likelihood
        runner_up_ll = runner_up_trace.sequence_log_likelihood if runner_up_trace else -99.9
        evidence_margin = max(0.0, best_ll - runner_up_ll)

        # Compute posterior probabilities over executable plans via softmax with temperature
        # Scale temperature by sequence length * demonstrations count for total evidence
        evidence_scale = max(1.0, min(10.0, float(len(demonstrations) * 3)))
        ll_tensor = torch.tensor([t.sequence_log_likelihood for t in sorted_traces], dtype=torch.float32)
        posterior_probs = F.softmax(ll_tensor * (evidence_scale / temperature), dim=0)

        for i, trace in enumerate(sorted_traces):
            trace.posterior_prob = round(float(posterior_probs[i].item()), 4)

        top_confidence = round(float(posterior_probs[0].item()), 4)

        # Calibrated Decision Logic:
        # 1. Absolute Fitness: Does the best plan assign adequate likelihood to demonstrations?
        #    If not, the task is unsupported/unfamiliar (outside known primitive repertoire).
        has_adequate_fitness = (
            best_ll >= min_log_likelihood and best_trace.token_accuracy >= min_token_accuracy
        )

        # 2. Relative Margin: Is the best plan clearly separated from the runner-up?
        #    If top candidate is not distinct from runner-up, evidence is ambiguous.
        has_adequate_margin = evidence_margin >= min_evidence_margin

        if not has_adequate_fitness:
            # Unfamiliar / uncomposable task detected -> correctly abstain
            selected_plan = abstain_plan
            status = PlanExecutionStatus.ABSTAINED_UNSUPPORTED
            is_abstained = True
            active_req = True
        elif not has_adequate_margin and top_confidence < 0.80:
            # Ambiguous evidence across multiple plans -> request additional demonstration
            selected_plan = abstain_plan
            status = PlanExecutionStatus.ABSTAINED_AMBIGUOUS
            is_abstained = True
            active_req = True
        else:
            # Confidently identified valid neural composition
            selected_plan = best_trace.plan
            status = PlanExecutionStatus.EXECUTED
            is_abstained = False
            active_req = False

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return ReliablePlanSearchResult(
            task_id=task_id,
            selected_plan=selected_plan,
            execution_status=status,
            is_abstained=is_abstained,
            confidence_score=top_confidence,
            evidence_margin=round(evidence_margin, 4),
            top_likelihood=best_ll,
            demonstrations_used=len(demonstrations),
            total_forward_passes=total_forward_passes,
            search_latency_ms=round(elapsed_ms, 2),
            candidate_traces=traces,
            active_evidence_requested=active_req,
        )
