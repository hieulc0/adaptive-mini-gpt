"""Phase 2 & 5: Autonomous Bounded Neural Plan Discovery Engine.

Given a small set of observable input/output demonstrations (N in {0, 2, 4, 8}),
the engine executes actual forward passes using available frozen neural primitives,
scores candidate compositions against the demonstrations, and selects the supported plan
or abstains if evidence is insufficient.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
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

logger = logging.getLogger(__name__)


@dataclass
class PlanEvaluationTrace:
    """Detailed trace of a candidate plan's evaluation on demonstrations."""

    plan: CandidatePlan
    exact_match: float
    token_accuracy: float
    forward_passes: int
    mean_cross_entropy: float


@dataclass
class PlanSearchResult:
    """The decision and audit log of the autonomous plan discovery engine."""

    task_id: str
    selected_plan: CandidatePlan
    is_abstained: bool
    confidence_score: float
    evidence_margin: float
    demonstrations_used: int
    total_forward_passes: int
    search_latency_ms: float
    candidate_traces: dict[str, PlanEvaluationTrace]
    fallback_requested: bool


class AutonomousPlanDiscoveryEngine:
    """Bounded, deterministic plan discovery engine executing actual neural skill inference."""

    TOKEN_ACC_THRESHOLD = 0.85
    EXACT_MATCH_THRESHOLD = 0.75
    MIN_EVIDENCE_MARGIN = 0.05

    @classmethod
    def execute_single_primitive(
        cls,
        model: MiniGPT,
        adapter_rec: QualifiedPrimitiveRecord,
        inputs: list[tuple[int, ...]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute a single neural primitive adapter on a batch of token sequences."""
        batch_size = len(inputs)
        seq_len = len(inputs[0])
        # Convert to ProceduralExample for extraction
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
    def evaluate_candidate_plan_on_demonstrations(
        cls,
        model: MiniGPT,
        plan: CandidatePlan,
        demonstrations: list[CompoundProceduralExample],
        primitives: dict[PrimitiveSkillType, QualifiedPrimitiveRecord],
        device: torch.device,
    ) -> PlanEvaluationTrace:
        """Run real forward passes for candidate plan and measure match with demonstrations."""
        if plan.is_empty or len(demonstrations) == 0:
            return PlanEvaluationTrace(
                plan=plan,
                exact_match=0.0,
                token_accuracy=0.0,
                forward_passes=0,
                mean_cross_entropy=99.9,
            )

        inputs = [d.input_tokens for d in demonstrations]
        targets = torch.tensor([list(d.target_tokens) for d in demonstrations], dtype=torch.long, device=device)
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
            # Stage 2 takes predictions of stage 1 as input
            stage1_outputs = [tuple(p.tolist()) for p in preds1]
            prim2_type = PrimitiveSkillType(plan.stages[1])
            prim2_rec = primitives[prim2_type]
            logits2, preds2 = cls.execute_single_primitive(model, prim2_rec, stage1_outputs, device)
            total_passes += len(inputs)
            final_logits = logits2
            final_preds = preds2
        else:
            raise ValueError(f"Depth {plan.depth} not supported in M77 bounded engine")

        token_acc = float((final_preds == targets).float().mean().item())
        seq_em = float((final_preds == targets).all(dim=-1).float().mean().item())
        loss = float(F.cross_entropy(final_logits.view(-1, 16), targets.view(-1)).item())

        return PlanEvaluationTrace(
            plan=plan,
            exact_match=round(seq_em, 4),
            token_accuracy=round(token_acc, 4),
            forward_passes=total_passes,
            mean_cross_entropy=round(loss, 4),
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
        exact_match_threshold: float = EXACT_MATCH_THRESHOLD,
        token_acc_threshold: float = TOKEN_ACC_THRESHOLD,
        min_margin: float = MIN_EVIDENCE_MARGIN,
    ) -> PlanSearchResult:
        """Search candidate plans against demonstrations and choose supported plan or abstain."""
        start_time = time.perf_counter()

        if candidate_plans is None:
            candidate_plans = PlanSpace.generate_candidate_plans(
                available_primitives=list(primitives.keys()),
                max_depth=2,
                include_abstain=True,
            )

        abstain_plan = next(p for p in candidate_plans if p.plan_type == PlanType.ABSTAIN)
        executable_plans = [p for p in candidate_plans if not p.is_empty]

        # Case: Zero demonstrations provided (N=0) -> cannot identify plan without observation
        if len(demonstrations) == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return PlanSearchResult(
                task_id=task_id,
                selected_plan=abstain_plan,
                is_abstained=True,
                confidence_score=0.0,
                evidence_margin=0.0,
                demonstrations_used=0,
                total_forward_passes=0,
                search_latency_ms=round(elapsed_ms, 2),
                candidate_traces={},
                fallback_requested=True,
            )

        traces: dict[str, PlanEvaluationTrace] = {}
        total_forward_passes = 0

        for plan in executable_plans:
            trace = cls.evaluate_candidate_plan_on_demonstrations(
                model=model,
                plan=plan,
                demonstrations=demonstrations,
                primitives=primitives,
                device=device,
            )
            traces[plan.plan_id] = trace
            total_forward_passes += trace.forward_passes

        # Rank executable plans by Exact Match, then Token Accuracy, then -loss
        sorted_traces = sorted(
            traces.values(),
            key=lambda t: (t.exact_match, t.token_accuracy, -t.mean_cross_entropy),
            reverse=True,
        )

        best_trace = sorted_traces[0]
        runner_up_trace = sorted_traces[1] if len(sorted_traces) > 1 else None

        best_score = best_trace.exact_match
        runner_up_score = runner_up_trace.exact_match if runner_up_trace else 0.0
        margin = best_score - runner_up_score

        # Abstention logic:
        # 1. Best plan does not meet threshold (e.g. unfamiliar or uncomposable rule)
        # 2. Ambiguity: multiple plans tie with identical score < 1.0 and margin < min_margin
        is_supported = (
            best_score >= exact_match_threshold or best_trace.token_accuracy >= token_acc_threshold
        )

        # If best candidate is poor, or if demonstrations are too ambiguous to discriminate
        if not is_supported:
            selected_plan = abstain_plan
            is_abstained = True
            fallback = True
        elif margin < min_margin and best_score < 1.0:
            # Ambiguity detected among candidates without decisive evidence
            selected_plan = abstain_plan
            is_abstained = True
            fallback = True
        else:
            selected_plan = best_trace.plan
            is_abstained = False
            fallback = False

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return PlanSearchResult(
            task_id=task_id,
            selected_plan=selected_plan,
            is_abstained=is_abstained,
            confidence_score=best_score,
            evidence_margin=round(margin, 4),
            demonstrations_used=len(demonstrations),
            total_forward_passes=total_forward_passes,
            search_latency_ms=round(elapsed_ms, 2),
            candidate_traces=traces,
            fallback_requested=fallback,
        )
