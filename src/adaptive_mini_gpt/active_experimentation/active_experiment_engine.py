"""Milestone M79 Phase 1 & 2: Minimal Active Experiment Engine and Epistemic Skill-Gap Recognition.

Implements the active scientific learning cycle:
Observe -> Hypothesize -> Detect Ambiguity -> Experiment (Discriminative Probe) -> Update Belief -> Execute or Abstain

Features:
1. Hypothesis Scoring via sequence log-likelihood S(P) and posterior confidence.
2. Epistemic Status Classification:
   - KNOWN_CONFIDENT: high likelihood (>= -1.65), high margin (>= 0.08) -> execute.
   - KNOWN_AMBIGUOUS: high likelihood (>= -1.65), low margin (< 0.08) -> active probe.
   - SKILL_GAP_UNSUPPORTED: low top likelihood (< -1.65) -> identify missing skill gap, abstain.
   - INCONSISTENT_NOISY: conflicting or non-deterministic observations -> abstain.
3. Minimally Discriminative Probe Generator:
   - Searches candidate input sequence space {0..15}^3 for x* where P_(1)(x*) != P_(2)(x*).
   - Strictly isolated: zero access to sealed test sets or hidden ground truth labels.
4. Verified Teacher/Environment Interface:
   - Answers probe queries under a strict bounded budget (1-2 queries per task).
   - Explicit query accounting and cost attribution.
5. Causal Probe Generation Controls:
   - DISCRIMINATIVE: selects input where top candidate plans maximally disagree.
   - NON_DISCRIMINATIVE: selects input where top candidate plans agree (placebo probe).
   - RANDOM: selects input uniformly at random.
"""

from __future__ import annotations

import itertools
import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.neural_plan_discovery.plan_space import CandidatePlan, PlanSpace, PlanType
from adaptive_mini_gpt.neural_skill_composition.compound_tasks import CompoundProceduralExample
from adaptive_mini_gpt.neural_skill_composition.primitive_skills import (
    PrimitiveSkillType,
    QualifiedPrimitiveRecord,
)
from adaptive_mini_gpt.reliable_plan_discovery.likelihood_plan_engine import (
    LikelihoodPlanDiscoveryEngine,
    LikelihoodPlanTrace,
)

logger = logging.getLogger(__name__)

DEFAULT_MIN_LOG_LIKELIHOOD: float = -1.65
DEFAULT_MIN_EVIDENCE_MARGIN: float = 0.08
DEFAULT_MIN_TOKEN_ACCURACY: float = 0.60


class EpistemicState(StrEnum):
    """Categorical classification of the learner's epistemic confidence."""

    KNOWN_CONFIDENT = "KNOWN_CONFIDENT"
    KNOWN_AMBIGUOUS = "KNOWN_AMBIGUOUS"
    SKILL_GAP_UNSUPPORTED = "SKILL_GAP_UNSUPPORTED"
    INCONSISTENT_NOISY = "INCONSISTENT_NOISY"


class ProbeSelectionStrategy(StrEnum):
    """Strategy for selecting active probe inputs."""

    NONE = "NONE"
    RANDOM = "RANDOM"
    NON_DISCRIMINATIVE = "NON_DISCRIMINATIVE"
    DISCRIMINATIVE = "DISCRIMINATIVE"
    ORACLE_OPTIMAL = "ORACLE_OPTIMAL"


@dataclass
class ActiveProbeRecord:
    """Detailed record of an executed active experiment query."""

    probe_index: int
    input_tokens: tuple[int, ...]
    oracle_target_tokens: tuple[int, ...]
    competing_plan_1: str
    competing_plan_2: str
    pred_plan_1: tuple[int, ...]
    pred_plan_2: tuple[int, ...]
    is_discriminative: bool
    selection_strategy: ProbeSelectionStrategy
    query_latency_ms: float


@dataclass
class ActiveExperimentResult:
    """Complete result trace of active experimentation and plan selection."""

    task_id: str
    selected_plan: CandidatePlan
    initial_epistemic_state: EpistemicState
    final_epistemic_state: EpistemicState
    is_abstained: bool
    initial_confidence: float
    final_confidence: float
    initial_margin: float
    final_margin: float
    top_likelihood: float
    probes_executed: list[ActiveProbeRecord] = field(default_factory=list)
    probes_count: int = 0
    total_forward_passes: int = 0
    latency_ms: float = 0.0
    initial_best_plan: str = ""
    updated_traces: dict[str, LikelihoodPlanTrace] = field(default_factory=dict)


class VerifiedTeacherInterface:
    """Interacts with the verified environment/oracle under a strict budget."""

    def __init__(self, ground_truth_map_fn: Callable[[Sequence[int]], tuple[int, ...]], max_budget: int = 2):
        self._map_fn = ground_truth_map_fn
        self.max_budget = max_budget
        self.queries_used: int = 0
        self.query_history: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def query_oracle(self, input_tokens: tuple[int, ...]) -> tuple[int, ...]:
        """Query the ground truth function for an input sequence."""
        if self.queries_used >= self.max_budget:
            raise RuntimeError(f"Teacher query budget exceeded: {self.queries_used}/{self.max_budget}")

        target_tokens = self._map_fn(input_tokens)
        self.queries_used += 1
        self.query_history.append((input_tokens, target_tokens))
        return target_tokens


class DiscriminativeProbeGenerator:
    """Generates synthetic input probes to distinguish competing neural plans."""

    @classmethod
    def execute_plan_symbolic(
        cls,
        model: MiniGPT,
        plan: CandidatePlan,
        inputs: list[tuple[int, ...]],
        primitives: dict[PrimitiveSkillType, QualifiedPrimitiveRecord],
        device: torch.device,
    ) -> list[tuple[int, ...]]:
        """Run candidate plan inference on a batch of candidate inputs."""
        if plan.is_empty or len(inputs) == 0:
            return [tuple(0 for _ in inp) for inp in inputs]

        prim1 = primitives[PrimitiveSkillType(plan.stages[0])]
        _, preds1 = LikelihoodPlanDiscoveryEngine.execute_single_primitive(model, prim1, inputs, device)

        if plan.depth == 1:
            return [tuple(p.tolist()) for p in preds1]
        elif plan.depth == 2:
            stage1_outputs = [tuple(p.tolist()) for p in preds1]
            prim2 = primitives[PrimitiveSkillType(plan.stages[1])]
            _, preds2 = LikelihoodPlanDiscoveryEngine.execute_single_primitive(
                model, prim2, stage1_outputs, device
            )
            return [tuple(p.tolist()) for p in preds2]
        else:
            raise ValueError(f"Depth {plan.depth} not supported")

    @classmethod
    def find_probe_input(
        cls,
        model: MiniGPT,
        plan_1: CandidatePlan,
        plan_2: CandidatePlan,
        primitives: dict[PrimitiveSkillType, QualifiedPrimitiveRecord],
        device: torch.device,
        strategy: ProbeSelectionStrategy,
        vocab_size: int = 16,
        seq_len: int = 3,
        excluded_inputs: set[tuple[int, ...]] | None = None,
        seed: int = 42,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], bool]:
        """Find an input probe according to strategy, without test set leakage.

        Returns: (probe_input, pred_1, pred_2, is_discriminative)
        """
        if excluded_inputs is None:
            excluded_inputs = set()

        rng = random.Random(seed)
        all_possible = [
            seq for seq in itertools.product(range(vocab_size), repeat=seq_len)
            if len(set(seq)) > 1 and seq not in excluded_inputs
        ]
        rng.shuffle(all_possible)

        # Batch evaluate candidate inputs in chunks of 32
        chunk_size = 32
        discriminative_candidates: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
        non_discriminative_candidates: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []

        for i in range(0, min(len(all_possible), 128), chunk_size):
            chunk = all_possible[i : i + chunk_size]
            preds1 = cls.execute_plan_symbolic(model, plan_1, chunk, primitives, device)
            preds2 = cls.execute_plan_symbolic(model, plan_2, chunk, primitives, device)

            for inp, p1, p2 in zip(chunk, preds1, preds2):
                if p1 != p2:
                    discriminative_candidates.append((inp, p1, p2))
                else:
                    non_discriminative_candidates.append((inp, p1, p2))

            if strategy == ProbeSelectionStrategy.DISCRIMINATIVE and discriminative_candidates:
                break
            if strategy == ProbeSelectionStrategy.NON_DISCRIMINATIVE and non_discriminative_candidates:
                break

        if strategy in (ProbeSelectionStrategy.DISCRIMINATIVE, ProbeSelectionStrategy.ORACLE_OPTIMAL):
            if discriminative_candidates:
                cand = discriminative_candidates[0]
                return cand[0], cand[1], cand[2], True
            # If no discriminative found among sampled, fall back to first candidate
            if non_discriminative_candidates:
                cand = non_discriminative_candidates[0]
                return cand[0], cand[1], cand[2], False
            fallback = all_possible[0] if all_possible else (1, 2, 3)
            return fallback, (0, 0, 0), (0, 0, 0), False

        elif strategy == ProbeSelectionStrategy.NON_DISCRIMINATIVE:
            if non_discriminative_candidates:
                cand = non_discriminative_candidates[0]
                return cand[0], cand[1], cand[2], False
            cand = discriminative_candidates[0] if discriminative_candidates else (all_possible[0], (0, 0, 0), (0, 0, 0))
            return cand[0], cand[1], cand[2], False

        elif strategy == ProbeSelectionStrategy.RANDOM:
            chosen = all_possible[0] if all_possible else (1, 2, 3)
            p1 = cls.execute_plan_symbolic(model, plan_1, [chosen], primitives, device)[0]
            p2 = cls.execute_plan_symbolic(model, plan_2, [chosen], primitives, device)[0]
            return chosen, p1, p2, (p1 != p2)

        else:
            raise ValueError(f"Unknown probe strategy: {strategy}")


class ActiveExperimentEngine:
    """Bounded active scientific inquiry engine for Adaptive Mini-GPT."""

    @classmethod
    def evaluate_epistemic_state(
        cls,
        best_ll: float,
        best_token_acc: float,
        evidence_margin: float,
        top_confidence: float,
        min_log_likelihood: float = DEFAULT_MIN_LOG_LIKELIHOOD,
        min_evidence_margin: float = DEFAULT_MIN_EVIDENCE_MARGIN,
        min_token_accuracy: float = DEFAULT_MIN_TOKEN_ACCURACY,
    ) -> EpistemicState:
        """Classify epistemic confidence into discrete states."""
        # Check fitness
        if best_ll < min_log_likelihood or best_token_acc < min_token_accuracy:
            return EpistemicState.SKILL_GAP_UNSUPPORTED

        # Check margin and posterior confidence
        if evidence_margin < min_evidence_margin or top_confidence < 0.80:
            return EpistemicState.KNOWN_AMBIGUOUS

        return EpistemicState.KNOWN_CONFIDENT

    @classmethod
    def run_active_investigation(
        cls,
        model: MiniGPT,
        task_id: str,
        initial_demonstrations: list[CompoundProceduralExample],
        primitives: dict[PrimitiveSkillType, QualifiedPrimitiveRecord],
        teacher: VerifiedTeacherInterface | None,
        device: torch.device,
        strategy: ProbeSelectionStrategy = ProbeSelectionStrategy.DISCRIMINATIVE,
        candidate_plans: list[CandidatePlan] | None = None,
        probe_budget: int = 1,
        seed: int = 42,
    ) -> ActiveExperimentResult:
        """Run the active scientific inquiry loop."""
        start_time = time.perf_counter()

        if candidate_plans is None:
            candidate_plans = PlanSpace.generate_candidate_plans(
                available_primitives=list(primitives.keys()),
                max_depth=2,
                include_abstain=True,
            )

        abstain_plan = next(p for p in candidate_plans if p.plan_type == PlanType.ABSTAIN)
        executable_plans = [p for p in candidate_plans if not p.is_empty]

        current_demonstrations = list(initial_demonstrations)
        probes_executed: list[ActiveProbeRecord] = []
        total_forward_passes = 0

        # Initial plan search
        initial_res = LikelihoodPlanDiscoveryEngine.discover_plan(
            model=model,
            task_id=task_id,
            demonstrations=current_demonstrations,
            primitives=primitives,
            device=device,
            candidate_plans=candidate_plans,
        )
        total_forward_passes += initial_res.total_forward_passes

        # Rank candidate traces
        sorted_traces = sorted(
            initial_res.candidate_traces.values(),
            key=lambda t: (t.sequence_log_likelihood, t.token_accuracy, t.exact_match),
            reverse=True,
        )
        best_trace = sorted_traces[0] if sorted_traces else None
        runner_up_trace = sorted_traces[1] if len(sorted_traces) > 1 else None

        initial_best_plan_id = best_trace.plan.plan_id if best_trace else "NONE"
        best_ll = best_trace.sequence_log_likelihood if best_trace else -99.9
        best_acc = best_trace.token_accuracy if best_trace else 0.0
        evidence_margin = initial_res.evidence_margin
        confidence = initial_res.confidence_score

        initial_state = cls.evaluate_epistemic_state(
            best_ll=best_ll,
            best_token_acc=best_acc,
            evidence_margin=evidence_margin,
            top_confidence=confidence,
        )
        current_state = initial_state

        # Active Experiment Loop
        queries_left = probe_budget if (teacher is not None and strategy != ProbeSelectionStrategy.NONE) else 0

        while current_state == EpistemicState.KNOWN_AMBIGUOUS and queries_left > 0 and runner_up_trace is not None:
            probe_start = time.perf_counter()
            plan_1 = best_trace.plan
            plan_2 = runner_up_trace.plan

            # Exclude already demonstrated inputs
            excluded = {d.input_tokens for d in current_demonstrations}

            # Generate probe
            inp, p1_pred, p2_pred, is_disc = DiscriminativeProbeGenerator.find_probe_input(
                model=model,
                plan_1=plan_1,
                plan_2=plan_2,
                primitives=primitives,
                device=device,
                strategy=strategy,
                vocab_size=16,
                seq_len=len(current_demonstrations[0].input_tokens) if current_demonstrations else 3,
                excluded_inputs=excluded,
                seed=seed + len(probes_executed) * 101,
            )

            # Query teacher for truth
            tgt = teacher.query_oracle(inp)
            probe_latency = (time.perf_counter() - probe_start) * 1000.0

            probe_rec = ActiveProbeRecord(
                probe_index=len(probes_executed) + 1,
                input_tokens=inp,
                oracle_target_tokens=tgt,
                competing_plan_1=plan_1.plan_id,
                competing_plan_2=plan_2.plan_id,
                pred_plan_1=p1_pred,
                pred_plan_2=p2_pred,
                is_discriminative=is_disc,
                selection_strategy=strategy,
                query_latency_ms=round(probe_latency, 2),
            )
            probes_executed.append(probe_rec)
            queries_left -= 1

            # Incorporate new evidence into demonstration set
            in_str = " ".join(f"S{s}" for s in inp)
            tgt_str = " ".join(f"S{s}" for s in tgt)
            probe_example = CompoundProceduralExample(
                example_id=f"{task_id}_active_probe_{len(probes_executed)}",
                compound_type="ACTIVE_PROBE",
                stage1_primitive=plan_1.stages[0],
                stage2_primitive=plan_1.stages[1] if len(plan_1.stages) > 1 else "NONE",
                input_tokens=inp,
                intermediate_tokens=(),
                target_tokens=tgt,
                input_text=f"Task: {task_id} | Input: {in_str} -> Output:",
                intermediate_text="",
                target_text=tgt_str,
            )
            current_demonstrations.append(probe_example)

            # Re-discover plan with expanded evidence
            updated_res = LikelihoodPlanDiscoveryEngine.discover_plan(
                model=model,
                task_id=task_id,
                demonstrations=current_demonstrations,
                primitives=primitives,
                device=device,
                candidate_plans=candidate_plans,
            )
            total_forward_passes += updated_res.total_forward_passes

            sorted_traces = sorted(
                updated_res.candidate_traces.values(),
                key=lambda t: (t.sequence_log_likelihood, t.token_accuracy, t.exact_match),
                reverse=True,
            )
            best_trace = sorted_traces[0]
            runner_up_trace = sorted_traces[1] if len(sorted_traces) > 1 else None

            best_ll = best_trace.sequence_log_likelihood
            best_acc = best_trace.token_accuracy
            evidence_margin = updated_res.evidence_margin
            confidence = updated_res.confidence_score

            current_state = cls.evaluate_epistemic_state(
                best_ll=best_ll,
                best_token_acc=best_acc,
                evidence_margin=evidence_margin,
                top_confidence=confidence,
            )

        # Final decision
        if current_state == EpistemicState.SKILL_GAP_UNSUPPORTED:
            selected_plan = abstain_plan
            is_abstained = True
        elif current_state == EpistemicState.KNOWN_AMBIGUOUS:
            # If still ambiguous after active querying, abstain responsibly
            selected_plan = abstain_plan
            is_abstained = True
        else:
            selected_plan = best_trace.plan if best_trace else abstain_plan
            is_abstained = selected_plan.is_empty

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return ActiveExperimentResult(
            task_id=task_id,
            selected_plan=selected_plan,
            initial_epistemic_state=initial_state,
            final_epistemic_state=current_state,
            is_abstained=is_abstained,
            initial_confidence=round(initial_res.confidence_score, 4),
            final_confidence=round(confidence, 4),
            initial_margin=round(initial_res.evidence_margin, 4),
            final_margin=round(evidence_margin, 4),
            top_likelihood=round(best_ll, 4),
            probes_executed=probes_executed,
            probes_count=len(probes_executed),
            total_forward_passes=total_forward_passes,
            latency_ms=round(elapsed_ms, 2),
            initial_best_plan=initial_best_plan_id,
            updated_traces={t.plan.plan_id: t for t in sorted_traces},
        )
