"""Closed-Loop Interaction with Step-Level Error Diagnosis and Targeted Practice (M31 Phase 5).

Implements the end-to-end self-improvement loop:
  1. Interaction encounter & failure detection.
  2. Diagnosis of first failing step in the intermediate trace.
  3. Generation of targeted practice curriculum with intermediate reasoning steps.
  4. Candidate fine-tuning with multi-domain replay.
  5. Evaluation on fresh unseen problems & retention verification.
  6. Automated gate decision (Accept / Reject) comparing Scratchpad vs Direct practice.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from adaptive_mini_gpt.data.language_corpus import normalize_text
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.evaluation.multi_domain import DomainType
from adaptive_mini_gpt.interaction.knowledge_gap import (
    ErrorCategory,
    KnowledgeGapRecord,
)
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.reasoning.dataset import (
    MultiRepresentationReasoningDataset,
    collate_reasoning_batch,
)
from adaptive_mini_gpt.reasoning.evaluator import (
    evaluate_reasoning_sample_free_rollout,
)
from adaptive_mini_gpt.reasoning.scratchpad import (
    ApproachType,
    ReasoningDomain,
    ReasoningSample,
    build_arithmetic_reasoning_sample,
)
from adaptive_mini_gpt.rules.reliable_transition import (
    ElementaryTransitionDataset,
    audit_elementary_transition_space,
    collate_elementary_batch,
    get_all_200_transitions,
)

logger = logging.getLogger(__name__)


@dataclass
class StepDiagnosisResult:
    """Diagnostic isolation of the first failing step."""

    gap_record: KnowledgeGapRecord
    first_failing_step_index: int
    step_description: str
    target_curriculum_domain: str


@dataclass
class ClosedLoopReasoningComparison:
    """Comparison between Scratchpad-trained candidate and Direct-trained candidate."""

    initial_problem: str
    first_failing_step: int
    direct_candidate_fresh_acc: float
    scratchpad_candidate_fresh_acc: float
    direct_candidate_carry_retention: float
    scratchpad_candidate_carry_retention: float
    gate_decision: str  # "ACCEPTED", "REJECTED"
    rationale: str


def diagnose_first_failing_step(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    sample: ReasoningSample,
    device: torch.device,
) -> StepDiagnosisResult:
    """Identify the exact intermediate step where reasoning broke down."""
    is_exact, _, _, first_err, _, _, pred = evaluate_reasoning_sample_free_rollout(
        model=model,
        tokenizer=tokenizer,
        sample=sample,
        approach=ApproachType.SCRATCHPAD,
        device=device,
    )

    step_idx = first_err if first_err is not None else 0
    gt_step = (
        sample.intermediate_ground_truth[step_idx]
        if step_idx < len(sample.intermediate_ground_truth)
        else "unknown"
    )

    gap = KnowledgeGapRecord.create(
        domain=DomainType.ARITHMETIC if sample.domain == ReasoningDomain.ARITHMETIC else DomainType.LOGIC,
        question=sample.prompt_scratchpad,
        predicted_answer=pred,
        ground_truth=sample.expected_final_answer,
        confidence_score=0.50,
        verification_status="FAIL",
        error_category=ErrorCategory.REPEATED_PROCEDURAL_FAILURE,
        context=f"Failed at step {step_idx + 1}: expected {gt_step}",
    )

    return StepDiagnosisResult(
        gap_record=gap,
        first_failing_step_index=step_idx,
        step_description=f"Step {step_idx + 1} ({gt_step})",
        target_curriculum_domain=sample.domain.value,
    )


def train_bounded_candidate(
    base_model: MiniGPT,
    tokenizer: BPETokenizer,
    practice_samples: list[ReasoningSample],
    approach: ApproachType,
    device: torch.device,
    epochs: int = 10,
    lr: float = 3e-4,
    batch_size: int = 8,
) -> MiniGPT:
    """Train candidate model on targeted practice samples with balanced replay."""
    candidate = copy.deepcopy(base_model).to(device)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=lr, weight_decay=0.01)

    max_ctx = getattr(candidate.config, "context_length", 256)
    pad_id = getattr(tokenizer, "pad_id", 0)

    # Practice DataLoader
    ds_practice = MultiRepresentationReasoningDataset(
        samples=practice_samples,
        tokenizer=tokenizer,
        approach=approach,
        context_length=max_ctx,
    )
    loader_practice = DataLoader(
        ds_practice,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_reasoning_batch(b, pad_token_id=pad_id),
    )

    # Replay DataLoader (Arithmetic transitions to protect carry)
    transitions = get_all_200_transitions()
    ds_replay = ElementaryTransitionDataset(
        transitions=transitions,
        tokenizer=tokenizer,
        format_type="canonical_state",
        context_length=max_ctx,
    )
    loader_replay = DataLoader(
        ds_replay,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_elementary_batch(b, pad_token_id=pad_id),
    )

    candidate.train()
    steps_per_epoch = max(len(loader_practice), len(loader_replay))

    for _ in range(epochs):
        iter_p = iter(loader_practice)
        iter_r = iter(loader_replay)

        for _ in range(steps_per_epoch):
            try:
                xp, yp = next(iter_p)
            except StopIteration:
                iter_p = iter(loader_practice)
                xp, yp = next(iter_p)

            xp, yp = xp.to(device), yp.to(device)
            optimizer.zero_grad()
            out_p = candidate(xp)
            logits_p = out_p[0] if isinstance(out_p, tuple) else out_p
            loss_p = nn.functional.cross_entropy(
                logits_p[:, :-1, :].contiguous().view(-1, logits_p.size(-1)),
                yp[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
            loss_p.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
            optimizer.step()

            # Replay step
            try:
                xr, yr = next(iter_r)
            except StopIteration:
                iter_r = iter(loader_replay)
                xr, yr = next(iter_r)

            xr, yr = xr.to(device), yr.to(device)
            optimizer.zero_grad()
            out_r = candidate(xr)
            logits_r = out_r[0] if isinstance(out_r, tuple) else out_r
            loss_r = nn.functional.cross_entropy(
                logits_r[:, :-1, :].contiguous().view(-1, logits_r.size(-1)),
                yr[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
            loss_r.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.0)
            optimizer.step()

    return candidate


def run_closed_loop_reasoning_cycle(
    base_model: MiniGPT,
    tokenizer: BPETokenizer,
    initial_sample: ReasoningSample,
    practice_pool: list[ReasoningSample],
    fresh_test_pool: list[ReasoningSample],
    device: torch.device,
) -> ClosedLoopReasoningComparison:
    """Execute end-to-end closed-loop interaction and comparison."""
    # Step 1 & 2: Diagnose first failing step
    diagnosis = diagnose_first_failing_step(base_model, tokenizer, initial_sample, device)

    # Step 3 & 4: Train candidate using Direct approach
    cand_direct = train_bounded_candidate(
        base_model=base_model,
        tokenizer=tokenizer,
        practice_samples=practice_pool,
        approach=ApproachType.DIRECT,
        device=device,
        epochs=8,
    )

    # Train candidate using Scratchpad approach
    cand_scratchpad = train_bounded_candidate(
        base_model=base_model,
        tokenizer=tokenizer,
        practice_samples=practice_pool,
        approach=ApproachType.SCRATCHPAD,
        device=device,
        epochs=8,
    )

    # Step 5: Evaluate both on fresh held-out test problems
    direct_correct = 0
    scratchpad_correct = 0
    for s in fresh_test_pool:
        is_ex_d, _, _, _, _, _, _ = evaluate_reasoning_sample_free_rollout(
            model=cand_direct,
            tokenizer=tokenizer,
            sample=s,
            approach=ApproachType.DIRECT,
            device=device,
        )
        if is_ex_d:
            direct_correct += 1

        is_ex_s, _, _, _, _, _, _ = evaluate_reasoning_sample_free_rollout(
            model=cand_scratchpad,
            tokenizer=tokenizer,
            sample=s,
            approach=ApproachType.SCRATCHPAD,
            device=device,
        )
        if is_ex_s:
            scratchpad_correct += 1

    total_test = max(1, len(fresh_test_pool))
    acc_direct = round(direct_correct / total_test, 4)
    acc_scratchpad = round(scratchpad_correct / total_test, 4)

    # Step 6: Carry retention audit
    audit_d = audit_elementary_transition_space(cand_direct, tokenizer, device, format_type="canonical_state")
    ret_direct = round(audit_d.get("overall_carry_acc", 0.0) / 100.0, 4)

    audit_s = audit_elementary_transition_space(cand_scratchpad, tokenizer, device, format_type="canonical_state")
    ret_scratchpad = round(audit_s.get("overall_carry_acc", 0.0) / 100.0, 4)

    # Gate decision
    if acc_scratchpad > acc_direct and ret_scratchpad >= 0.80:
        decision = "ACCEPTED"
        rationale = f"Scratchpad candidate exceeded direct baseline ({acc_scratchpad*100}% vs {acc_direct*100}%) while preserving carry retention ({ret_scratchpad*100}%)."
    elif acc_scratchpad == 0.0 and acc_direct == 0.0:
        decision = "REJECTED"
        rationale = "Neither candidate produced exact match on fresh held-out problems."
    else:
        decision = "REJECTED"
        rationale = f"Scratchpad candidate ({acc_scratchpad*100}%) did not show sufficient advantage over direct ({acc_direct*100}%) or failed retention ({ret_scratchpad*100}%)."

    return ClosedLoopReasoningComparison(
        initial_problem=initial_sample.prompt_direct,
        first_failing_step=diagnosis.first_failing_step_index,
        direct_candidate_fresh_acc=acc_direct,
        scratchpad_candidate_fresh_acc=acc_scratchpad,
        direct_candidate_carry_retention=ret_direct,
        scratchpad_candidate_carry_retention=ret_scratchpad,
        gate_decision=decision,
        rationale=rationale,
    )
