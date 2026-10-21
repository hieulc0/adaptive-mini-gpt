"""M32 Phase 1: Recurrent Transition Audit and State Failure Diagnostics.

Audits:
  1. All 200 elementary decimal addition states (10 x 10 x 2) independently.
  2. Teacher-forced transition accuracy and loss vs. autonomous single-step rollout.
  3. Classification of transition failures:
     - WRONG_OUTPUT_DIGIT
     - WRONG_CARRY_STATE
     - WRONG_DIGIT_AND_CARRY
     - INCORRECT_STOPPING
     - STATE_SERIALIZATION_ERROR
     - ACCUMULATED_ROLLOUT_ERROR
  4. Multi-digit autonomous rollout failure attribution.
"""

from __future__ import annotations

import enum
import math
import re
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.rules.reliable_transition import (
    ElementaryTransition,
    format_elementary_prompt_and_target,
    get_all_200_transitions,
)


class TransitionFailureType(str, enum.Enum):
    """Classification of recurrent transition failure modes."""

    NONE = "none"
    WRONG_OUTPUT_DIGIT = "wrong_output_digit"
    WRONG_CARRY_STATE = "wrong_carry_state"
    WRONG_DIGIT_AND_CARRY = "wrong_digit_and_carry"
    INCORRECT_STOPPING = "incorrect_stopping"
    STATE_SERIALIZATION_ERROR = "state_serialization_error"
    ACCUMULATED_ROLLOUT_ERROR = "accumulated_rollout_error"


@dataclass
class ElementaryAuditRecord:
    """Detailed audit record for a single elementary transition (a, b, c_in)."""

    a: int
    b: int
    c_in: int
    expected_sum: int
    expected_write: int
    expected_carry: int
    category: str  # "none", "generate", "propagate"
    teacher_loss: float
    teacher_top1_correct: bool
    autonomous_output_raw: str
    predicted_write: int | None
    predicted_carry: int | None
    is_exact_autonomous: bool
    failure_type: TransitionFailureType


@dataclass
class RecurrentAuditSummary:
    """Consolidated summary of all 200 elementary transitions."""

    total_evaluated: int
    overall_exact_accuracy: float
    overall_digit_accuracy: float
    overall_carry_accuracy: float
    mean_teacher_loss: float
    teacher_forced_accuracy: float
    by_category: dict[str, dict[str, float]]
    failure_distribution: dict[str, int]
    failing_transitions: list[dict[str, Any]]


def audit_elementary_state(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    tr: ElementaryTransition,
    device: torch.device,
    max_tokens: int = 16,
) -> ElementaryAuditRecord:
    """Audit a single elementary transition under both teacher-forcing and autonomous generation."""
    model.eval()
    prompt_str, target_str = format_elementary_prompt_and_target(tr, format_type="m23_baseline")
    enc_prompt = tokenizer.encode(prompt_str)
    enc_target = tokenizer.encode(target_str)
    full_tokens = enc_prompt + enc_target
    max_ctx = getattr(model.config, "context_length", 256)

    # 1. Teacher-Forced Evaluation
    input_ids = torch.tensor([full_tokens[:-1]], dtype=torch.long, device=device)[:, -max_ctx:]
    targets = torch.tensor([full_tokens[1:]], dtype=torch.long, device=device)[:, -max_ctx:]
    target_start_idx = max(0, len(enc_prompt) - 1)

    loss_fn = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        out = model(input_ids)
        logits = out[0] if isinstance(out, tuple) else out
        losses = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        target_losses = losses[target_start_idx:]
        teacher_loss = float(target_losses.mean().item()) if len(target_losses) > 0 else 0.0

        preds = torch.argmax(logits, dim=-1)
        target_preds = preds[0, target_start_idx:]
        target_gts = targets[0, target_start_idx:]
        teacher_correct = bool((target_preds == target_gts).all().item()) if len(target_gts) > 0 else False

    # 2. Autonomous Rollout from Prompt
    gen_tokens = list(enc_prompt)
    eos_id = getattr(tokenizer, "eos_id", -1)
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([gen_tokens[-max_ctx:]], dtype=torch.long, device=device)
            out = model(inp)
            lg = out[0] if isinstance(out, tuple) else out
            next_id = int(torch.argmax(lg[0, -1, :]).item())
            if next_id == eos_id:
                break
            gen_tokens.append(next_id)
            decoded_suffix = tokenizer.decode(gen_tokens[len(enc_prompt):])
            if "\n" in decoded_suffix:
                break

    raw_output = tokenizer.decode(gen_tokens[len(enc_prompt):]).strip()

    # 3. Parse and Classify Failure
    m = re.search(r"(\d+)\s*>\s*(\d+)\s*:\s*(\d+)", raw_output)
    if m:
        pred_w = int(m.group(2)) % 10
        pred_c = int(m.group(3))
        has_syntax = True
    else:
        digits = [int(d) for d in re.findall(r"\d+", raw_output)]
        if len(digits) >= 2:
            pred_w = digits[1] % 10
            pred_c = digits[0] // 10 if digits[0] >= 10 else (digits[2] if len(digits) > 2 else 0)
            has_syntax = False
        elif len(digits) == 1:
            pred_w = digits[0] % 10
            pred_c = digits[0] // 10
            has_syntax = False
        else:
            pred_w = None
            pred_c = None
            has_syntax = False

    if not has_syntax or pred_w is None or pred_c is None:
        failure_type = TransitionFailureType.STATE_SERIALIZATION_ERROR
        is_exact = False
    elif pred_w == tr.write_digit and pred_c == tr.carry_out:
        failure_type = TransitionFailureType.NONE
        is_exact = True
    elif pred_w != tr.write_digit and pred_c == tr.carry_out:
        failure_type = TransitionFailureType.WRONG_OUTPUT_DIGIT
        is_exact = False
    elif pred_w == tr.write_digit and pred_c != tr.carry_out:
        failure_type = TransitionFailureType.WRONG_CARRY_STATE
        is_exact = False
    else:
        failure_type = TransitionFailureType.WRONG_DIGIT_AND_CARRY
        is_exact = False

    return ElementaryAuditRecord(
        a=tr.a,
        b=tr.b,
        c_in=tr.c_in,
        expected_sum=tr.total_sum,
        expected_write=tr.write_digit,
        expected_carry=tr.carry_out,
        category=tr.carry_category,
        teacher_loss=round(teacher_loss, 4),
        teacher_top1_correct=teacher_correct,
        autonomous_output_raw=raw_output,
        predicted_write=pred_w,
        predicted_carry=pred_c,
        is_exact_autonomous=is_exact,
        failure_type=failure_type,
    )


def audit_all_200_recurrent_states(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    device: torch.device,
) -> RecurrentAuditSummary:
    """Exhaustively audit all 200 elementary decimal addition transitions."""
    all_200 = get_all_200_transitions()
    records: list[ElementaryAuditRecord] = []
    category_counts: dict[str, dict[str, int]] = {
        "none": {"total": 0, "exact": 0, "digit": 0, "carry": 0},
        "generate": {"total": 0, "exact": 0, "digit": 0, "carry": 0},
        "propagate": {"total": 0, "exact": 0, "digit": 0, "carry": 0},
    }
    failure_dist: dict[str, int] = {f.value: 0 for f in TransitionFailureType}
    failing_list: list[dict[str, Any]] = []

    total_exact = 0
    total_digit = 0
    total_carry = 0
    total_teacher_correct = 0
    total_loss = 0.0

    for tr in all_200:
        rec = audit_elementary_state(model, tokenizer, tr, device)
        records.append(rec)
        cat = rec.category
        category_counts[cat]["total"] += 1
        total_loss += rec.teacher_loss

        if rec.teacher_top1_correct:
            total_teacher_correct += 1

        d_ok = (rec.predicted_write == rec.expected_write)
        c_ok = (rec.predicted_carry == rec.expected_carry)
        if d_ok:
            total_digit += 1
            category_counts[cat]["digit"] += 1
        if c_ok:
            total_carry += 1
            category_counts[cat]["carry"] += 1
        if rec.is_exact_autonomous:
            total_exact += 1
            category_counts[cat]["exact"] += 1

        failure_dist[rec.failure_type.value] += 1
        if not rec.is_exact_autonomous:
            failing_list.append({
                "a": rec.a,
                "b": rec.b,
                "c_in": rec.c_in,
                "expected": f"{rec.expected_sum} > {rec.expected_write} : {rec.expected_carry}",
                "predicted": f"{rec.predicted_write} : {rec.predicted_carry}",
                "category": rec.category,
                "failure_type": rec.failure_type.value,
                "raw_output": rec.autonomous_output_raw,
            })

    n = len(all_200)
    by_category_pct: dict[str, dict[str, float]] = {}
    for cat, d in category_counts.items():
        tot = max(1, d["total"])
        by_category_pct[cat] = {
            "total": d["total"],
            "exact_pct": round(d["exact"] / tot * 100.0, 2),
            "digit_pct": round(d["digit"] / tot * 100.0, 2),
            "carry_pct": round(d["carry"] / tot * 100.0, 2),
        }

    return RecurrentAuditSummary(
        total_evaluated=n,
        overall_exact_accuracy=round(total_exact / n * 100.0, 2),
        overall_digit_accuracy=round(total_digit / n * 100.0, 2),
        overall_carry_accuracy=round(total_carry / n * 100.0, 2),
        mean_teacher_loss=round(total_loss / n, 4),
        teacher_forced_accuracy=round(total_teacher_correct / n * 100.0, 2),
        by_category=by_category_pct,
        failure_distribution=failure_dist,
        failing_transitions=failing_list,
    )
