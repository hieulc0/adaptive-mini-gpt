"""Recurrent Neural Reasoner Harness (M31 Phase 2 Approach C).

Applies a compact, position-independent neural transition repeatedly across:
  - Arithmetic: Single-column carry transition (unrolled from 2 to 32 digits).
  - Logic: Single-rule deduction transition unrolled across multi-hop reasoning chains.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.reasoning.scratchpad import (
    ParsedReasoningCompletion,
    ReasoningDomain,
    ReasoningSample,
)


@dataclass
class RecurrentExecutionTrace:
    """Detailed execution trace of a recurrent neural rollout."""

    sample_id: str
    domain: ReasoningDomain
    steps_executed: int
    step_predictions: list[str]
    step_ground_truths: list[str]
    step_accuracies: list[bool]
    first_error_step: int | None
    final_answer: str
    expected_answer: str
    is_exact_match: bool
    is_halt_clean: bool
    total_tokens_generated: int


def rollout_recurrent_arithmetic(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    a: int,
    b: int,
    device: torch.device,
    max_steps: int = 40,
    max_tokens_per_step: int = 16,
) -> RecurrentExecutionTrace:
    """Autonomously unroll recurrent column addition step-by-step."""
    model.eval()
    eos_id = getattr(tokenizer, "eos_id", -1)
    max_ctx = getattr(model.config, "context_length", 256)

    str_a = str(a)
    str_b = str(b)
    rev_a = [int(d) for d in reversed(str_a)]
    rev_b = [int(d) for d in reversed(str_b)]
    max_cols = max(len(rev_a), len(rev_b))

    carry = 0
    col = 0
    output_digits: list[int] = []
    step_preds: list[str] = []
    step_truths: list[str] = []
    step_accs: list[bool] = []
    first_err: int | None = None
    clean_halt = False
    tokens_count = 0

    while col < max_cols or carry > 0:
        if col >= max_steps:
            break

        da = rev_a[col] if col < len(rev_a) else 0
        db = rev_b[col] if col < len(rev_b) else 0
        expected_col_sum = da + db + carry
        expected_w = expected_col_sum % 10
        expected_next_carry = expected_col_sum // 10
        step_truths.append(f"{expected_col_sum} > {expected_w} : {expected_next_carry}")

        step_prompt = f"<|user|>\nCol: {da} + {db} + {carry}\n<|assistant|>\n"
        enc_prompt = tokenizer.encode(step_prompt)
        input_ids = torch.tensor([enc_prompt[-max_ctx:]], dtype=torch.long, device=device)

        gen_tokens: list[int] = []
        with torch.no_grad():
            for _ in range(max_tokens_per_step):
                out = model(input_ids[:, -max_ctx:])
                logits = out[0] if isinstance(out, tuple) else out
                next_id = int(torch.argmax(logits[0, -1, :]).item())
                if next_id == eos_id:
                    break
                gen_tokens.append(next_id)
                tokens_count += 1
                input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
                decoded_partial = tokenizer.decode(gen_tokens)
                if "\n" in decoded_partial:
                    break

        step_output = tokenizer.decode(gen_tokens).strip()
        step_preds.append(step_output)

        # Parse neural prediction: e.g. "13 > 3 : 1" or "3 > 3 : 0"
        m = re.search(r"(\d+)\s*>\s*(\d+)\s*:\s*(\d+)", step_output)
        if m:
            pred_w = int(m.group(2))
            pred_next_c = int(m.group(3))
            is_step_correct = (pred_w == expected_w and pred_next_c == expected_next_carry)
        else:
            # Fallback parse for single digit
            digits = [int(d) for d in re.findall(r"\d+", step_output)]
            if digits:
                pred_w = digits[-1] % 10
                pred_next_c = 0
            else:
                pred_w = 0
                pred_next_c = 0
            is_step_correct = False

        step_accs.append(is_step_correct)
        if not is_step_correct and first_err is None:
            first_err = col

        output_digits.append(pred_w)
        carry = pred_next_c
        col += 1

        if "HALT" in step_output:
            clean_halt = True
            break

    # Build final answer string (reversed to most-significant digit)
    final_ans_str = "".join(str(d) for d in reversed(output_digits)).lstrip("0") or "0"
    expected_ans_str = str(a + b)
    is_exact = (final_ans_str == expected_ans_str)

    return RecurrentExecutionTrace(
        sample_id=f"rec_arith_{a}_{b}",
        domain=ReasoningDomain.ARITHMETIC,
        steps_executed=col,
        step_predictions=step_preds,
        step_ground_truths=step_truths,
        step_accuracies=step_accs,
        first_error_step=first_err,
        final_answer=final_ans_str,
        expected_answer=expected_ans_str,
        is_exact_match=is_exact,
        is_halt_clean=clean_halt,
        total_tokens_generated=tokens_count,
    )


def rollout_recurrent_logic(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    sample: ReasoningSample,
    device: torch.device,
    max_steps: int = 10,
    max_tokens_per_step: int = 16,
) -> RecurrentExecutionTrace:
    """Autonomously unroll recurrent multi-hop logic deduction step-by-step."""
    model.eval()
    eos_id = getattr(tokenizer, "eos_id", -1)
    max_ctx = getattr(model.config, "context_length", 256)

    step_preds: list[str] = []
    step_truths: list[str] = []
    step_accs: list[bool] = []
    first_err: int | None = None
    clean_halt = False
    tokens_count = 0
    final_val_str = ""

    for step_idx, (rec_prompt, rec_target) in enumerate(sample.recurrent_steps):
        if step_idx >= max_steps:
            break

        step_truths.append(rec_target.strip())
        enc_prompt = tokenizer.encode(rec_prompt)
        input_ids = torch.tensor([enc_prompt[-max_ctx:]], dtype=torch.long, device=device)

        gen_tokens: list[int] = []
        with torch.no_grad():
            for _ in range(max_tokens_per_step):
                out = model(input_ids[:, -max_ctx:])
                logits = out[0] if isinstance(out, tuple) else out
                next_id = int(torch.argmax(logits[0, -1, :]).item())
                if next_id == eos_id:
                    break
                gen_tokens.append(next_id)
                tokens_count += 1
                input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
                decoded_partial = tokenizer.decode(gen_tokens)
                if "\n" in decoded_partial:
                    break

        step_output = tokenizer.decode(gen_tokens).strip()
        step_preds.append(step_output)

        is_step_correct = (step_output.split("|")[0].strip() == rec_target.split("|")[0].strip())
        step_accs.append(is_step_correct)
        if not is_step_correct and first_err is None:
            first_err = step_idx

        # Update last known fact
        if "=" in step_output:
            final_val_str = step_output.split("=")[-1].split("|")[0].strip()
        elif "True" in step_output:
            final_val_str = "True"
        elif "False" in step_output:
            final_val_str = "False"

        if "HALT" in step_output:
            clean_halt = True
            break

    is_exact = (final_val_str.lower() == sample.expected_final_answer.lower())

    return RecurrentExecutionTrace(
        sample_id=sample.sample_id,
        domain=ReasoningDomain.LOGIC,
        steps_executed=len(step_preds),
        step_predictions=step_preds,
        step_ground_truths=step_truths,
        step_accuracies=step_accs,
        first_error_step=first_err,
        final_answer=final_val_str,
        expected_answer=sample.expected_final_answer,
        is_exact_match=is_exact,
        is_halt_clean=clean_halt,
        total_tokens_generated=tokens_count,
    )
