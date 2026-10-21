"""Phase 1: Diagnostic Audit of Answer-Generation Failures.

Audits why autoregressive generation fails on fresh problems despite low held-out loss:
  - Measures token-level probability of ground-truth target tokens under teacher forcing.
  - Pinpoints first incorrect generated token in autonomous rollout.
  - Compares exact match vs semantic normalization.
  - Detects premature termination vs context-budget exhaustion.
  - Distinguishes reasoning failure from answer-representation and decoding failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from adaptive_mini_gpt.data.language_corpus import normalize_text
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.transformer import MiniGPT


@dataclass
class GenerationAuditResult:
    """Detailed trace of an autonomous generation attempt."""

    sample_id: str
    prompt: str
    expected_answer: str
    generated_text: str
    is_exact_match: bool
    is_semantic_match: bool
    mean_target_token_prob: float
    first_incorrect_token_idx: int | None
    first_incorrect_token_expected: str | None
    first_incorrect_token_generated: str | None
    is_premature_termination: bool
    is_context_budget_exhausted: bool
    teacher_forced_loss: float
    error_origin: str


def audit_sample_generation(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    sample_id: str,
    prompt: str,
    expected_answer: str,
    device: torch.device,
    max_new_tokens: int = 64,
) -> GenerationAuditResult:
    """Audit single sample under teacher-forced scoring and autonomous rollout."""
    model.eval()
    max_ctx = getattr(model.config, "context_length", 256)
    pad_id = getattr(tokenizer, "pad_id", 0)
    eos_id = getattr(tokenizer, "eos_id", -1)

    enc_prompt = tokenizer.encode(prompt)
    enc_target = tokenizer.encode(expected_answer)

    # 1. Teacher-Forced Target Probability & Loss
    full_enc = enc_prompt + enc_target
    if len(full_enc) > max_ctx:
        full_enc = full_enc[-max_ctx:]

    target_probs: list[float] = []
    teacher_loss = 0.0

    with torch.no_grad():
        x_tf = torch.tensor([full_enc], dtype=torch.long, device=device)
        out_tf = model(x_tf)
        logits_tf = out_tf[0] if isinstance(out_tf, tuple) else out_tf

        # Measure log-probs for target tokens (which start at len(enc_prompt) - 1 in causal shift)
        start_idx = max(0, len(enc_prompt) - 1)
        end_idx = min(logits_tf.size(1) - 1, len(full_enc) - 1)

        total_tgt_loss = 0.0
        tgt_count = 0
        for i in range(start_idx, end_idx):
            target_token_id = full_enc[i + 1]
            token_logits = logits_tf[0, i, :]
            token_probs = nn.functional.softmax(token_logits, dim=-1)
            prob = float(token_probs[target_token_id].item())
            target_probs.append(prob)
            total_tgt_loss += -math.log(max(1e-12, prob))
            tgt_count += 1

        teacher_loss = round(total_tgt_loss / max(1, tgt_count), 4)

    mean_prob = round(sum(target_probs) / max(1, len(target_probs)), 4)

    # 2. Autonomous Rollout
    input_ids = torch.tensor([enc_prompt[-max_ctx:]], dtype=torch.long, device=device)
    generated_token_ids: list[int] = []
    premature_term = False
    context_exhausted = False

    with torch.no_grad():
        for step in range(max_new_tokens):
            if input_ids.size(1) >= max_ctx:
                context_exhausted = True
                break

            out = model(input_ids[:, -max_ctx:])
            logits = out[0] if isinstance(out, tuple) else out
            next_id = int(torch.argmax(logits[0, -1, :]).item())

            if next_id == eos_id:
                break
            generated_token_ids.append(next_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)

            decoded_so_far = tokenizer.decode(generated_token_ids)
            if "\n" in decoded_so_far and ("Final Answer:" in decoded_so_far or "Answer:" in prompt):
                # Completed response line
                break

    generated_text = tokenizer.decode(generated_token_ids).strip()
    if len(generated_token_ids) == 0:
        premature_term = True

    # 3. First Incorrect Generated Token Tracking
    first_err_idx: int | None = None
    first_err_exp: str | None = None
    first_err_gen: str | None = None

    for idx in range(min(len(enc_target), len(generated_token_ids))):
        if enc_target[idx] != generated_token_ids[idx]:
            first_err_idx = idx
            first_err_exp = tokenizer.decode([enc_target[idx]])
            first_err_gen = tokenizer.decode([generated_token_ids[idx]])
            break

    if first_err_idx is None and len(generated_token_ids) < len(enc_target):
        first_err_idx = len(generated_token_ids)
        first_err_exp = tokenizer.decode([enc_target[len(generated_token_ids)]])
        first_err_gen = "<EOS/STOP>"

    # 4. Exact vs Semantic Scoring
    norm_gen = normalize_text(generated_text)
    norm_exp = normalize_text(expected_answer)
    is_exact = (generated_text == expected_answer)
    is_semantic = (norm_gen == norm_exp) or (norm_exp in norm_gen)

    # 5. Determine Error Origin
    if is_exact or is_semantic:
        err_origin = "none"
    elif premature_term:
        err_origin = "syntax_truncation"
    elif context_exhausted:
        err_origin = "context_exhaustion"
    elif mean_prob > 0.40 and first_err_idx is not None and first_err_idx > 0:
        err_origin = "decoding_greedy_divergence"
    elif mean_prob > 0.40:
        err_origin = "answer_representation_mismatch"
    else:
        err_origin = "reasoning_misdirection"

    return GenerationAuditResult(
        sample_id=sample_id,
        prompt=prompt,
        expected_answer=expected_answer,
        generated_text=generated_text,
        is_exact_match=is_exact,
        is_semantic_match=is_semantic,
        mean_target_token_prob=mean_prob,
        first_incorrect_token_idx=first_err_idx,
        first_incorrect_token_expected=first_err_exp,
        first_incorrect_token_generated=first_err_gen,
        is_premature_termination=premature_term,
        is_context_budget_exhausted=context_exhausted,
        teacher_forced_loss=teacher_loss,
        error_origin=err_origin,
    )


def run_phase1_audit(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    samples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    """Execute Phase 1 diagnostic audit across a set of benchmark samples."""
    audit_traces: list[GenerationAuditResult] = []
    error_counts: dict[str, int] = {}

    for s in samples:
        res = audit_sample_generation(
            model=model,
            tokenizer=tokenizer,
            sample_id=s.get("id", "sample"),
            prompt=s["prompt"],
            expected_answer=s["expected"],
            device=device,
        )
        audit_traces.append(res)
        error_counts[res.error_origin] = error_counts.get(res.error_origin, 0) + 1

    total = len(audit_traces)
    exact_count = sum(1 for t in audit_traces if t.is_exact_match)
    semantic_count = sum(1 for t in audit_traces if t.is_semantic_match)
    mean_target_prob = sum(t.mean_target_token_prob for t in audit_traces) / max(1, total)
    mean_teacher_loss = sum(t.teacher_forced_loss for t in audit_traces) / max(1, total)

    return {
        "total_audited": total,
        "exact_match_count": exact_count,
        "exact_match_rate": round(exact_count / max(1, total), 4),
        "semantic_match_count": semantic_count,
        "semantic_match_rate": round(semantic_count / max(1, total), 4),
        "mean_target_token_prob": round(mean_target_prob, 4),
        "mean_teacher_loss": round(mean_teacher_loss, 4),
        "error_distribution": error_counts,
        "sample_traces": [
            {
                "id": t.sample_id,
                "expected": t.expected_answer,
                "generated": t.generated_text,
                "exact": t.is_exact_match,
                "semantic": t.is_semantic_match,
                "target_prob": t.mean_target_token_prob,
                "first_err_idx": t.first_incorrect_token_idx,
                "first_err_exp": t.first_incorrect_token_expected,
                "first_err_gen": t.first_incorrect_token_generated,
                "origin": t.error_origin,
            }
            for t in audit_traces[:15]
        ],
    }
