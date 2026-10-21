"""Phase 1: M21 Failure Audit Script.

Analyzes token lengths, generation limits, and failure modes across arithmetic problems
to determine the exact cause of 0% syntax validity on 3-carry problems in M21.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.rules.arithmetic_dataset import (
    ArithmeticProblem,
    format_problem,
    generate_controlled_addition_dataset,
)
from adaptive_mini_gpt.training.checkpoint import CheckpointManager


def audit_m21_dataset_token_lengths(
    tokenizer: BPETokenizer,
    problems: list[ArithmeticProblem],
    max_new_tokens: int = 120,
    context_length: int = 256,
) -> dict[str, Any]:
    """Audit token lengths and classify truncation risks for arithmetic problems."""
    audit_results: list[dict[str, Any]] = []
    truncation_count = 0
    context_overflow_count = 0

    carry_breakdown: dict[int, dict[str, int]] = {}
    length_breakdown: dict[int, dict[str, int]] = {}

    for prob in problems:
        c_bucket = min(prob.num_carries, 3)
        l_bucket = prob.max_length

        if c_bucket not in carry_breakdown:
            carry_breakdown[c_bucket] = {"total": 0, "truncated_by_gen_limit": 0, "overflow_context": 0}
        if l_bucket not in length_breakdown:
            length_breakdown[l_bucket] = {"total": 0, "truncated_by_gen_limit": 0, "overflow_context": 0}

        carry_breakdown[c_bucket]["total"] += 1
        length_breakdown[l_bucket]["total"] += 1

        prompt_str, target_str = format_problem(prob, "scratchpad")
        p_toks = len(tokenizer.encode(prompt_str))
        t_toks = len(tokenizer.encode(target_str))
        tot_toks = p_toks + t_toks

        truncated_by_gen = t_toks > max_new_tokens
        overflows_ctx = tot_toks > context_length

        if truncated_by_gen:
            truncation_count += 1
            carry_breakdown[c_bucket]["truncated_by_gen_limit"] += 1
            length_breakdown[l_bucket]["truncated_by_gen_limit"] += 1

        if overflows_ctx:
            context_overflow_count += 1
            carry_breakdown[c_bucket]["overflow_context"] += 1
            length_breakdown[l_bucket]["overflow_context"] += 1

        audit_results.append({
            "a": prob.a,
            "b": prob.b,
            "result": prob.result,
            "length": l_bucket,
            "carries": prob.num_carries,
            "prompt_tokens": p_toks,
            "target_tokens": t_toks,
            "total_tokens": tot_toks,
            "truncated_by_gen": truncated_by_gen,
            "overflows_context": overflows_ctx,
        })

    return {
        "total_problems": len(problems),
        "max_new_tokens_threshold": max_new_tokens,
        "context_length_threshold": context_length,
        "total_truncated_by_gen": truncation_count,
        "total_overflows_context": context_overflow_count,
        "truncation_rate": round(truncation_count / len(problems) * 100, 2) if problems else 0.0,
        "context_overflow_rate": round(context_overflow_count / len(problems) * 100, 2) if problems else 0.0,
        "carry_breakdown": carry_breakdown,
        "length_breakdown": length_breakdown,
    }


def main() -> None:
    ckpt_path = Path("runs/kaggle/m18-checkpoints/seed_42/base_pretrain/best.pt")
    if not ckpt_path.exists():
        print(f"Checkpoint {ckpt_path} not found.")
        return

    raw = CheckpointManager.load_raw_checkpoint(ckpt_path)
    tokenizer = CheckpointManager.load_tokenizer(raw)

    print("Generating M21 dataset partitions...")
    train_probs = generate_controlled_addition_dataset(seed=1021, num_samples=1200, lengths=(2, 3), split="train")
    test_in_dist = generate_controlled_addition_dataset(seed=3021, num_samples=200, lengths=(2, 3), split="test")
    test_extrap = generate_controlled_addition_dataset(seed=4021, num_samples=100, lengths=(4, 5), split="test_extrapolation")

    audit_in_dist = audit_m21_dataset_token_lengths(tokenizer, test_in_dist, max_new_tokens=120)
    audit_extrap = audit_m21_dataset_token_lengths(tokenizer, test_extrap, max_new_tokens=120)

    print("\n--- In-Distribution (2-3 Digits) Audit ---")
    print(json.dumps(audit_in_dist, indent=2))

    print("\n--- Extrapolation (4-5 Digits) Audit ---")
    print(json.dumps(audit_extrap, indent=2))


if __name__ == "__main__":
    main()
