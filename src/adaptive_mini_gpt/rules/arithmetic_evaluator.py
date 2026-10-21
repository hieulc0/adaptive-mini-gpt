"""Evaluation and Training Harness for M21/M22 Digit-Level Neural Arithmetic."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.rules.arithmetic_dataset import (
    ArithmeticDataset,
    ArithmeticProblem,
    collate_arithmetic_batch,
    format_problem,
    parse_arithmetic_completion,
)

logger = logging.getLogger(__name__)


@dataclass
class ArithmeticEvaluationMetrics:
    """Detailed arithmetic evaluation metrics across length, carry, digit, and truncation."""

    total_problems: int
    exact_answer_accuracy: float
    digit_level_accuracy: float
    syntax_validity_rate: float
    carry_prediction_accuracy: float
    truncation_rate: float
    mean_latency_ms: float
    loss: float
    accuracy_by_length: dict[int, dict[str, float]] = field(default_factory=dict)
    accuracy_by_carries: dict[int, dict[str, float]] = field(default_factory=dict)
    per_problem_details: list[dict[str, Any]] = field(default_factory=list)


def evaluate_arithmetic_unaided(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    problems: list[ArithmeticProblem],
    arm: str,
    device: torch.device,
    max_new_tokens: int = 250,
) -> ArithmeticEvaluationMetrics:
    """Evaluate unaided neural arithmetic generation without external execution tools."""
    model.eval()
    was_training = model.training

    exact_matches = 0
    valid_syntax_count = 0
    total_digits = 0
    correct_digits = 0
    total_carries_evaluated = 0
    correct_carries_evaluated = 0
    total_truncated = 0
    latencies: list[float] = []

    length_buckets: dict[int, dict[str, int]] = {}
    carry_buckets: dict[int, dict[str, int]] = {}
    problem_details: list[dict[str, Any]] = []

    for prob in problems:
        m_len = prob.max_length
        c_count = min(prob.num_carries, 3)  # bucket 0, 1, 2, 3+

        if m_len not in length_buckets:
            length_buckets[m_len] = {"total": 0, "exact": 0, "valid": 0, "truncated": 0}
        if c_count not in carry_buckets:
            carry_buckets[c_count] = {"total": 0, "exact": 0, "valid": 0, "truncated": 0}

        length_buckets[m_len]["total"] += 1
        carry_buckets[c_count]["total"] += 1

        prompt_str, _ = format_problem(prob, arm)
        prompt_tokens = tokenizer.encode(prompt_str)

        t0 = time.perf_counter()
        generated_tokens = list(prompt_tokens)
        hit_eos = False

        with torch.no_grad():
            for _ in range(max_new_tokens):
                curr_ids = torch.tensor(
                    [generated_tokens[-model.config.context_length :]],
                    dtype=torch.long,
                    device=device,
                )
                logits, _ = model(curr_ids)
                next_tok = int(torch.argmax(logits[0, -1, :]).item())
                generated_tokens.append(next_tok)

                if next_tok == tokenizer.eos_id:
                    hit_eos = True
                    break
                decoded_so_far = tokenizer.decode(generated_tokens[len(prompt_tokens) :])
                if arm in ("standard", "digit_msd", "digit_lsd"):
                    if "\n" in decoded_so_far:
                        break
                elif arm == "scratchpad":
                    if "Answer:" in decoded_so_far and "\n" in decoded_so_far.split("Answer:")[-1]:
                        break
                elif arm == "compact_scratchpad":
                    if "ANS:" in decoded_so_far and (
                        "\n" in decoded_so_far.split("ANS:")[-1] or "<|eos|>" in decoded_so_far
                    ):
                        break

        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        completion_tokens = generated_tokens[len(prompt_tokens) :]
        is_truncated = (len(completion_tokens) >= max_new_tokens and not hit_eos)
        if is_truncated:
            total_truncated += 1
            length_buckets[m_len]["truncated"] += 1
            carry_buckets[c_count]["truncated"] += 1

        raw_completion = tokenizer.decode(completion_tokens)
        parsed = parse_arithmetic_completion(raw_completion, arm)

        pred_ans = parsed.get("predicted_answer")
        exp_ans = str(prob.result)

        # 1. Exact Answer Match (without tools)
        is_exact = (pred_ans is not None and pred_ans == exp_ans)
        if is_exact:
            exact_matches += 1
            length_buckets[m_len]["exact"] += 1
            carry_buckets[c_count]["exact"] += 1

        # 2. Syntax Validity
        if parsed.get("is_valid", False):
            valid_syntax_count += 1
            length_buckets[m_len]["valid"] += 1
            carry_buckets[c_count]["valid"] += 1

        # 3. Digit-Level Accuracy
        if pred_ans is not None:
            rev_pred = pred_ans[::-1]
            rev_exp = exp_ans[::-1]
            compare_len = max(len(rev_pred), len(rev_exp))
            for i in range(compare_len):
                total_digits += 1
                d_pred = rev_pred[i] if i < len(rev_pred) else None
                d_exp = rev_exp[i] if i < len(rev_exp) else None
                if d_pred == d_exp:
                    correct_digits += 1
        else:
            total_digits += len(exp_ans)

        # 4. Carry Accuracy (for scratchpad and compact_scratchpad)
        if arm in ("scratchpad", "compact_scratchpad"):
            gt_carries = [col.carry_out for col in prob.columns]
            pred_carries = parsed.get("predicted_carries", [])
            for idx, gt_c in enumerate(gt_carries):
                total_carries_evaluated += 1
                if idx < len(pred_carries) and pred_carries[idx] == gt_c:
                    correct_carries_evaluated += 1

        problem_details.append({
            "a": prob.a,
            "b": prob.b,
            "expected_answer": exp_ans,
            "predicted_answer": pred_ans,
            "is_exact": is_exact,
            "is_truncated": is_truncated,
            "length": m_len,
            "num_carries": prob.num_carries,
            "raw_completion": raw_completion[:120],
            "latency_ms": round(latency_ms, 2),
        })

    n = max(1, len(problems))
    exact_acc = round(exact_matches / n * 100.0, 2)
    valid_acc = round(valid_syntax_count / n * 100.0, 2)
    digit_acc = round(correct_digits / max(1, total_digits) * 100.0, 2)
    carry_acc = (
        round(correct_carries_evaluated / max(1, total_carries_evaluated) * 100.0, 2)
        if total_carries_evaluated > 0
        else 0.0
    )
    trunc_rate = round(total_truncated / n * 100.0, 2)

    accuracy_by_length = {
        l: {
            "exact_acc": round(v["exact"] / max(1, v["total"]) * 100.0, 2),
            "valid_acc": round(v["valid"] / max(1, v["total"]) * 100.0, 2),
            "trunc_acc": round(v["truncated"] / max(1, v["total"]) * 100.0, 2),
            "total": v["total"],
        }
        for l, v in sorted(length_buckets.items())
    }

    accuracy_by_carries = {
        c: {
            "exact_acc": round(v["exact"] / max(1, v["total"]) * 100.0, 2),
            "valid_acc": round(v["valid"] / max(1, v["total"]) * 100.0, 2),
            "trunc_acc": round(v["truncated"] / max(1, v["total"]) * 100.0, 2),
            "total": v["total"],
        }
        for c, v in sorted(carry_buckets.items())
    }

    # Evaluate loss
    ds = ArithmeticDataset(problems, tokenizer, arm=arm, context_length=model.config.context_length)
    total_loss = 0.0
    loss_tokens = 0
    with torch.no_grad():
        for inp, lab in ds:
            inp_t = inp.unsqueeze(0).to(device)
            lab_t = lab.unsqueeze(0).to(device)
            logits, _ = model(inp_t)
            s_log = logits[..., :-1, :].contiguous()
            s_lab = lab_t[..., 1:].contiguous()
            loss = F.cross_entropy(
                s_log.view(-1, model.config.vocab_size), s_lab.view(-1), ignore_index=-100
            )
            valid_mask = s_lab.view(-1) != -100
            n_toks = valid_mask.sum().item()
            if n_toks > 0:
                total_loss += loss.item() * n_toks
                loss_tokens += n_toks

    mean_loss = total_loss / max(1, loss_tokens)

    if was_training:
        model.train()

    return ArithmeticEvaluationMetrics(
        total_problems=len(problems),
        exact_answer_accuracy=exact_acc,
        digit_level_accuracy=digit_acc,
        syntax_validity_rate=valid_acc,
        carry_prediction_accuracy=carry_acc,
        truncation_rate=trunc_rate,
        mean_latency_ms=round(sum(latencies) / n, 2),
        loss=round(mean_loss, 4),
        accuracy_by_length=accuracy_by_length,
        accuracy_by_carries=accuracy_by_carries,
        per_problem_details=problem_details,
    )


def train_arithmetic_model(
    model: MiniGPT,
    train_dataset: ArithmeticDataset,
    val_dataset: ArithmeticDataset,
    device: torch.device,
    epochs: int = 10,
    lr: float = 5e-4,
    weight_decay: float = 0.01,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Train MiniGPT on arithmetic representation dataset with AdamW."""
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_arithmetic_batch,
    )

    history: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    total_tokens_trained = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_tokens = 0
        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(input_ids)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, model.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            valid_mask = shift_labels != -100
            n_toks = int(valid_mask.sum().item())
            epoch_loss += loss.item() * n_toks
            epoch_tokens += n_toks

        total_tokens_trained += epoch_tokens
        avg_train_loss = epoch_loss / max(1, epoch_tokens)

        # Fast validation check
        val_loss = 0.0
        val_tokens = 0
        model.eval()
        with torch.no_grad():
            for inp, lab in val_dataset:
                inp_t = inp.unsqueeze(0).to(device)
                lab_t = lab.unsqueeze(0).to(device)
                l_out, _ = model(inp_t)
                s_log = l_out[..., :-1, :].contiguous()
                s_lab = lab_t[..., 1:].contiguous()
                l_val = F.cross_entropy(
                    s_log.view(-1, model.config.vocab_size), s_lab.view(-1), ignore_index=-100
                )
                v_toks = int((s_lab != -100).sum().item())
                val_loss += l_val.item() * v_toks
                val_tokens += v_toks
        model.train()

        avg_val_loss = val_loss / max(1, val_tokens)
        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "tokens": epoch_tokens,
        })

    total_time_s = time.perf_counter() - t_start

    return {
        "final_train_loss": round(avg_train_loss, 4),
        "final_val_loss": round(avg_val_loss, 4),
        "total_tokens_trained": total_tokens_trained,
        "training_time_s": round(total_time_s, 2),
        "history": history,
    }
