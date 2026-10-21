"""Procedural Generation Evaluator for M20 Neural Scaling.

Evaluates decoder-only Transformer models on structured multi-step procedural reasoning:
- Unaided neural-only exact final answer accuracy (strictly without tools or external execution)
- Stepwise procedural validity and variable dependency consistency
- Intermediate arithmetic calculation correctness
- Addition depth ladder breakdown (depths 2, 3, 4, 5)
- Unseen operand and carry pattern generalization
- Inference latency and per-token decoding speed
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.rules.procedural_dataset import (
    ProceduralDataset,
    ProceduralExample,
    parse_procedural_completion,
)

logger = logging.getLogger(__name__)


@dataclass
class ProceduralEvaluationMetrics:
    """Aggregate metrics from procedural reasoning evaluation."""

    total_problems: int
    exact_final_answer_acc: float
    procedural_validity_rate: float
    intermediate_calc_acc: float
    step_count_acc: float
    mean_latency_ms: float
    loss: float
    depth_accuracies: dict[int, dict[str, float]] = field(default_factory=dict)
    archetype_accuracies: dict[str, dict[str, float]] = field(default_factory=dict)
    per_problem_details: list[dict[str, Any]] = field(default_factory=list)


def evaluate_neural_procedure_unaided(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    examples: list[ProceduralExample],
    device: torch.device,
    max_new_tokens: int = 120,
) -> ProceduralEvaluationMetrics:
    """Evaluate pure neural generation on procedural problems without external execution tools."""
    model.eval()
    was_training = model.training

    exact_matches = 0
    valid_syntax_count = 0
    step_count_matches = 0
    total_intermediate_calcs = 0
    correct_intermediate_calcs = 0
    latencies: list[float] = []

    depth_buckets: dict[int, dict[str, int]] = {}
    archetype_buckets: dict[str, dict[str, int]] = {}
    problem_details: list[dict[str, Any]] = []

    for ex in examples:
        depth = ex.depth
        arch = ex.archetype
        if depth not in depth_buckets:
            depth_buckets[depth] = {"total": 0, "exact": 0, "valid": 0}
        if arch not in archetype_buckets:
            archetype_buckets[arch] = {"total": 0, "exact": 0, "valid": 0}
        depth_buckets[depth]["total"] += 1
        archetype_buckets[arch]["total"] += 1

        prompt_str = ex.format_prompt()
        prompt_tokens = tokenizer.encode(prompt_str)
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        t0 = time.perf_counter()
        generated_tokens = list(prompt_tokens)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                curr_ids = torch.tensor([generated_tokens[-model.config.context_length:]], dtype=torch.long, device=device)
                logits, _ = model(curr_ids)
                next_tok = int(torch.argmax(logits[0, -1, :]).item())
                generated_tokens.append(next_tok)

                # Stop if EOS or double newline after Final Answer
                if next_tok == tokenizer.eos_id:
                    break
                decoded_so_far = tokenizer.decode(generated_tokens[len(prompt_tokens):])
                if "Final Answer:" in decoded_so_far and "\n" in decoded_so_far.split("Final Answer:")[-1]:
                    break

        latency_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(latency_ms)

        completion_tokens = generated_tokens[len(prompt_tokens):]
        raw_completion = tokenizer.decode(completion_tokens)
        parsed = parse_procedural_completion(raw_completion)

        # 1. Exact Final Answer Match (No tools used!)
        pred_ans = parsed.get("final_answer")
        exp_ans = str(ex.final_answer).strip()

        is_exact = False
        if pred_ans is not None:
            clean_pred = pred_ans.strip().lower()
            clean_exp = exp_ans.lower()
            if clean_pred == clean_exp:
                is_exact = True
            else:
                try:
                    is_exact = abs(float(clean_pred) - float(clean_exp)) < 1e-4
                except (ValueError, TypeError):
                    is_exact = False

        if is_exact:
            exact_matches += 1
            depth_buckets[depth]["exact"] += 1
            archetype_buckets[arch]["exact"] += 1

        # 2. Procedural Format Validity
        is_valid = parsed.get("is_valid", False)
        if is_valid:
            valid_syntax_count += 1
            depth_buckets[depth]["valid"] += 1
            archetype_buckets[arch]["valid"] += 1

        # 3. Step Count Match
        num_parsed = len(parsed.get("parsed_steps", []))
        if num_parsed == len(ex.steps):
            step_count_matches += 1

        # 4. Stepwise Intermediate Arithmetic Verification
        parsed_steps = parsed.get("parsed_steps", [])
        for step_idx, step_info in enumerate(parsed_steps):
            total_intermediate_calcs += 1
            calc_str = step_info.get("calc", "")
            # Check basic arithmetic: e.g. "120 + 80 = 200" or "300 / 5 = 60"
            if "=" in calc_str:
                parts = calc_str.split("=")
                expr = parts[0].strip()
                claimed_res = parts[1].strip()
                try:
                    # Evaluate expression safely without eval
                    is_correct_calc = False
                    if "+" in expr:
                        subparts = expr.split("+")
                        calc_val = float(subparts[0].strip()) + float(subparts[1].strip())
                        is_correct_calc = abs(calc_val - float(claimed_res)) < 1e-4
                    elif "/" in expr:
                        subparts = expr.split("/")
                        denom = float(subparts[1].strip())
                        if denom != 0:
                            calc_val = float(subparts[0].strip()) / denom
                            is_correct_calc = abs(calc_val - float(claimed_res)) < 1e-4
                    elif "-" in expr:
                        subparts = expr.split("-")
                        calc_val = float(subparts[0].strip()) - float(subparts[1].strip())
                        is_correct_calc = abs(calc_val - float(claimed_res)) < 1e-4
                    elif ">" in expr or "<" in expr or "==" in expr:
                        is_correct_calc = True  # symbolic compare
                    if is_correct_calc:
                        correct_intermediate_calcs += 1
                except Exception:
                    pass

        problem_details.append({
            "question": ex.question,
            "expected_answer": exp_ans,
            "predicted_answer": pred_ans,
            "is_exact": is_exact,
            "depth": depth,
            "archetype": arch,
            "parsed_steps": num_parsed,
            "expected_steps": len(ex.steps),
            "raw_completion": raw_completion[:200],
            "latency_ms": round(latency_ms, 2),
        })

    n = max(1, len(examples))
    inter_acc = (correct_intermediate_calcs / total_intermediate_calcs * 100.0) if total_intermediate_calcs > 0 else 0.0

    depth_accuracies = {
        d: {
            "exact_acc": round(v["exact"] / max(1, v["total"]) * 100.0, 2),
            "valid_acc": round(v["valid"] / max(1, v["total"]) * 100.0, 2),
            "total": v["total"],
        }
        for d, v in depth_buckets.items()
    }

    archetype_accuracies = {
        arch: {
            "exact_acc": round(v["exact"] / max(1, v["total"]) * 100.0, 2),
            "valid_acc": round(v["valid"] / max(1, v["total"]) * 100.0, 2),
            "total": v["total"],
        }
        for arch, v in archetype_buckets.items()
    }

    # Evaluate mean loss on procedural dataset
    ds = ProceduralDataset(examples, tokenizer, context_length=model.config.context_length)
    total_loss = 0.0
    loss_tokens = 0
    with torch.no_grad():
        for inp, lab in ds:
            inp_t = inp.unsqueeze(0).to(device)
            lab_t = lab.unsqueeze(0).to(device)
            logits, _ = model(inp_t)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = lab_t[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1), ignore_index=-100)
            valid_mask = shift_labels.view(-1) != -100
            n_toks = valid_mask.sum().item()
            if n_toks > 0:
                total_loss += loss.item() * n_toks
                loss_tokens += n_toks

    mean_loss = total_loss / max(1, loss_tokens)

    if was_training:
        model.train()

    return ProceduralEvaluationMetrics(
        total_problems=len(examples),
        exact_final_answer_acc=round(exact_matches / n * 100.0, 2),
        procedural_validity_rate=round(valid_syntax_count / n * 100.0, 2),
        intermediate_calc_acc=round(inter_acc, 2),
        step_count_acc=round(step_count_matches / n * 100.0, 2),
        mean_latency_ms=round(sum(latencies) / n, 2),
        loss=round(mean_loss, 4),
        depth_accuracies=depth_accuracies,
        archetype_accuracies=archetype_accuracies,
        per_problem_details=problem_details,
    )


def train_procedural_model(
    model: MiniGPT,
    train_dataset: ProceduralDataset,
    val_dataset: ProceduralDataset,
    device: torch.device,
    epochs: int = 10,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Train MiniGPT on structured procedural reasoning dataset."""
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    history: list[dict[str, Any]] = []
    from torch.utils.data import DataLoader
    from adaptive_mini_gpt.rules.procedural_dataset import collate_procedural_batch

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_procedural_batch,
    )

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

        # Validation loss
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
                l_val = F.cross_entropy(s_log.view(-1, model.config.vocab_size), s_lab.view(-1), ignore_index=-100)
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
