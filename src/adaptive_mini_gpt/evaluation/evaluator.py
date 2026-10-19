"""Evaluation, exact-match skill verification, and generation utilities."""

from __future__ import annotations

import math
from typing import Any

import torch

from adaptive_mini_gpt.data.dataset import TokenDataset, is_holdout_arithmetic_pair
from adaptive_mini_gpt.data.tokenizer import CharacterTokenizer
from adaptive_mini_gpt.model.transformer import MiniGPT


def has_addition_carry(a: int, b: int) -> bool:
    """Return True if adding integers a and b produces a decimal carry at any column."""
    carry = 0
    while a > 0 or b > 0:
        carry = (a % 10 + b % 10 + carry) // 10
        if carry > 0:
            return True
        a //= 10
        b //= 10
    return False


def has_subtraction_borrow(a: int, b: int) -> bool:
    """Return True if subtracting b from a (a >= b) requires a borrow at any decimal column."""
    borrow = 0
    while a > 0 or b > 0:
        da = a % 10 - borrow
        db = b % 10
        if da < db:
            borrow = 1
            return True
        borrow = 0
        a //= 10
        b //= 10
    return False


@torch.no_grad()
def evaluate_dataset(
    model: MiniGPT,
    dataset: TokenDataset,
    batch_size: int,
    device: torch.device | str = "cpu",
    max_batches: int = 50,
    task_id: int | None = None,
    bypass_moe: bool = False,
) -> dict[str, float]:
    """Evaluate model loss, perplexity, and bits-per-character without redundant wrapping."""
    model.eval()
    total_loss = 0.0
    batches_run = 0

    tokens_per_batch = batch_size * model.config.context_length
    natural_batches = max(1, math.ceil(len(dataset) / tokens_per_batch))
    effective_batches = min(max_batches, natural_batches)

    for i in range(effective_batches):
        start_idx = i * tokens_per_batch
        x, y, _ = dataset.get_sequential_batch(
            start_idx=start_idx,
            batch_size=batch_size,
            device=device,
        )
        _, loss = model(x, y, task_id=task_id, bypass_moe=bypass_moe)
        if loss is not None and model.last_aux_info.get("ce_loss", 0.0) > 0.0:
            total_loss += model.last_aux_info["ce_loss"]
            batches_run += 1
        elif loss is not None:
            total_loss += loss.item()
            batches_run += 1

    mean_loss = total_loss / batches_run if batches_run > 0 else 0.0
    perplexity = math.exp(min(mean_loss, 20.0))
    bpc = mean_loss / math.log(2)

    return {
        "loss": mean_loss,
        "perplexity": perplexity,
        "bits_per_character": bpc,
        "batches_evaluated": batches_run,
    }


@torch.no_grad()
def evaluate_arithmetic_accuracy(
    model: MiniGPT,
    tokenizer: CharacterTokenizer,
    num_problems: int = 100,
    seed: int = 999,
    device: torch.device | str = "cpu",
    task_id: int | None = None,
    ood_extrapolation: bool = False,
    bypass_moe: bool = False,
) -> dict[str, Any]:
    """Verify exact-match greedy accuracy on strictly disjoint held-out arithmetic problems."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    correct = 0
    tested = 0
    op_names = {0: "ADD", 1: "SUB", 2: "PAR"}
    op_totals = {"ADD": 0, "SUB": 0, "PAR": 0}
    op_correct = {"ADD": 0, "SUB": 0, "PAR": 0}
    samples: list[dict[str, str]] = []

    while tested < num_problems:
        op_type = int(torch.randint(0, 3, (1,), generator=g).item())
        if not ood_extrapolation:
            is_single_digit = int(torch.randint(0, 2, (1,), generator=g).item()) == 1
            upper = 10 if is_single_digit else 100
            a = int(torch.randint(0, upper, (1,), generator=g).item())
            b = int(torch.randint(0, upper, (1,), generator=g).item())
        else:
            a = int(torch.randint(100, 200, (1,), generator=g).item())
            b = int(torch.randint(100, 200, (1,), generator=g).item())

        if op_type == 1:
            a, b = max(a, b), min(a, b)
        elif op_type == 2:
            b = 0

        if not ood_extrapolation and not is_holdout_arithmetic_pair(op_type, a, b):
            continue

        if op_type == 0:
            prompt = f"ADD {a}+{b}="
            expected = str(a + b)
        elif op_type == 1:
            prompt = f"SUB {a}-{b}="
            expected = str(a - b)
        else:
            prompt = f"PAR {a}="
            expected = "EVEN" if (a % 2 == 0) else "ODD"

        res = generate_sample(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=6,
            temperature=0.0,
            stop_on_newline=True,
            device=device,
            task_id=task_id,
            bypass_moe=bypass_moe,
        )
        pred = res["completion"].strip()
        op_key = op_names[op_type]
        op_totals[op_key] += 1
        if pred == expected:
            correct += 1
            op_correct[op_key] += 1
        if len(samples) < 5:
            samples.append({"prompt": prompt, "expected": expected, "predicted": pred})
        tested += 1

    by_op = {
        k: round(op_correct[k] / max(1, op_totals[k]), 4) for k in ("ADD", "SUB", "PAR")
    }

    return {
        "exact_match_accuracy": round(correct / max(1, num_problems), 4),
        "accuracy_by_op": by_op,
        "correct": correct,
        "total_problems": num_problems,
        "ood_extrapolation": ood_extrapolation,
        "sample_predictions": samples,
    }


@torch.no_grad()
def evaluate_arithmetic_detailed(
    model: MiniGPT,
    tokenizer: CharacterTokenizer,
    num_per_category: int = 25,
    seed: int = 999,
    device: torch.device | str = "cpu",
    task_id: int | None = None,
    bypass_moe: bool = False,
) -> dict[str, Any]:
    """Evaluate fine-grained arithmetic categories: carry/borrow, parity, and OOD."""
    model.eval()
    g = torch.Generator().manual_seed(seed)

    categories = [
        "ADD_NO_CARRY",
        "ADD_CARRY",
        "SUB_NO_BORROW",
        "SUB_BORROW",
        "PARITY",
        "OOD_EXTRAPOLATION",
    ]
    cat_correct = {cat: 0 for cat in categories}
    cat_total = {cat: 0 for cat in categories}
    cat_samples: dict[str, list[dict[str, str]]] = {cat: [] for cat in categories}

    for cat in categories:
        tested_in_cat = 0
        attempts = 0
        while tested_in_cat < num_per_category and attempts < 2000:
            attempts += 1
            if cat == "ADD_NO_CARRY":
                a = int(torch.randint(0, 100, (1,), generator=g).item())
                b = int(torch.randint(0, 100, (1,), generator=g).item())
                if has_addition_carry(a, b) or not is_holdout_arithmetic_pair(0, a, b):
                    continue
                prompt = f"ADD {a}+{b}="
                expected = str(a + b)
            elif cat == "ADD_CARRY":
                a = int(torch.randint(0, 100, (1,), generator=g).item())
                b = int(torch.randint(0, 100, (1,), generator=g).item())
                if not has_addition_carry(a, b) or not is_holdout_arithmetic_pair(0, a, b):
                    continue
                prompt = f"ADD {a}+{b}="
                expected = str(a + b)
            elif cat == "SUB_NO_BORROW":
                a = int(torch.randint(0, 100, (1,), generator=g).item())
                b = int(torch.randint(0, 100, (1,), generator=g).item())
                a, b = max(a, b), min(a, b)
                if has_subtraction_borrow(a, b) or not is_holdout_arithmetic_pair(1, a, b):
                    continue
                prompt = f"SUB {a}-{b}="
                expected = str(a - b)
            elif cat == "SUB_BORROW":
                a = int(torch.randint(0, 100, (1,), generator=g).item())
                b = int(torch.randint(0, 100, (1,), generator=g).item())
                a, b = max(a, b), min(a, b)
                if not has_subtraction_borrow(a, b) or not is_holdout_arithmetic_pair(1, a, b):
                    continue
                prompt = f"SUB {a}-{b}="
                expected = str(a - b)
            elif cat == "PARITY":
                a = int(torch.randint(0, 100, (1,), generator=g).item())
                if not is_holdout_arithmetic_pair(2, a, 0):
                    continue
                prompt = f"PAR {a}="
                expected = "EVEN" if (a % 2 == 0) else "ODD"
            else:  # OOD_EXTRAPOLATION (3-digit operands 100..499)
                op_t = int(torch.randint(0, 3, (1,), generator=g).item())
                a = int(torch.randint(100, 500, (1,), generator=g).item())
                b = int(torch.randint(100, 500, (1,), generator=g).item())
                if op_t == 0:
                    prompt = f"ADD {a}+{b}="
                    expected = str(a + b)
                elif op_t == 1:
                    a, b = max(a, b), min(a, b)
                    prompt = f"SUB {a}-{b}="
                    expected = str(a - b)
                else:
                    prompt = f"PAR {a}="
                    expected = "EVEN" if (a % 2 == 0) else "ODD"

            res = generate_sample(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=8,
                temperature=0.0,
                stop_on_newline=True,
                device=device,
                task_id=task_id,
                bypass_moe=bypass_moe,
            )
            pred = res["completion"].strip()
            cat_total[cat] += 1
            if pred == expected:
                cat_correct[cat] += 1
            if len(cat_samples[cat]) < 3:
                cat_samples[cat].append(
                    {"prompt": prompt, "expected": expected, "predicted": pred}
                )
            tested_in_cat += 1

    by_category = {
        cat: round(cat_correct[cat] / max(1, cat_total[cat]), 4) for cat in categories
    }
    in_domain_cats = ["ADD_NO_CARRY", "ADD_CARRY", "SUB_NO_BORROW", "SUB_BORROW", "PARITY"]
    in_domain_corr = sum(cat_correct[c] for c in in_domain_cats)
    in_domain_tot = sum(cat_total[c] for c in in_domain_cats)
    in_domain_acc = round(in_domain_corr / max(1, in_domain_tot), 4)
    ood_acc = by_category.get("OOD_EXTRAPOLATION", 0.0)
    total_corr = sum(cat_correct.values())
    total_all = sum(cat_total.values())

    return {
        "in_domain_accuracy": in_domain_acc,
        "ood_accuracy": ood_acc,
        "overall_accuracy": round(total_corr / max(1, total_all), 4),
        "accuracy_by_category": by_category,
        "category_totals": cat_total,
        "category_correct": cat_correct,
        "samples_by_category": cat_samples,
    }


@torch.no_grad()
def evaluate_text_retention_suite(
    model: MiniGPT,
    tokenizer: CharacterTokenizer,
    device: torch.device | str = "cpu",
    max_batches: int = 20,
) -> dict[str, Any]:
    """Evaluate language retention across clean Shakespeare, numerical prose, and mixed prose."""
    text_clean = (
        "From fairest creatures we desire increase, That thereby beauty's rose might never die.\n"
        "When forty winters shall besiege thy brow, And dig deep trenches in thy beauty's field.\n"
        "Look in thy glass, and tell the face thou viewest Now is the time that face another.\n"
        "Unthrifty loveliness, why dost thou spend Upon thyself thy beauty's legacy?\n"
    ) * 10
    tok_clean = torch.tensor(tokenizer.encode(text_clean), dtype=torch.long)
    ds_clean = TokenDataset(tok_clean, context_length=min(model.config.context_length, 64))

    text_numerical = (
        "In the year 1594, Shakespeare joined the company of actors at the theatre in London.\n"
        "Act 2, Scene 4: Enter King Henry with 10 soldiers and 3 messengers at 12 o'clock.\n"
        "The company performed 154 sonnets across 3 theatres before 2000 royal patrons.\n"
        "In 1603, King James ascended the throne, granting royal patronage to 9 actors.\n"
        "The Globe Theatre, erected in 1599, burned down on 29 June 1613 during Henry VIII.\n"
        "By 1623, John Heminges and Henry Condell published the First Folio containing 36 plays.\n"
        "At age 52, Shakespeare passed away on 23 April 1616 in Stratford-upon-Avon.\n"
    ) * 8
    tok_num = torch.tensor(tokenizer.encode(text_numerical), dtype=torch.long)
    ds_num = TokenDataset(tok_num, context_length=min(model.config.context_length, 64))

    text_mixed = (
        "The steward counted 14 sacks of wheat and 25 casks of wine. ADD 14+25=39\n"
        "The royal guard numbered 85 spears and issued 32 on patrol. SUB 85-32=53\n"
        "The garrison recorded 42 heralds at the gate. PAR 42=EVEN\n"
        "The baker baked 7 loaves at dawn and 9 at noon. ADD 7+9=16\n"
        "From 48 gold crowns, the merchant paid 19 in taxes. SUB 48-19=29\n"
    ) * 10
    tok_mix = torch.tensor(tokenizer.encode(text_mixed), dtype=torch.long)
    ds_mix = TokenDataset(tok_mix, context_length=min(model.config.context_length, 64))

    res_clean_auto = evaluate_dataset(
        model, ds_clean, batch_size=8, device=device, max_batches=max_batches, bypass_moe=False
    )
    res_clean_byp = evaluate_dataset(
        model, ds_clean, batch_size=8, device=device, max_batches=max_batches, bypass_moe=True
    )

    res_num_auto = evaluate_dataset(
        model, ds_num, batch_size=8, device=device, max_batches=max_batches, bypass_moe=False
    )
    res_num_byp = evaluate_dataset(
        model, ds_num, batch_size=8, device=device, max_batches=max_batches, bypass_moe=True
    )

    res_mix_auto = evaluate_dataset(
        model, ds_mix, batch_size=8, device=device, max_batches=max_batches, bypass_moe=False
    )
    res_mix_byp = evaluate_dataset(
        model, ds_mix, batch_size=8, device=device, max_batches=max_batches, bypass_moe=True
    )

    return {
        "clean_prose": {
            "loss_auto": round(res_clean_auto["loss"], 4),
            "bpc_auto": round(res_clean_auto["bits_per_character"], 4),
            "loss_bypassed": round(res_clean_byp["loss"], 4),
            "bpc_bypassed": round(res_clean_byp["bits_per_character"], 4),
            "delta_loss": round(res_clean_auto["loss"] - res_clean_byp["loss"], 6),
        },
        "numerical_prose": {
            "loss_auto": round(res_num_auto["loss"], 4),
            "bpc_auto": round(res_num_auto["bits_per_character"], 4),
            "loss_bypassed": round(res_num_byp["loss"], 4),
            "bpc_bypassed": round(res_num_byp["bits_per_character"], 4),
            "delta_loss": round(res_num_auto["loss"] - res_num_byp["loss"], 6),
        },
        "mixed_prose_math": {
            "loss_auto": round(res_mix_auto["loss"], 4),
            "bpc_auto": round(res_mix_auto["bits_per_character"], 4),
            "loss_bypassed": round(res_mix_byp["loss"], 4),
            "bpc_bypassed": round(res_mix_byp["bits_per_character"], 4),
            "delta_loss": round(res_mix_auto["loss"] - res_mix_byp["loss"], 6),
        },
    }


@torch.no_grad()
def generate_sample(
    model: MiniGPT,
    tokenizer: CharacterTokenizer,
    prompt: str = "The ",
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    stop_on_newline: bool = False,
    device: torch.device | str = "cpu",
    task_id: int | None = None,
    bypass_moe: bool = False,
) -> dict[str, Any]:
    """Generate structured text completion from a prompt using the model and tokenizer."""
    model.eval()
    encoded = tokenizer.encode(prompt)
    if not encoded:
        encoded = [0]
    idx = torch.tensor([encoded], dtype=torch.long, device=device)
    stop_id = tokenizer.stoi.get("\n") if stop_on_newline else None

    out = model.generate(
        idx=idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_id=stop_id,
        task_id=task_id,
        bypass_moe=bypass_moe,
    )

    out_tokens = out[0].tolist()
    prompt_len = len(encoded)
    completion_tokens = out_tokens[prompt_len:]

    return {
        "prompt": prompt,
        "prompt_tokens": prompt_len,
        "total_tokens": len(out_tokens),
        "generated_text": tokenizer.decode(out_tokens),
        "completion": tokenizer.decode(completion_tokens),
    }
