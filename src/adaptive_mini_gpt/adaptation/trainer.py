"""Fast Adaptation Trainer and Learning Diagnostics (M30 Phase 1 & 3).

Supports:
  - Phase 1 Diagnostics: Gradient norm tracking, loss reduction, semantic error analysis,
    and replay gradient competition measurement.
  - 4 Adaptation Regimes:
      1. Full fine-tuning without replay (forgetting diagnostic).
      2. Full fine-tuning with multi-domain replay.
      3. Frozen backbone with rank-4 LoRA adapters.
      4. Frozen backbone with rank-8 LoRA adapters.
  - Evaluation of adapters with adapter active (enabled=True) vs disabled (enabled=False).
"""

from __future__ import annotations

import copy
import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from adaptive_mini_gpt.data.language_corpus import (
    normalize_text,
    parse_language_completion,
)
from adaptive_mini_gpt.data.language_data import (
    BenchmarkSample,
    DomainType,
    LanguageComprehensionDataset,
    collate_language_batch,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.lora import (
    freeze_base_parameters,
    get_lora_parameter_stats,
    inject_lora_adapters,
    toggle_lora_adapters,
)
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.rules.reliable_transition import (
    ElementaryTransition,
    ElementaryTransitionDataset,
    audit_elementary_transition_space,
    collate_elementary_batch,
)

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticResult:
    """Detailed diagnostics of neural learning behavior on a specific gap."""

    initial_train_loss: float
    final_train_loss: float
    train_loss_reduction_pct: float
    initial_test_loss: float
    final_test_loss: float
    test_loss_reduction_pct: float
    mean_practice_grad_norm: float
    mean_replay_grad_norm: float
    grad_norm_ratio_practice_to_replay: float
    semantic_correct_count: int
    format_failure_count: int
    total_evaluated: int


@dataclass
class AdaptationRunResult:
    """Traceable results of an adaptation experiment."""

    regime_name: str
    scale: int
    trainable_params: int
    total_params: int
    trainable_ratio_pct: float
    initial_train_loss: float
    final_train_loss: float
    id_test_accuracy: float
    ood_test_accuracy: float
    retention_carry_accuracy: float
    adapter_disabled_id_accuracy: float | None = None
    adapter_disabled_retention_carry: float | None = None
    base_weights_preserved: bool = True
    diagnostics: DiagnosticResult | None = None


def compute_dataset_loss(
    model: MiniGPT,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute mean cross-entropy loss over a DataLoader with causal label shift."""
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            max_ctx = getattr(model.config, "context_length", 256)
            out = model(x[:, -max_ctx:])
            logits = out[0] if isinstance(out, tuple) else out
            loss = nn.functional.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                y[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
            total_loss += float(loss.item())
            steps += 1
    return round(total_loss / max(1, steps), 4)


def evaluate_samples_accuracy(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    samples: list[BenchmarkSample],
    domain: DomainType,
    device: torch.device,
    max_new_tokens: int = 16,
) -> tuple[float, int, int]:
    """Evaluate exact match accuracy, returning (accuracy, correct_count, total)."""
    model.eval()
    correct = 0
    total = len(samples)
    if total == 0:
        return 0.0, 0, 0

    max_ctx = getattr(model.config, "context_length", 256)
    with torch.no_grad():
        for s in samples:
            prompt = s.prompt
            enc = tokenizer.encode(prompt)
            if len(enc) > max_ctx - max_new_tokens:
                enc = enc[-(max_ctx - max_new_tokens):]
            input_ids = torch.tensor([enc], dtype=torch.long, device=device)

            for _ in range(max_new_tokens):
                out = model(input_ids[:, -max_ctx:])
                logits = out[0] if isinstance(out, tuple) else out
                next_id = torch.argmax(logits[0, -1, :]).item()
                input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
                decoded_partial = tokenizer.decode(input_ids[0].tolist())
                eos_id = getattr(tokenizer, "eos_id", -1)
                if "\n" in decoded_partial[len(prompt):] or next_id == eos_id:
                    break

            full_decoded = tokenizer.decode(input_ids[0].tolist())
            completion = full_decoded[len(prompt):].strip()

            if domain == DomainType.LANGUAGE:
                pred, _, _ = parse_language_completion(completion)
                if normalize_text(pred) == normalize_text(s.expected_answer):
                    correct += 1
            else:
                pred = completion.split("\n")[0].strip()
                if normalize_text(pred) == normalize_text(s.expected_answer):
                    correct += 1

    return round(correct / total, 4), correct, total


def hash_base_parameters(model: MiniGPT) -> str:
    """Compute deterministic SHA-256 hash of all non-LoRA parameters."""
    hasher = hashlib.sha256()
    for name, param in sorted(model.named_parameters()):
        if "lora_A" not in name and "lora_B" not in name:
            hasher.update(name.encode("utf-8"))
            hasher.update(param.data.cpu().numpy().tobytes())
    return hasher.hexdigest()[:16]


def run_fast_adaptation_regime(
    regime: str,
    base_model: MiniGPT,
    tokenizer: BPETokenizer,
    practice_samples: list[BenchmarkSample],
    id_test_samples: list[BenchmarkSample],
    ood_test_samples: list[BenchmarkSample],
    domain: DomainType,
    arith_transitions: list[ElementaryTransition],
    logic_replay_samples: list[BenchmarkSample],
    device: torch.device,
    epochs: int = 15,
    lr: float = 3e-4,
    batch_size: int = 8,
    lora_r: int = 4,
    lora_alpha: float = 8.0,
) -> AdaptationRunResult:
    """Execute a single controlled adaptation regime on practice samples."""
    pad_id = getattr(tokenizer, "pad_id", 0)
    max_ctx = getattr(base_model.config, "context_length", 256)

    # 1. Prepare Model for Regime
    model = copy.deepcopy(base_model).to(device)
    base_hash_before = hash_base_parameters(model)
    is_lora = regime in ("lora_r4", "lora_r8")

    if is_lora:
        r = 8 if regime == "lora_r8" else 4
        model, _ = inject_lora_adapters(
            model=model,
            r=r,
            lora_alpha=lora_alpha,
            target_modules=["c_attn", "c_proj"],
        )
        freeze_base_parameters(model)
        stats = get_lora_parameter_stats(model)
        # Trainable parameters are strictly LoRA parameters
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=0.01,
        )
    else:
        # Full fine-tuning
        stats = {
            "trainable_params": sum(p.numel() for p in model.parameters()),
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_ratio_pct": 100.0,
        }
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # 2. Prepare DataLoaders
    ds_practice = LanguageComprehensionDataset(
        samples=practice_samples,
        tokenizer=tokenizer,
        context_length=max_ctx,
        ignore_index=-100,
    )
    loader_practice = DataLoader(
        ds_practice,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_language_batch(b, pad_token_id=pad_id, ignore_index=-100),
    )

    ds_id_test = LanguageComprehensionDataset(
        samples=id_test_samples,
        tokenizer=tokenizer,
        context_length=max_ctx,
        ignore_index=-100,
    )
    loader_id_test = DataLoader(
        ds_id_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_language_batch(b, pad_token_id=pad_id, ignore_index=-100),
    )

    # Replay Loaders (only used in full_ft_with_replay)
    has_replay = regime == "full_ft_with_replay"
    if has_replay:
        ds_arith = ElementaryTransitionDataset(
            transitions=arith_transitions,
            tokenizer=tokenizer,
            format_type="canonical_state",
            context_length=max_ctx,
        )
        loader_arith = DataLoader(
            ds_arith,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_elementary_batch(b, pad_token_id=pad_id),
        )

        ds_logic = LanguageComprehensionDataset(
            samples=logic_replay_samples,
            tokenizer=tokenizer,
            context_length=max_ctx,
            ignore_index=-100,
        )
        loader_logic = DataLoader(
            ds_logic,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_language_batch(b, pad_token_id=pad_id, ignore_index=-100),
        )

    # Initial loss measurement
    init_train_loss = compute_dataset_loss(model, loader_practice, device)
    init_test_loss = compute_dataset_loss(model, loader_id_test, device)

    # 3. Training Loop with Diagnostics
    practice_grad_norms: list[float] = []
    replay_grad_norms: list[float] = []

    model.train()
    for ep in range(epochs):
        if has_replay:
            iter_p = iter(loader_practice)
            iter_a = iter(loader_arith)
            iter_l = iter(loader_logic)
            steps_per_epoch = max(len(loader_practice), len(loader_arith), len(loader_logic))

            for _ in range(steps_per_epoch):
                # Practice step
                try:
                    x_p, y_p = next(iter_p)
                except StopIteration:
                    iter_p = iter(loader_practice)
                    x_p, y_p = next(iter_p)

                x_p, y_p = x_p.to(device), y_p.to(device)
                optimizer.zero_grad()
                out_p = model(x_p)
                logits_p = out_p[0] if isinstance(out_p, tuple) else out_p
                loss_p = nn.functional.cross_entropy(
                    logits_p[:, :-1, :].contiguous().view(-1, logits_p.size(-1)),
                    y_p[:, 1:].contiguous().view(-1),
                    ignore_index=-100,
                )
                loss_p.backward()
                p_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
                practice_grad_norms.append(p_norm)
                optimizer.step()

                # Arithmetic replay step
                try:
                    x_a, y_a = next(iter_a)
                except StopIteration:
                    iter_a = iter(loader_arith)
                    x_a, y_a = next(iter_a)

                x_a, y_a = x_a.to(device), y_a.to(device)
                optimizer.zero_grad()
                out_a = model(x_a)
                logits_a = out_a[0] if isinstance(out_a, tuple) else out_a
                loss_a = nn.functional.cross_entropy(
                    logits_a[:, :-1, :].contiguous().view(-1, logits_a.size(-1)),
                    y_a[:, 1:].contiguous().view(-1),
                    ignore_index=-100,
                )
                loss_a.backward()
                a_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
                replay_grad_norms.append(a_norm)
                optimizer.step()

                # Logic replay step
                try:
                    x_l, y_l = next(iter_l)
                except StopIteration:
                    iter_l = iter(loader_logic)
                    x_l, y_l = next(iter_l)

                x_l, y_l = x_l.to(device), y_l.to(device)
                optimizer.zero_grad()
                out_l = model(x_l)
                logits_l = out_l[0] if isinstance(out_l, tuple) else out_l
                loss_l = nn.functional.cross_entropy(
                    logits_l[:, :-1, :].contiguous().view(-1, logits_l.size(-1)),
                    y_l[:, 1:].contiguous().view(-1),
                    ignore_index=-100,
                )
                loss_l.backward()
                l_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
                replay_grad_norms.append(l_norm)
                optimizer.step()
        else:
            # Dedicated practice steps
            for x_b, y_b in loader_practice:
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                out = model(x_b)
                logits = out[0] if isinstance(out, tuple) else out
                loss = nn.functional.cross_entropy(
                    logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                    y_b[:, 1:].contiguous().view(-1),
                    ignore_index=-100,
                )
                loss.backward()
                g_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
                practice_grad_norms.append(g_norm)
                optimizer.step()

    # Final loss measurement
    final_train_loss = compute_dataset_loss(model, loader_practice, device)
    final_test_loss = compute_dataset_loss(model, loader_id_test, device)

    # 4. Evaluation on Fresh Held-Out Test Sets
    id_acc, id_correct, id_tot = evaluate_samples_accuracy(
        model=model,
        tokenizer=tokenizer,
        samples=id_test_samples,
        domain=domain,
        device=device,
    )
    ood_acc, _, _ = evaluate_samples_accuracy(
        model=model,
        tokenizer=tokenizer,
        samples=ood_test_samples,
        domain=domain,
        device=device,
    )

    # Retention carry on 200 elementary transitions
    try:
        audit_res = audit_elementary_transition_space(
            model, tokenizer, device, format_type="canonical_state"
        )
        carry_retention = round(audit_res.get("overall_carry_acc", 0.0) / 100.0, 4)
    except Exception:
        carry_retention = 0.0

    # 5. For LoRA: Ablation with Adapters Disabled
    disabled_id_acc = None
    disabled_carry = None
    if is_lora:
        toggle_lora_adapters(model, enabled=False)
        disabled_id_acc, _, _ = evaluate_samples_accuracy(
            model=model,
            tokenizer=tokenizer,
            samples=id_test_samples,
            domain=domain,
            device=device,
        )
        try:
            audit_dis = audit_elementary_transition_space(
                model, tokenizer, device, format_type="canonical_state"
            )
            disabled_carry = round(audit_dis.get("overall_carry_acc", 0.0) / 100.0, 4)
        except Exception:
            disabled_carry = 0.0
        # Re-enable
        toggle_lora_adapters(model, enabled=True)

    # Verify base weights are preserved
    base_hash_after = hash_base_parameters(model)
    base_preserved = (base_hash_before == base_hash_after)

    # 6. Diagnostics Package
    mean_p_norm = sum(practice_grad_norms) / max(1, len(practice_grad_norms))
    mean_r_norm = sum(replay_grad_norms) / max(1, len(replay_grad_norms)) if replay_grad_norms else 0.0
    grad_ratio = mean_p_norm / max(1e-6, mean_r_norm) if mean_r_norm > 0 else 1.0

    train_red = round(((init_train_loss - final_train_loss) / max(1e-6, init_train_loss)) * 100.0, 2)
    test_red = round(((init_test_loss - final_test_loss) / max(1e-6, init_test_loss)) * 100.0, 2)

    diag = DiagnosticResult(
        initial_train_loss=init_train_loss,
        final_train_loss=final_train_loss,
        train_loss_reduction_pct=train_red,
        initial_test_loss=init_test_loss,
        final_test_loss=final_test_loss,
        test_loss_reduction_pct=test_red,
        mean_practice_grad_norm=round(mean_p_norm, 4),
        mean_replay_grad_norm=round(mean_r_norm, 4),
        grad_norm_ratio_practice_to_replay=round(grad_ratio, 3),
        semantic_correct_count=id_correct,
        format_failure_count=id_tot - id_correct,
        total_evaluated=id_tot,
    )

    return AdaptationRunResult(
        regime_name=regime,
        scale=len(practice_samples),
        trainable_params=stats["trainable_params"],
        total_params=stats["total_params"],
        trainable_ratio_pct=stats["trainable_ratio_pct"],
        initial_train_loss=init_train_loss,
        final_train_loss=final_train_loss,
        id_test_accuracy=id_acc,
        ood_test_accuracy=ood_acc,
        retention_carry_accuracy=carry_retention,
        adapter_disabled_id_accuracy=disabled_id_acc,
        adapter_disabled_retention_carry=disabled_carry,
        base_weights_preserved=base_preserved,
        diagnostics=diag,
    )
