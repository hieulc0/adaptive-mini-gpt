"""M34 Phase 3, 4, 5: Modular Continual Learning and Checkpoint Growth.

Compares three experimental approaches on the same starting backbone:
  Approach A: Full fine-tuning with balanced replay (M33 baseline).
  Approach B: Frozen backbone with separate skill adapters (no replay).
  Approach C: Frozen backbone with separate skill adapters + balanced replay.

Executes sequential multi-episode learning:
  Foundation V1 -> Episode A (Arithmetic) -> V2 -> Episode B (Logic) -> V3 -> Episode C (Language) -> V4
Guarantees:
  1. Base parameters remain 100% invariant when training adapters.
  2. Router dynamically switches adapters according to task domain.
  3. Evaluates all three domains after every episode to measure forward transfer and backward retention.
"""

from __future__ import annotations

import copy
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from adaptive_mini_gpt.data.language_corpus import (
    ABSTENTION_TARGET,
    PRETRAIN_PROSE_CORPUS,
    normalize_text,
)
from adaptive_mini_gpt.data.retention_data import evaluate_audited_retention
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.learning.cumulative_learner import (
    EpisodeSpec,
    build_m33_arithmetic_episode,
    build_m33_language_episode,
    build_m33_logic_episode,
)
from adaptive_mini_gpt.learning.modular_lineage import (
    ModularCheckpointRecord,
    ModularLineageManager,
)
from adaptive_mini_gpt.model.modular_adapter import (
    add_domain_adapter,
    compute_base_model_hash,
    freeze_for_adapter_training,
    get_adapter_state_dict,
    inject_multi_domain_adapters,
    load_adapter_state_dict,
    set_active_domain_adapter,
)
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.reasoning.adaptive_router import (
    DeterministicRouter,
    ReasoningAction,
)
from adaptive_mini_gpt.reasoning.dataset import (
    MultiRepresentationReasoningDataset,
    collate_reasoning_batch,
)
from adaptive_mini_gpt.reasoning.recurrent_reasoner import (
    rollout_recurrent_arithmetic,
    rollout_recurrent_logic,
)
from adaptive_mini_gpt.reasoning.scratchpad import ApproachType, ReasoningSample
from adaptive_mini_gpt.rules.reliable_transition import (
    ElementaryTransitionDataset,
    audit_elementary_transition_space,
    collate_elementary_batch,
    get_all_200_transitions,
)

logger = logging.getLogger(__name__)


# =========================================================================
# 1. Multi-Domain Evaluation Harness for Modular and Dense Models
# =========================================================================

def evaluate_multi_domain_system(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    device: torch.device,
    arith_test_samples: list[ReasoningSample],
    logic_test_samples: list[ReasoningSample],
    lang_test_samples: list[ReasoningSample],
    is_modular: bool = True,
    active_domain: str | None = None,
) -> dict[str, float]:
    """Evaluate system across Arithmetic, Logic, Language, Carry, and Prose retention.

    If is_modular is True and active_domain is None, automatically sets the domain adapter
    appropriate for each task.
    """
    model.eval()
    results: dict[str, float] = {}

    # 1. Arithmetic Evaluation
    if is_modular:
        set_active_domain_adapter(model, active_domain or "arithmetic")

    arith_exact = 0
    total_steps = 0
    correct_steps = 0
    for s in arith_test_samples:
        a = s.metadata.get("a", 0)
        b = s.metadata.get("b", 0)
        trace = rollout_recurrent_arithmetic(model, tokenizer, a, b, device)
        if trace.is_exact_match:
            arith_exact += 1
        total_steps += len(trace.step_accuracies)
        correct_steps += sum(1 for ok in trace.step_accuracies if ok)

    n_arith = max(1, len(arith_test_samples))
    results["arithmetic_exact"] = round(arith_exact / n_arith, 4)
    results["arithmetic_step_acc"] = (
        round(correct_steps / max(1, total_steps), 4) if total_steps > 0 else 0.0
    )

    # 2. Logic Evaluation
    if is_modular:
        set_active_domain_adapter(model, active_domain or "logic")

    logic_exact = 0
    for s in logic_test_samples:
        trace_l = rollout_recurrent_logic(model, tokenizer, s, device)
        if trace_l.is_exact_match:
            logic_exact += 1
    n_logic = max(1, len(logic_test_samples))
    results["logic_exact"] = round(logic_exact / n_logic, 4)

    # 3. Language Evaluation
    if is_modular:
        set_active_domain_adapter(model, active_domain or "language")

    max_ctx = getattr(model.config, "context_length", 256)
    eos_id = getattr(tokenizer, "eos_id", -1)
    lang_exact = 0
    abstention_tested = 0
    abstention_correct = 0

    for s in lang_test_samples:
        enc_prompt = tokenizer.encode(s.prompt_direct)
        inp = torch.tensor([enc_prompt[-max_ctx:]], dtype=torch.long, device=device)
        gen_tokens: list[int] = []
        with torch.no_grad():
            for _ in range(16):
                out = model(inp[:, -max_ctx:])
                lg = out[0] if isinstance(out, tuple) else out
                nid = int(torch.argmax(lg[0, -1, :]).item())
                if nid == eos_id:
                    break
                gen_tokens.append(nid)
                inp = torch.cat([inp, torch.tensor([[nid]], device=device)], dim=1)
                decoded_partial = tokenizer.decode(gen_tokens)
                if "\n" in decoded_partial:
                    break

        pred_ans = tokenizer.decode(gen_tokens).strip()
        exp_ans = s.expected_final_answer.strip()
        norm_pred = normalize_text(pred_ans)
        norm_exp = normalize_text(exp_ans)

        is_match = (norm_pred == norm_exp) or (norm_exp in norm_pred)
        if is_match:
            lang_exact += 1

        if s.metadata.get("is_abstention", False):
            abstention_tested += 1
            if "insufficient information" in norm_pred or "unanswerable" in norm_pred:
                abstention_correct += 1

    n_lang = max(1, len(lang_test_samples))
    results["language_exact"] = round(lang_exact / n_lang, 4)
    results["language_abstention"] = (
        round(abstention_correct / abstention_tested, 4) if abstention_tested > 0 else 1.0
    )

    # 4. Elementary Carry Retention (200 transitions)
    if is_modular:
        # Run carry retention through arithmetic adapter (or base if no arithmetic adapter)
        set_active_domain_adapter(model, active_domain or "arithmetic")

    ret_rep = audit_elementary_transition_space(
        model=model,
        tokenizer=tokenizer,
        device=device,
        format_type="m23_baseline",
    )
    results["carry_retention_200"] = round(ret_rep["overall_carry_acc"] / 100.0, 4)

    # 5. Audited Prose Retention
    if is_modular:
        set_active_domain_adapter(model, None)  # Base weights evaluate prose

    prose_rep = evaluate_audited_retention(model, tokenizer, device)
    results["prose_retention_loss"] = round(prose_rep["mean_loss"], 4)

    return results


# =========================================================================
# 2. Candidate Training Functions (Modular Adapter vs. Full Fine-Tuning)
# =========================================================================

def train_domain_adapter(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    domain_name: str,
    practice_samples: list[ReasoningSample],
    replay_samples: list[ReasoningSample],
    device: torch.device,
    use_replay: bool = False,
    epochs: int = 12,
    lr: float = 3e-4,
    batch_size: int = 8,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Train only the parameters of domain_name adapter while base parameters remain frozen."""
    logger.info(
        f"Training modular adapter '{domain_name}' (use_replay={use_replay}, "
        f"{len(practice_samples)} practice samples)..."
    )

    # Ensure only domain_name adapter requires grad
    trainable, frozen = freeze_for_adapter_training(model, domain_name)
    set_active_domain_adapter(model, domain_name)

    base_hash_before = compute_base_model_hash(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    max_ctx = getattr(model.config, "context_length", 256)
    pad_id = getattr(tokenizer, "pad_id", 0)

    ds_practice = MultiRepresentationReasoningDataset(
        samples=practice_samples,
        tokenizer=tokenizer,
        approach=ApproachType.RECURRENT,
        context_length=max_ctx,
    )
    loader_practice = DataLoader(
        ds_practice,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_reasoning_batch(b, pad_token_id=pad_id),
    )

    loader_replay = None
    if use_replay:
        transitions = get_all_200_transitions()
        ds_elementary = ElementaryTransitionDataset(
            transitions=transitions,
            tokenizer=tokenizer,
            format_type="m23_baseline",
            context_length=max_ctx,
        )
        all_replay_items: list[dict[str, torch.Tensor]] = []
        for idx in range(len(ds_elementary)):
            inp, lab = ds_elementary[idx]
            all_replay_items.append({"input_ids": inp, "labels": lab})

        if len(replay_samples) > 0:
            ds_prior = MultiRepresentationReasoningDataset(
                samples=replay_samples,
                tokenizer=tokenizer,
                approach=ApproachType.RECURRENT,
                context_length=max_ctx,
            )
            for idx in range(len(ds_prior)):
                all_replay_items.append(ds_prior[idx])

        class ReplayDS(Dataset[dict[str, torch.Tensor]]):
            def __len__(self) -> int:
                return len(all_replay_items)

            def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                return all_replay_items[idx]

        loader_replay = DataLoader(
            ReplayDS(),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_reasoning_batch(b, pad_token_id=pad_id),
        )

    model.train()
    iter_replay = iter(loader_replay) if loader_replay is not None else None

    for _ in range(epochs):
        for b_inputs, b_labels in loader_practice:
            b_inputs = b_inputs.to(device)
            b_labels = b_labels.to(device)

            out_p = model(b_inputs)
            logits_p = out_p[0] if isinstance(out_p, tuple) else out_p
            loss_p = nn.functional.cross_entropy(
                logits_p[:, :-1, :].contiguous().view(-1, logits_p.size(-1)),
                b_labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )

            if use_replay and iter_replay is not None:
                try:
                    r_inputs, r_labels = next(iter_replay)
                except StopIteration:
                    iter_replay = iter(loader_replay)
                    r_inputs, r_labels = next(iter_replay)

                r_inputs = r_inputs.to(device)
                r_labels = r_labels.to(device)

                out_r = model(r_inputs)
                logits_r = out_r[0] if isinstance(out_r, tuple) else out_r
                loss_r = nn.functional.cross_entropy(
                    logits_r[:, :-1, :].contiguous().view(-1, logits_r.size(-1)),
                    r_labels[:, 1:].contiguous().view(-1),
                    ignore_index=-100,
                )
                total_loss = loss_p + 0.5 * loss_r
            else:
                total_loss = loss_p

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

    model.eval()

    # Verify base parameter hash was completely invariant
    base_hash_after = compute_base_model_hash(model)
    if base_hash_before != base_hash_after:
        raise RuntimeError(
            f"Base weight invariance violated! Hash changed from {base_hash_before} to {base_hash_after}"
        )

    adapter_state = get_adapter_state_dict(model, domain_name)
    stats = {
        "trainable_params": trainable,
        "frozen_params": frozen,
        "trainable_pct": round(trainable / max(1, trainable + frozen) * 100.0, 2),
        "base_hash": base_hash_after,
        "base_invariant": True,
    }
    return adapter_state, stats


# =========================================================================
# 3. Router and Modular Inference Harness
# =========================================================================

class ModularReasoner:
    """Combines deterministic router with multi-adapter MiniGPT for end-to-end inference."""

    def __init__(
        self,
        model: MiniGPT,
        tokenizer: BPETokenizer,
        device: torch.device,
        available_adapters: list[str],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.available_adapters = available_adapters
        self.router = DeterministicRouter()

    def reason(self, prompt: str) -> tuple[str, str, float]:
        """Route prompt, activate domain adapter, and execute reasoning procedure.

        Returns:
            (routed_action, output_answer, latency_ms)
        """
        t0 = time.perf_counter()
        decision = self.router.route(prompt)

        # Activate domain adapter based on action & prompt keywords
        lower = prompt.lower()
        if "add:" in lower or "col:" in lower or "sum" in lower:
            dom = "arithmetic"
        elif "rule:" in lower or "facts:" in lower or "deduce" in lower:
            dom = "logic"
        elif "context:" in lower:
            dom = "language"
        else:
            dom = None

        if dom is not None and dom in self.available_adapters:
            set_active_domain_adapter(self.model, dom)
        else:
            set_active_domain_adapter(self.model, None)

        # Autoregressive generation
        max_ctx = getattr(self.model.config, "context_length", 256)
        eos_id = getattr(self.tokenizer, "eos_id", -1)
        enc = self.tokenizer.encode(prompt)
        inp = torch.tensor([enc[-max_ctx:]], dtype=torch.long, device=self.device)
        gen_tokens: list[int] = []

        with torch.no_grad():
            for _ in range(32):
                out = self.model(inp[:, -max_ctx:])
                lg = out[0] if isinstance(out, tuple) else out
                nid = int(torch.argmax(lg[0, -1, :]).item())
                if nid == eos_id:
                    break
                gen_tokens.append(nid)
                inp = torch.cat([inp, torch.tensor([[nid]], device=self.device)], dim=1)
                decoded = self.tokenizer.decode(gen_tokens)
                if "\n" in decoded:
                    break

        ans = self.tokenizer.decode(gen_tokens).strip()
        lat_ms = (time.perf_counter() - t0) * 1000.0
        return decision.action.value, ans, round(lat_ms, 2)
