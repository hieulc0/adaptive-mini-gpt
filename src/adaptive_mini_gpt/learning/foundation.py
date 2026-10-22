"""M34 Phase 2: Foundation Checkpoint Analysis and Capability Baseline.

Compares:
  Foundation A: Raw M18 base neural root (expanded to 1024-BPE, 0% prior arithmetic/logic).
  Foundation B: Verified M32 recurrently trained foundation with documented skills
                (93.3% 2-digit, 89.6% 4-digit addition, 100% carry retention).

Ensures all subsequent modular and full fine-tuning experiments evaluate retained skills
relative to an actual acquired capability, rather than an untrained root.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from adaptive_mini_gpt.data.bpe_scaling import (
    expand_model_vocabulary,
    train_scaled_bpe_tokenizer,
)
from adaptive_mini_gpt.data.language_corpus import PRETRAIN_PROSE_CORPUS
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.transformer import MiniGPT, ModelConfig
from adaptive_mini_gpt.reasoning.dataset import (
    MultiRepresentationReasoningDataset,
    collate_reasoning_batch,
)
from adaptive_mini_gpt.reasoning.recurrent_curriculum import (
    build_failure_targeted_recurrent_practice,
)
from adaptive_mini_gpt.reasoning.scratchpad import ApproachType, ReasoningSample
from adaptive_mini_gpt.rules.reliable_transition import (
    ElementaryTransitionDataset,
    collate_elementary_batch,
    get_all_200_transitions,
)
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


@dataclass
class FoundationComparisonRecord:
    """Documented baseline performance of a starting foundation checkpoint."""

    foundation_id: str
    description: str
    parameter_count: int
    arithmetic_4digit_exact: float
    arithmetic_step_acc: float
    logic_exact: float
    language_exact: float
    carry_retention_200: float
    prose_loss: float


def train_m32_recurrent_foundation(
    base_model: MiniGPT,
    tokenizer: BPETokenizer,
    device: torch.device,
    seed: int = 42,
    epochs: int = 12,
    lr: float = 3e-4,
    batch_size: int = 8,
) -> MiniGPT:
    """Train the verified M32 recurrent foundation model with failure-targeted practice and 200-transition replay."""
    logger.info("Training M32 Recurrent Foundation model...")
    model = copy.deepcopy(base_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    max_ctx = getattr(model.config, "context_length", 256)
    pad_id = getattr(tokenizer, "pad_id", 0)

    practice_samples = build_failure_targeted_recurrent_practice(
        target_sample_count=128,
        seed=3000 + seed,
    )
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

    transitions = get_all_200_transitions()
    ds_replay = ElementaryTransitionDataset(
        transitions=transitions,
        tokenizer=tokenizer,
        format_type="m23_baseline",
        context_length=max_ctx,
    )
    loader_replay = DataLoader(
        ds_replay,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_elementary_batch(b, pad_token_id=pad_id),
    )

    model.train()
    iter_replay = iter(loader_replay)
    for _ in range(epochs):
        for b_inputs, b_labels in loader_practice:
            b_inputs = b_inputs.to(device)
            b_labels = b_labels.to(device)

            try:
                r_inputs, r_labels = next(iter_replay)
            except StopIteration:
                iter_replay = iter(loader_replay)
                r_inputs, r_labels = next(iter_replay)

            r_inputs = r_inputs.to(device)
            r_labels = r_labels.to(device)

            out_p = model(b_inputs)
            logits_p = out_p[0] if isinstance(out_p, tuple) else out_p
            loss_p = nn.functional.cross_entropy(
                logits_p[:, :-1, :].contiguous().view(-1, logits_p.size(-1)),
                b_labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )

            out_r = model(r_inputs)
            logits_r = out_r[0] if isinstance(out_r, tuple) else out_r
            loss_r = nn.functional.cross_entropy(
                logits_r[:, :-1, :].contiguous().view(-1, logits_r.size(-1)),
                r_labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )

            total_loss = loss_p + 0.5 * loss_r
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    return model
