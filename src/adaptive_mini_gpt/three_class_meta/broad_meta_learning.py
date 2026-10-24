"""Phase 4 & 5: Narrow vs Broad-Family Reptile Meta-Training & Controlled Baselines."""
from __future__ import annotations

import copy
import enum
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter
from adaptive_mini_gpt.guided_experiment.environment import AutonomousEnvironment
from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.three_class_meta.three_class_sampler import ThreeClassFewShotSampler

logger = logging.getLogger(__name__)


class M64InitializationArm(str, enum.Enum):
    """Candidate initializations for M64 cross-family meta-generalization."""
    ARM_A_FRESH_RANDOM = "Arm_A_Fresh_Random"
    ARM_B_MULTITASK_PRETRAINED = "Arm_B_Multitask_Pretrained"
    ARM_C_NARROW_REPTILE = "Arm_C_Narrow_Reptile"
    ARM_D_BROAD_REPTILE = "Arm_D_Broad_Reptile"
    ARM_E_SHUFFLED_CONTROL = "Arm_E_Shuffled_Control"
    ARM_F_COMPATIBLE_TRANSFER = "Arm_F_Compatible_Transfer"
    ARM_G_COMPUTE_MATCHED_EXTENDED_FRESH = "Arm_G_Compute_Matched_Extended_Fresh"
    ARM_H_FIXED_HYBRID = "Arm_H_Fixed_Hybrid"


class ThreeClassReptileLearner:
    """Reptile Meta-Learner operating under genuine 3-class supervision (YES, NO, UNKNOWN)."""

    CLASS_LABELS = ["NO", "YES", "UNKNOWN"]

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: Any,
        device: torch.device,
        inner_lr: float = 1e-3,
        meta_lr: float = 0.10,
        inner_steps: int = 6,
        bottleneck_dim: int = 16,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.d_model = getattr(base_model.config, "d_model", getattr(base_model.config, "n_embd", 128))

        self.meta_adapter = PersistentCandidateAdapter(
            d_model=self.d_model,
            bottleneck_dim=bottleneck_dim,
            num_classes=3,
        ).to(device)

    def _extract_hidden(self, text: str) -> torch.Tensor:
        toks = self.tokenizer.encode(text)
        inp = torch.tensor([toks], dtype=torch.long, device=self.device)
        b, t = inp.size()
        tok_emb = self.base_model.wte(inp)
        pos = torch.arange(0, t, dtype=torch.long, device=self.device)
        x = self.base_model.drop(tok_emb + self.base_model.wpe(pos))
        for block in self.base_model.blocks:
            x, _ = block(x)
        return self.base_model.ln_f(x)

    def meta_train_iteration(self, env: AutonomousEnvironment) -> float:
        """Run one meta-training step on a task environment with balanced 3-class practice queries."""
        worker = copy.deepcopy(self.meta_adapter)
        worker.train()
        optimizer = torch.optim.AdamW(worker.parameters(), lr=self.inner_lr, weight_decay=1e-4)

        # Draw practice subset with balanced YES, NO, UNKNOWN
        manifest = ThreeClassFewShotSampler.get_budgeted_subset(env, n_samples=16)
        queries = manifest.queries

        for step in range(self.inner_steps):
            optimizer.zero_grad()
            q = queries[step % len(queries)]
            h = self._extract_hidden(q.question_text)
            logits = worker(h)
            tgt_idx = self.CLASS_LABELS.index(q.expected_answer)
            loss = F.cross_entropy(logits, torch.tensor([tgt_idx], dtype=torch.long, device=self.device))

            if env.minimal_pairs:
                mp = env.minimal_pairs[step % len(env.minimal_pairs)]
                h_b = self._extract_hidden(mp.base_query.question_text)
                h_p = self._extract_hidden(mp.perturbed_query.question_text)
                l_b = worker(h_b)
                l_p = worker(h_p)
                t_b = self.CLASS_LABELS.index(mp.base_query.expected_answer)
                t_p = self.CLASS_LABELS.index(mp.perturbed_query.expected_answer)
                loss_pair = 0.5 * (
                    F.cross_entropy(l_b, torch.tensor([t_b], dtype=torch.long, device=self.device))
                    + F.cross_entropy(l_p, torch.tensor([t_p], dtype=torch.long, device=self.device))
                )
                loss = loss + loss_pair

            loss.backward()
            torch.nn.utils.clip_grad_norm_(worker.parameters(), max_norm=1.0)
            optimizer.step()

        # Outer Reptile update: theta <- theta + meta_lr * (worker - theta)
        meta_dict = self.meta_adapter.state_dict()
        worker_dict = worker.state_dict()
        step_norm = 0.0

        with torch.no_grad():
            for k in meta_dict:
                diff = worker_dict[k] - meta_dict[k]
                meta_dict[k] += self.meta_lr * diff
                step_norm += float(torch.norm(diff).item())

        self.meta_adapter.load_state_dict(meta_dict)
        return step_norm

    def get_meta_state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.meta_adapter.state_dict().items()}


def train_three_class_multitask_adapter(
    base_model: MiniGPT,
    tokenizer: Any,
    device: torch.device,
    envs: list[AutonomousEnvironment],
    total_steps: int = 50,
    bottleneck_dim: int = 16,
) -> dict[str, torch.Tensor]:
    """Train standard multitask baseline with genuine 3-class practice queries."""
    adapter = PersistentCandidateAdapter(
        d_model=getattr(base_model.config, "d_model", 128),
        bottleneck_dim=bottleneck_dim,
        num_classes=3,
    ).to(device)
    adapter.train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)

    def extract_h(text: str) -> torch.Tensor:
        toks = tokenizer.encode(text)
        inp = torch.tensor([toks], dtype=torch.long, device=device)
        b, t = inp.size()
        tok_emb = base_model.wte(inp)
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = base_model.drop(tok_emb + base_model.wpe(pos))
        for block in base_model.blocks:
            x, _ = block(x)
        return base_model.ln_f(x)

    class_labels = ["NO", "YES", "UNKNOWN"]

    for step in range(total_steps):
        env = envs[step % len(envs)]
        manifest = ThreeClassFewShotSampler.get_budgeted_subset(env, n_samples=16)
        if not manifest.queries:
            continue
        q = manifest.queries[step % len(manifest.queries)]

        optimizer.zero_grad()
        h = extract_h(q.question_text)
        logits = adapter(h)
        tgt_idx = class_labels.index(q.expected_answer)
        loss = F.cross_entropy(logits, torch.tensor([tgt_idx], dtype=torch.long, device=device))

        if env.minimal_pairs:
            mp = env.minimal_pairs[step % len(env.minimal_pairs)]
            h_b = extract_h(mp.base_query.question_text)
            h_p = extract_h(mp.perturbed_query.question_text)
            l_b = adapter(h_b)
            l_p = adapter(h_p)
            t_b = class_labels.index(mp.base_query.expected_answer)
            t_p = class_labels.index(mp.perturbed_query.expected_answer)
            loss_pair = 0.5 * (
                F.cross_entropy(l_b, torch.tensor([t_b], dtype=torch.long, device=device))
                + F.cross_entropy(l_p, torch.tensor([t_p], dtype=torch.long, device=device))
            )
            loss = loss + loss_pair

        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
        optimizer.step()

    return {k: v.detach().clone() for k, v in adapter.state_dict().items()}


def create_hybrid_init(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    alpha: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Blend two parameter initializations: alpha * state_a + (1 - alpha) * state_b."""
    hybrid = {}
    for k in state_a:
        if k in state_b and state_a[k].shape == state_b[k].shape:
            hybrid[k] = alpha * state_a[k] + (1.0 - alpha) * state_b[k]
        else:
            hybrid[k] = state_a[k].clone()
    return hybrid
