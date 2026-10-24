"""Phase 6: Controlled Meta-Learning & Multi-Arm Generalization Evaluator."""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.guided_experiment.environment import AutonomousEnvironment
from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.three_valued_reasoning.architectures import (
    M65ArchitectureVariant,
    Rank16ResidualAdapter,
)
from adaptive_mini_gpt.three_valued_reasoning.engine import (
    ThreeValuedAdaptationTrajectory,
    ThreeValuedReasoningEngine,
)
from adaptive_mini_gpt.three_valued_reasoning.semantics import GroundedThreeValuedEnvironment

logger = logging.getLogger(__name__)


class M65MetaLearningArm(str, Enum):
    """5 arms compared under Phase 6 with matched compute and repaired inputs."""
    ARM_A_FRESH_RANDOM = "Arm_A_Fresh_Random"
    ARM_B_MULTITASK_PRETRAINED = "Arm_B_Multitask_Pretrained"
    ARM_C_NARROW_REPTILE = "Arm_C_Narrow_Reptile"
    ARM_D_BROAD_REPTILE = "Arm_D_Broad_Reptile"
    ARM_E_COMPUTE_MATCHED_FRESH = "Arm_E_Compute_Matched_Fresh"


def train_three_valued_multitask_adapter(
    base_model: MiniGPT,
    tokenizer: Any,
    device: torch.device,
    training_envs: list[GroundedThreeValuedEnvironment | AutonomousEnvironment],
    total_steps: int = 40,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Train multitask initialization across training environments with 3-class supervision."""
    torch.manual_seed(seed)
    adapter = Rank16ResidualAdapter(d_model=getattr(base_model.config, "d_model", 128), bottleneck_dim=16, num_classes=3).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)
    engine = ThreeValuedReasoningEngine(base_model, tokenizer, device, learning_rate=1e-3)

    pools = [engine.get_budgeted_subset(env, 32) for env in training_envs]
    adapter.train()

    for s in range(total_steps):
        opt.zero_grad()
        env_idx = s % len(training_envs)
        env = training_envs[env_idx]
        pool = pools[env_idx]
        q = pool[s % len(pool)]
        h = engine._extract_hidden(q.question_text)
        logits = adapter(h)
        tgt_idx = engine.CLASS_LABELS.index(q.expected_answer)
        loss = F.cross_entropy(logits, torch.tensor([tgt_idx], device=device))

        if env.minimal_pairs:
            mp = env.minimal_pairs[s % len(env.minimal_pairs)]
            h_b = engine._extract_hidden(mp.base_query.question_text)
            h_p = engine._extract_hidden(mp.perturbed_query.question_text)
            l_b = adapter(h_b)
            l_p = adapter(h_p)
            t_b = engine.CLASS_LABELS.index(mp.base_query.expected_answer)
            t_p = engine.CLASS_LABELS.index(mp.perturbed_query.expected_answer)
            loss_pair = 0.5 * (
                F.cross_entropy(l_b, torch.tensor([t_b], device=device))
                + F.cross_entropy(l_p, torch.tensor([t_p], device=device))
            )
            loss = loss + loss_pair

        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        opt.step()

    return copy.deepcopy(adapter.state_dict())


class ThreeValuedReptileLearner:
    """Reptile first-order meta-learning with repaired three-valued supervision."""

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
        self.d_model = getattr(base_model.config, "d_model", 128)
        self.engine = ThreeValuedReasoningEngine(base_model, tokenizer, device, learning_rate=inner_lr)

        self.meta_adapter = Rank16ResidualAdapter(
            d_model=self.d_model,
            bottleneck_dim=bottleneck_dim,
            num_classes=3,
        ).to(device)

    def meta_train_step(self, task_env: GroundedThreeValuedEnvironment | AutonomousEnvironment) -> float:
        """Perform one Reptile meta-update step on a task environment."""
        local_adapter = copy.deepcopy(self.meta_adapter)
        local_adapter.train()
        opt = torch.optim.AdamW(local_adapter.parameters(), lr=self.inner_lr, weight_decay=1e-4)

        practice_pool = self.engine.get_budgeted_subset(task_env, 32)
        if not practice_pool:
            practice_pool = task_env.held_out_test_queries

        inner_losses = []
        for s in range(self.inner_steps):
            opt.zero_grad()
            q = practice_pool[s % len(practice_pool)]
            h = self.engine._extract_hidden(q.question_text)
            logits = local_adapter(h)
            tgt_idx = self.engine.CLASS_LABELS.index(q.expected_answer)
            loss = F.cross_entropy(logits, torch.tensor([tgt_idx], device=self.device))

            if task_env.minimal_pairs:
                mp = task_env.minimal_pairs[s % len(task_env.minimal_pairs)]
                h_b = self.engine._extract_hidden(mp.base_query.question_text)
                h_p = self.engine._extract_hidden(mp.perturbed_query.question_text)
                l_b = local_adapter(h_b)
                l_p = local_adapter(h_p)
                t_b = self.engine.CLASS_LABELS.index(mp.base_query.expected_answer)
                t_p = self.engine.CLASS_LABELS.index(mp.perturbed_query.expected_answer)
                loss_pair = 0.5 * (
                    F.cross_entropy(l_b, torch.tensor([t_b], device=self.device))
                    + F.cross_entropy(l_p, torch.tensor([t_p], device=self.device))
                )
                loss = loss + loss_pair

            loss.backward()
            torch.nn.utils.clip_grad_norm_(local_adapter.parameters(), 1.0)
            opt.step()
            inner_losses.append(float(loss.item()))

        # Reptile outer step: meta_param += meta_lr * (local_param - meta_param)
        with torch.no_grad():
            for meta_p, local_p in zip(self.meta_adapter.parameters(), local_adapter.parameters()):
                meta_p.data.add_(local_p.data - meta_p.data, alpha=self.meta_lr)

        return float(np.mean(inner_losses))

    def get_meta_state_dict(self) -> dict[str, torch.Tensor]:
        return copy.deepcopy(self.meta_adapter.state_dict())
