"""Phase 4 & 5: Fair Meta-Initialization Re-Evaluation, Hybrid Blending, & Generalization."""
from __future__ import annotations

import copy
import enum
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter
from adaptive_mini_gpt.guided_experiment.environment import AutonomousEnvironment
from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.reliable_transfer.composition_models import ResidualSkillAdapter
from adaptive_mini_gpt.stable_few_shot.stable_optimizer import (
    FewShotSupervisionObjective,
    StableAdaptationTrajectory,
    StableFewShotEngine,
    compute_standard_horizon_auc,
)

logger = logging.getLogger(__name__)


class M63InitializationArm(str, enum.Enum):
    """Controlled arms for meta-initialization evaluation under stabilized protocol."""
    ARM_A_FRESH_RANDOM_INIT = "Arm_A_Fresh_Random_Init"
    ARM_B_COMPATIBLE_TRANSFER = "Arm_B_Compatible_Transfer"
    ARM_C_MULTITASK_PRETRAINED = "Arm_C_Multitask_Pretrained"
    ARM_D_REPTILE_META_LEARNED = "Arm_D_Reptile_Meta_Learned"
    ARM_E_SHUFFLED_META_LEARNING = "Arm_E_Shuffled_Meta_Learning"
    ARM_F_COMPUTE_MATCHED_EXTENDED_FRESH = "Arm_F_Compute_Matched_Extended_Fresh"
    ARM_G_MINIMAL_HYBRID_INIT = "Arm_G_Minimal_Hybrid_Init"


@dataclass
class MetaInitArmMetrics:
    """Evaluation summary for an initialization arm."""
    arm_name: str
    initial_f1: float
    final_f1: float
    horizon_quality_auc: float
    compute_adjusted_efficiency: float
    paired_correctness: float
    actual_steps: int
    wall_clock_ms: float
    n0_f1: float
    n2_f1: float
    n4_f1: float
    n8_f1: float
    n16_f1: float
    n32_f1: float
    regression_rate_pct: float


@dataclass
class GeneralizationAuditReport:
    """In-family vs out-of-family generalization comparison."""
    in_family_advantage: float
    cross_family_advantage: float
    generalization_verdict: str


class BalancedReptileMetaLearner:
    """Reptile Meta-Learner enforcing class-balanced inner loops and gradient stability."""

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: Any,
        device: torch.device,
        inner_lr: float = 1e-3,
        meta_lr: float = 0.10,
        inner_steps: int = 6,
        bottleneck_dim: int = 16,
        num_classes: int = 3,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.num_classes = num_classes
        self.d_model = getattr(base_model.config, "d_model", getattr(base_model.config, "n_embd", 128))

        self.meta_adapter = PersistentCandidateAdapter(
            d_model=self.d_model,
            bottleneck_dim=bottleneck_dim,
            num_classes=num_classes,
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

    def meta_train_step(self, env: AutonomousEnvironment) -> float:
        """Execute one meta-training iteration using class-balanced minimal-pair steps."""
        worker = copy.deepcopy(self.meta_adapter)
        worker.train()
        optimizer = torch.optim.AdamW(worker.parameters(), lr=self.inner_lr, weight_decay=1e-4)

        pairs = env.minimal_pairs
        if not pairs:
            return 0.0

        for step in range(self.inner_steps):
            optimizer.zero_grad()
            mp = pairs[step % len(pairs)]
            h_b = self._extract_hidden(mp.base_query.question_text)
            h_p = self._extract_hidden(mp.perturbed_query.question_text)
            l_b = worker(h_b)
            l_p = worker(h_p)

            t_b = 1 if mp.base_query.expected_answer == "YES" else 0
            t_p = 1 if mp.perturbed_query.expected_answer == "YES" else 0

            loss = 0.5 * (
                F.cross_entropy(l_b, torch.tensor([t_b], dtype=torch.long, device=self.device))
                + F.cross_entropy(l_p, torch.tensor([t_p], dtype=torch.long, device=self.device))
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(worker.parameters(), max_norm=1.0)
            optimizer.step()

        # Reptile outer update
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


def train_balanced_multitask_adapter(
    base_model: MiniGPT,
    tokenizer: Any,
    device: torch.device,
    envs: list[AutonomousEnvironment],
    total_steps: int = 50,
    bottleneck_dim: int = 16,
) -> dict[str, torch.Tensor]:
    """Train standard multitask baseline with class-balanced pairs."""
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

    for step in range(total_steps):
        env = envs[step % len(envs)]
        if not env.minimal_pairs:
            continue
        mp = env.minimal_pairs[step % len(env.minimal_pairs)]
        optimizer.zero_grad()
        h_b = extract_h(mp.base_query.question_text)
        h_p = extract_h(mp.perturbed_query.question_text)
        l_b = adapter(h_b)
        l_p = adapter(h_p)

        t_b = 1 if mp.base_query.expected_answer == "YES" else 0
        t_p = 1 if mp.perturbed_query.expected_answer == "YES" else 0

        loss = 0.5 * (
            F.cross_entropy(l_b, torch.tensor([t_b], dtype=torch.long, device=device))
            + F.cross_entropy(l_p, torch.tensor([t_p], dtype=torch.long, device=device))
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
        optimizer.step()

    return {k: v.detach().clone() for k, v in adapter.state_dict().items()}


def create_minimal_hybrid_init(
    meta_state: dict[str, torch.Tensor],
    multitask_state: dict[str, torch.Tensor],
    alpha: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Create linearly blended hybrid initialization: theta = alpha * meta + (1 - alpha) * multi."""
    hybrid = {}
    for k in meta_state:
        if k in multitask_state and meta_state[k].shape == multitask_state[k].shape:
            hybrid[k] = alpha * meta_state[k] + (1.0 - alpha) * multitask_state[k]
        else:
            hybrid[k] = meta_state[k].clone()
    return hybrid


class FairMetaInitEvaluator:
    """Evaluates all candidate initializations under matched, stabilized adaptation conditions."""

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: Any,
        device: torch.device,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.engine = StableFewShotEngine(
            base_model=base_model,
            tokenizer=tokenizer,
            device=device,
            learning_rate=5e-4,
            max_grad_norm=1.0,
            anchor_lambda=0.01,
        )

    def evaluate_initialization_arm(
        self,
        env: AutonomousEnvironment,
        arm: M63InitializationArm,
        init_state_dict: dict[str, torch.Tensor] | None = None,
        seed: int = 42,
    ) -> StableAdaptationTrajectory:
        """Run stabilized adaptation from the given initialization state."""
        max_steps = 60 if arm == M63InitializationArm.ARM_F_COMPUTE_MATCHED_EXTENDED_FRESH else 40
        return self.engine.adapt_and_evaluate(
            env=env,
            objective=FewShotSupervisionObjective.CONSERVATIVE_COMBINED,
            initial_state_dict=init_state_dict,
            max_steps=max_steps,
            seed=seed,
        )

    @staticmethod
    def audit_cross_family_generalization(
        arm_outcomes: dict[str, dict[str, StableAdaptationTrajectory]],
        meta_arm: str,
        baseline_arm: str,
    ) -> GeneralizationAuditReport:
        """Compare in-family vs out-of-family transfer advantages."""
        in_family_diffs: list[float] = []
        cross_family_diffs: list[float] = []

        for env_id, arms in arm_outcomes.items():
            meta_res = arms.get(meta_arm)
            base_res = arms.get(baseline_arm)
            if not meta_res or not base_res:
                continue

            diff = meta_res.horizon_quality_auc - base_res.horizon_quality_auc
            # In-family environments: deep_chain, balanced_hierarchy, branching
            if any(k in env_id for k in ["chain", "hierarchy", "branching"]):
                in_family_diffs.append(diff)
            else:
                cross_family_diffs.append(diff)

        in_mean = float(np.mean(in_family_diffs)) if in_family_diffs else 0.0
        cross_mean = float(np.mean(cross_family_diffs)) if cross_family_diffs else 0.0

        if cross_mean > 0.02 and in_mean > 0.02:
            verdict = "GENERALIZATION PASS: Meta-initialization transfers reliably across families."
        elif in_mean > 0.0:
            verdict = "IN-FAMILY ONLY: Meta-initialization improves within-family tasks but fails across families."
        else:
            verdict = "GENERALIZATION FAIL: No cross-family or in-family advantage established."

        return GeneralizationAuditReport(
            in_family_advantage=in_mean,
            cross_family_advantage=cross_mean,
            generalization_verdict=verdict,
        )
