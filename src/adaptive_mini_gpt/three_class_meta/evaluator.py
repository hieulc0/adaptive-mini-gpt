"""Phase 5 & 6: Cross-Family Meta-Evaluation & Generalization Audit."""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from adaptive_mini_gpt.guided_experiment.environment import AutonomousEnvironment
from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.three_class_meta.broad_meta_learning import M64InitializationArm
from adaptive_mini_gpt.three_class_meta.three_class_engine import (
    ThreeClassAdaptationTrajectory,
    ThreeClassSupervisionEngine,
    ThreeClassSupervisionMode,
)

logger = logging.getLogger(__name__)


@dataclass
class FamilyGeneralizationReport:
    """Rigorous audit of in-family vs cross-family transfer advantages."""
    in_family_advantage_auc: float
    cross_family_advantage_auc: float
    in_family_tasks: list[str]
    unseen_family_tasks: list[str]
    cross_family_paired_correctness: float
    generalization_verdict: str


class CrossFamilyMetaEvaluator:
    """Evaluates candidate initializations across familiar and unseen rule families."""

    def __init__(
        self,
        base_model: MiniGPT,
        tokenizer: Any,
        device: torch.device,
        learning_rate: float = 5e-4,
    ) -> None:
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device
        self.engine = ThreeClassSupervisionEngine(
            base_model=base_model,
            tokenizer=tokenizer,
            device=device,
            learning_rate=learning_rate,
            anchor_lambda=0.01,
            max_grad_norm=1.0,
        )

    def evaluate_initialization_arm(
        self,
        env: AutonomousEnvironment,
        arm: M64InitializationArm,
        init_state_dict: dict[str, torch.Tensor] | None = None,
        seed: int = 42,
    ) -> ThreeClassAdaptationTrajectory:
        """Run stabilized three-class adaptation for the assigned arm."""
        max_steps = 60 if arm == M64InitializationArm.ARM_G_COMPUTE_MATCHED_EXTENDED_FRESH else 40
        return self.engine.adapt_and_evaluate(
            env=env,
            mode=ThreeClassSupervisionMode.M64_THREE_CLASS_TRAINING,
            initial_state_dict=init_state_dict,
            max_steps=max_steps,
            seed=seed,
        )

    @classmethod
    def audit_family_generalization(
        cls,
        task_arm_evals: dict[str, dict[str, ThreeClassAdaptationTrajectory]],
        target_meta_arm: str,
        baseline_arm: str,
    ) -> FamilyGeneralizationReport:
        """Audit transfer advantage across familiar vs unseen rule families."""
        in_fam_diffs: list[float] = []
        cross_fam_diffs: list[float] = []
        in_fam_ids: list[str] = []
        cross_fam_ids: list[str] = []
        cross_paired_vals: list[float] = []

        for env_id, arms in task_arm_evals.items():
            meta_traj = arms.get(target_meta_arm)
            base_traj = arms.get(baseline_arm)
            if not meta_traj or not base_traj:
                continue

            diff = meta_traj.horizon_quality_auc - base_traj.horizon_quality_auc
            is_familiar = any(k in env_id for k in ["chain", "hierarchy", "branching", "tree"])

            if is_familiar:
                in_fam_diffs.append(diff)
                in_fam_ids.append(env_id)
            else:
                cross_fam_diffs.append(diff)
                cross_fam_ids.append(env_id)
                cross_paired_vals.append(meta_traj.final_metrics.paired_correctness)

        in_mean = float(np.mean(in_fam_diffs)) if in_fam_diffs else 0.0
        cross_mean = float(np.mean(cross_fam_diffs)) if cross_fam_diffs else 0.0
        cross_paired = float(np.mean(cross_paired_vals)) if cross_paired_vals else 0.0

        if cross_mean > 0.02 and in_mean > 0.02:
            verdict = "CROSS-FAMILY GENERALIZATION PASS: Meta-initialization transfers reliably to unseen families."
        elif in_mean > 0.0 and cross_mean > 0.0:
            verdict = "MARGINAL GENERALIZATION: Positive transfer observed across both familiar and unseen families."
        elif in_mean > 0.0:
            verdict = "IN-FAMILY ONLY: Meta-learning benefits familiar topologies but fails on unseen families."
        else:
            verdict = "CROSS-FAMILY FAIL: No reliable transfer advantage established."

        return FamilyGeneralizationReport(
            in_family_advantage_auc=in_mean,
            cross_family_advantage_auc=cross_mean,
            in_family_tasks=in_fam_ids,
            unseen_family_tasks=cross_fam_ids,
            cross_family_paired_correctness=cross_paired,
            generalization_verdict=verdict,
        )
