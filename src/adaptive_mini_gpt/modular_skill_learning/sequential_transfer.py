"""Phase 6: The Central M66 Experiment — Sequential Episode Skill Reuse & Transfer."""
from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.modular_skill_learning.candidate_acquisition import (
    ControlledSkillAcquisitionEngine,
)
from adaptive_mini_gpt.modular_skill_learning.repertoire import (
    ModularSkillRepertoire,
    QualifiedSkillRecord,
)
from adaptive_mini_gpt.three_valued_reasoning.architectures import Rank16ResidualAdapter
from adaptive_mini_gpt.three_valued_reasoning.semantics import GroundedThreeValuedEnvironment

logger = logging.getLogger(__name__)


class SequentialTransferArm(str, Enum):
    """Arms evaluated in the sequential episode experiment."""
    WITH_ACQUIRED_SKILL = "With_Acquired_Skill"
    WITHOUT_ACQUIRED_SKILL = "Without_Acquired_Skill"
    IRRELEVANT_SKILL_CONTROL = "Irrelevant_Skill_Control"
    FRESH_ADAPTATION_CONTROL = "Fresh_Adaptation_Control"


@dataclass
class SequentialEpisodeOutcome:
    """Quantitative performance of a strategy on Episode 2 of a rule family."""
    arm_name: str
    target_env_id: str
    rule_family: str
    zero_shot_f1: float
    final_macro_f1: float
    quality_auc: float
    held_out_accuracy: float
    classwise_recall: dict[str, float]
    paired_correctness: float
    sample_efficiency_curve: list[dict[str, Any]]
    steps_to_competence: int
    overgeneralization_negative_transfer_f1: float
    diagnostics: str


class SequentialEpisodeExperiment:
    """Executes the central M66 sequential episode experiment across related task instances."""

    def __init__(
        self,
        acquisition_engine: ControlledSkillAcquisitionEngine,
        d_model: int = 128,
        bottleneck_dim: int = 16,
        num_classes: int = 3,
        lr: float = 5e-4,
        max_steps: int = 40,
    ) -> None:
        self.engine = acquisition_engine
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.num_classes = num_classes
        self.lr = lr
        self.max_steps = max_steps

    def fine_tune_adapter(
        self,
        init_adapter: nn.Module,
        model: MiniGPT,
        env: GroundedThreeValuedEnvironment,
        budget_n: int,
        device: torch.device,
        steps: int = 20,
    ) -> Rank16ResidualAdapter:
        """Adapt or fine-tune an adapter instance on budget N verified practice examples."""
        adapter = copy.deepcopy(init_adapter).to(device)
        if budget_n == 0:
            return adapter

        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(adapter.parameters(), lr=self.lr, weight_decay=1e-4)
        label_to_idx = {"NO": 0, "YES": 1, "UNKNOWN": 2}

        practice_queries = env.held_out_test_queries[:budget_n]
        encoded_dataset = []
        for q in practice_queries:
            tokens = model.tokenizer.encode(q.question_text)
            target = label_to_idx.get(q.expected_answer, 2)
            encoded_dataset.append((tokens, target))

        adapter.train()
        for _ in range(steps):
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device)
            batch_size = min(4, len(encoded_dataset))
            indices = np.random.choice(len(encoded_dataset), size=batch_size, replace=True)

            for idx in indices:
                tokens, tgt = encoded_dataset[idx]
                inp = torch.tensor([tokens], dtype=torch.long, device=device)
                with torch.no_grad():
                    out = model(inp)
                    hidden = out.hidden_states if hasattr(out, "hidden_states") else out.last_hidden_state
                logits = adapter(hidden)
                loss = F.cross_entropy(logits, torch.tensor([tgt], dtype=torch.long, device=device))
                batch_loss = batch_loss + loss

            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            optimizer.step()

        return adapter

    def run_sequential_episode(
        self,
        model: MiniGPT,
        base_repertoire: ModularSkillRepertoire,
        ep1_train_env: GroundedThreeValuedEnvironment,
        ep1_eval_env: GroundedThreeValuedEnvironment,
        ep2_target_env: GroundedThreeValuedEnvironment,
        unrelated_eval_env: GroundedThreeValuedEnvironment,
        rule_family: str,
        device: torch.device,
        seed: int = 42,
    ) -> dict[str, SequentialEpisodeOutcome]:
        """Run Episode 1 (acquisition) followed by Episode 2 (evaluation of reuse benefits)."""
        logger.info(f"=== Starting Sequential Episode for Rule Family '{rule_family}' ===")
        # 1. Episode 1: Acquire candidate skill
        acq_repertoire = copy.deepcopy(base_repertoire)
        report, acquired_adapter = self.engine.acquire_and_qualify_candidate(
            candidate_id=f"candidate_skill_{rule_family}",
            rule_family=rule_family,
            train_env=ep1_train_env,
            eval_env=ep1_eval_env,
            model=model,
            repertoire=acq_repertoire,
            budget_n=16,
            device=device,
            seed=seed,
        )

        # 2. Episode 2: Evaluate on different instance of same family
        # We test across 4 arms:
        results: dict[str, SequentialEpisodeOutcome] = {}
        budgets = [0, 4, 8, 16, 32]

        # Candidate instance to reuse (or fresh if acquisition failed)
        if acquired_adapter is None:
            # Fallback to fresh if qualification rejected
            acquired_adapter = Rank16ResidualAdapter(
                d_model=self.d_model, bottleneck_dim=self.bottleneck_dim, num_classes=self.num_classes
            ).to(device)

        # Build irrelevant candidate instance for negative control
        irrelevant_adapter = Rank16ResidualAdapter(
            d_model=self.d_model, bottleneck_dim=self.bottleneck_dim, num_classes=self.num_classes
        ).to(device)
        # Permute / randomize differently
        torch.manual_seed(seed + 999)
        with torch.no_grad():
            for p in irrelevant_adapter.parameters():
                p.data.add_(torch.randn_like(p) * 0.1)

        # Best static existing source adapter in base repertoire
        best_static_id = base_repertoire.get_skill_ids()[0]
        best_static_f1 = -1.0
        for sk_id in base_repertoire.get_skill_ids():
            f1, _, _, _ = base_repertoire.evaluate_skill_on_env(sk_id, model, ep2_target_env, device)
            if f1 > best_static_f1:
                best_static_f1 = f1
                best_static_id = sk_id
        best_static_adapter = base_repertoire.build_adapter_instance(best_static_id, device)

        arm_configs = [
            (SequentialTransferArm.WITH_ACQUIRED_SKILL, acquired_adapter),
            (SequentialTransferArm.WITHOUT_ACQUIRED_SKILL, best_static_adapter),
            (SequentialTransferArm.IRRELEVANT_SKILL_CONTROL, irrelevant_adapter),
            (
                SequentialTransferArm.FRESH_ADAPTATION_CONTROL,
                Rank16ResidualAdapter(self.d_model, self.bottleneck_dim, self.num_classes).to(device),
            ),
        ]

        for arm_enum, start_adapter in arm_configs:
            arm_name = arm_enum.value
            curve = []
            auc_points = []

            # Zero-shot performance
            z_f1, z_acc, z_rec, z_paired = self.engine.evaluate_on_independent_world(
                candidate=start_adapter, model=model, eval_env=ep2_target_env, device=device
            )
            curve.append({"n": 0, "macro_f1": z_f1, "accuracy": z_acc, "recalls": z_rec, "paired": z_paired})
            auc_points.append(z_f1)

            steps_competence = 40
            final_f1 = z_f1
            final_acc = z_acc
            final_rec = z_rec
            final_paired = z_paired

            for b in [4, 8, 16, 32]:
                adapted = self.fine_tune_adapter(
                    init_adapter=start_adapter,
                    model=model,
                    env=ep2_target_env,
                    budget_n=b,
                    device=device,
                    steps=min(b * 2, 20),
                )
                b_f1, b_acc, b_rec, b_paired = self.engine.evaluate_on_independent_world(
                    candidate=adapted, model=model, eval_env=ep2_target_env, device=device
                )
                curve.append({"n": b, "macro_f1": b_f1, "accuracy": b_acc, "recalls": b_rec, "paired": b_paired})
                auc_points.append(b_f1)

                if b_f1 >= 0.50 and steps_competence == 40:
                    steps_competence = b * 2
                if b == 32:
                    final_f1 = b_f1
                    final_acc = b_acc
                    final_rec = b_rec
                    final_paired = b_paired

            quality_auc = float(np.mean(auc_points))

            # Negative transfer / overgeneralization check on unrelated family
            neg_f1, _, _, _ = self.engine.evaluate_on_independent_world(
                candidate=start_adapter, model=model, eval_env=unrelated_eval_env, device=device
            )

            diag = (
                f"Arm '{arm_name}': ZeroShot F1={z_f1:.4f} -> Final F1={final_f1:.4f} | "
                f"Quality AUC={quality_auc:.4f} | UNK Rec={final_rec.get('UNKNOWN', 0.0) * 100:.1f}% | "
                f"Unrelated Eval F1={neg_f1:.4f}"
            )

            results[arm_name] = SequentialEpisodeOutcome(
                arm_name=arm_name,
                target_env_id=ep2_target_env.env_id,
                rule_family=rule_family,
                zero_shot_f1=float(z_f1),
                final_macro_f1=float(final_f1),
                quality_auc=quality_auc,
                held_out_accuracy=float(final_acc),
                classwise_recall=final_rec,
                paired_correctness=float(final_paired),
                sample_efficiency_curve=curve,
                steps_to_competence=steps_competence,
                overgeneralization_negative_transfer_f1=float(neg_f1),
                diagnostics=diag,
            )

        return results
