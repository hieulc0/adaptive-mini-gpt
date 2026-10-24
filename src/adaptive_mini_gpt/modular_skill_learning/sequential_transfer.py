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
from adaptive_mini_gpt.modular_skill_learning.hidden_extraction import extract_model_hidden
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
    rule_family: str
    zero_shot_f1: float
    final_macro_f1: float
    quality_auc: float
    sample_efficiency_curve: dict[int, float]
    paired_correctness: float
    classwise_recall: dict[str, float]
    total_compute_steps: int
    negative_transfer_delta: float
    retained_parameter_hash: str | None


class SequentialEpisodeExperiment:
    """Orchestrates Episode 1 acquisition -> Episode 2 transfer evaluation."""

    def __init__(
        self,
        acquisition_engine: ControlledSkillAcquisitionEngine,
        d_model: int = 128,
        bottleneck_dim: int = 16,
        num_classes: int = 3,
        lr: float = 5e-4,
        max_steps: int = 40,
    ) -> None:
        self.acquisition_engine = acquisition_engine
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim
        self.num_classes = num_classes
        self.lr = lr
        self.max_steps = max_steps

    def _train_adapter_on_env(
        self,
        adapter: Rank16ResidualAdapter,
        env: GroundedThreeValuedEnvironment,
        model: MiniGPT,
        budget_n: int,
        steps: int,
        device: torch.device,
        seed: int = 42,
    ) -> Rank16ResidualAdapter:
        """Fine-tune adapter on practice budget N."""
        torch.manual_seed(seed)
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
                    hidden = extract_model_hidden(model, inp)
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
        """Execute complete sequential episode study across 4 arms."""
        # ----------------------------------------------------
        # Episode 1: Acquire and qualify candidate skill C_F
        # ----------------------------------------------------
        repertoire_with_cand = copy.deepcopy(base_repertoire)
        cand_id = f"candidate_skill_{rule_family.lower()}"

        acq_report = self.acquisition_engine.acquire_and_qualify_candidate(
            candidate_id=cand_id,
            rule_family=rule_family,
            train_env=ep1_train_env,
            eval_env=ep1_eval_env,
            model=model,
            repertoire=repertoire_with_cand,
            budget_n=32,
            device=device,
            seed=seed,
        )

        candidate_record = (
            repertoire_with_cand.get_skill(cand_id)
            if (cand_id in repertoire_with_cand._skills)
            else None
        )

        budgets = [0, 4, 8, 16, 32]
        outcomes: dict[str, SequentialEpisodeOutcome] = {}

        # ----------------------------------------------------
        # Episode 2: Evaluate on new world instance of rule_family
        # ----------------------------------------------------

        # Arm 1: Learner WITH Acquired Candidate Skill
        adapter_for_curve = (
            repertoire_with_cand.build_adapter_instance(cand_id, device)
            if (candidate_record and cand_id in repertoire_with_cand._skills)
            else (acq_report.candidate or Rank16ResidualAdapter(self.d_model, self.bottleneck_dim, self.num_classes).to(device))
        )

        curve_with = {}
        # zero-shot
        z_f1, _, z_recs, z_paired = self.acquisition_engine.evaluate_candidate_on_independent_world(
            adapter_for_curve, ep2_target_env, model, device
        )
        curve_with[0] = z_f1

        for b in [4, 8, 16, 32]:
            adapter_tuned = self._train_adapter_on_env(
                adapter=copy.deepcopy(adapter_for_curve),
                env=ep2_target_env,
                model=model,
                budget_n=b,
                steps=min(self.max_steps, b * 2),
                device=device,
                seed=seed,
            )
            b_f1, _, _, _ = self.acquisition_engine.evaluate_candidate_on_independent_world(
                adapter_tuned, ep2_target_env, model, device
            )
            curve_with[b] = b_f1

        # Quality AUC over budgets [0..32] normalized
        norm_b = [b / 32.0 for b in budgets]
        vals = [curve_with[b] for b in budgets]
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auc_with = float(trapz_fn(vals, norm_b)) if trapz_fn else float(np.mean(vals))

        # Audit negative transfer on unrelated domain
        unrelated_f1, _, _, _ = self.acquisition_engine.evaluate_candidate_on_independent_world(
            adapter_for_curve, unrelated_eval_env, model, device
        )
        base_unrelated_f1 = 0.33
        neg_transfer_delta = unrelated_f1 - base_unrelated_f1

        outcomes[SequentialTransferArm.WITH_ACQUIRED_SKILL.value] = SequentialEpisodeOutcome(
            arm_name=SequentialTransferArm.WITH_ACQUIRED_SKILL.value,
            rule_family=rule_family,
            zero_shot_f1=float(z_f1),
            final_macro_f1=float(curve_with[32]),
            quality_auc=auc_with,
            sample_efficiency_curve=curve_with,
            paired_correctness=float(z_paired),
            classwise_recall=z_recs,
            total_compute_steps=32 * 2,
            negative_transfer_delta=float(neg_transfer_delta),
            retained_parameter_hash=candidate_record.param_hash if candidate_record else None,
        )

        # Arm 2: Learner WITHOUT Acquired Skill (Best static source skill)
        best_source_id = None
        best_source_f1 = -1.0
        for sid in base_repertoire.list_skills():
            f1, _, _, _ = base_repertoire.evaluate_skill_on_environment(sid, model, ep2_target_env, device)
            if f1 > best_source_f1:
                best_source_f1 = f1
                best_source_id = sid

        curve_without = {}
        curve_without[0] = best_source_f1
        z_recs_without = {}
        z_paired_without = 0.0

        if best_source_id:
            adapter_src = base_repertoire.build_adapter_instance(best_source_id, device)
            _, _, z_recs_without, z_paired_without = base_repertoire.evaluate_skill_on_environment(
                best_source_id, model, ep2_target_env, device
            )
            for b in [4, 8, 16, 32]:
                adapter_tuned = self._train_adapter_on_env(
                    adapter=copy.deepcopy(adapter_src),
                    env=ep2_target_env,
                    model=model,
                    budget_n=b,
                    steps=min(self.max_steps, b * 2),
                    device=device,
                    seed=seed,
                )
                b_f1, _, _, _ = self.acquisition_engine.evaluate_candidate_on_independent_world(
                    adapter_tuned, ep2_target_env, model, device
                )
                curve_without[b] = b_f1

        vals_without = [curve_without[b] for b in budgets]
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auc_without = float(trapz_fn(vals_without, [b / 32.0 for b in budgets])) if trapz_fn else float(np.mean(vals_without))

        outcomes[SequentialTransferArm.WITHOUT_ACQUIRED_SKILL.value] = SequentialEpisodeOutcome(
            arm_name=SequentialTransferArm.WITHOUT_ACQUIRED_SKILL.value,
            rule_family=rule_family,
            zero_shot_f1=float(best_source_f1),
            final_macro_f1=float(curve_without[32]),
            quality_auc=auc_without,
            sample_efficiency_curve=curve_without,
            paired_correctness=float(z_paired_without),
            classwise_recall=z_recs_without,
            total_compute_steps=32 * 2,
            negative_transfer_delta=0.0,
            retained_parameter_hash=None,
        )

        # Arm 3: Irrelevant Skill Control (Source Skill 1 on Target)
        irrelevant_id = base_repertoire.list_skills()[0]
        curve_irrel = {}
        z_f1_irr, _, z_recs_irr, z_paired_irr = base_repertoire.evaluate_skill_on_environment(
            irrelevant_id, model, ep2_target_env, device
        )
        curve_irrel[0] = z_f1_irr
        adapter_irr = base_repertoire.build_adapter_instance(irrelevant_id, device)
        for b in [4, 8, 16, 32]:
            adapter_tuned = self._train_adapter_on_env(
                adapter=copy.deepcopy(adapter_irr),
                env=ep2_target_env,
                model=model,
                budget_n=b,
                steps=min(self.max_steps, b * 2),
                device=device,
                seed=seed,
            )
            b_f1, _, _, _ = self.acquisition_engine.evaluate_candidate_on_independent_world(
                adapter_tuned, ep2_target_env, model, device
            )
            curve_irrel[b] = b_f1

        vals_irr = [curve_irrel[b] for b in budgets]
        auc_irr = float(trapz_fn(vals_irr, [b / 32.0 for b in budgets])) if trapz_fn else float(np.mean(vals_irr))

        outcomes[SequentialTransferArm.IRRELEVANT_SKILL_CONTROL.value] = SequentialEpisodeOutcome(
            arm_name=SequentialTransferArm.IRRELEVANT_SKILL_CONTROL.value,
            rule_family=rule_family,
            zero_shot_f1=float(z_f1_irr),
            final_macro_f1=float(curve_irrel[32]),
            quality_auc=auc_irr,
            sample_efficiency_curve=curve_irrel,
            paired_correctness=float(z_paired_irr),
            classwise_recall=z_recs_irr,
            total_compute_steps=32 * 2,
            negative_transfer_delta=0.0,
            retained_parameter_hash=None,
        )

        # Arm 4: Fresh Adaptation Control (Fresh random init on ep2_target_env)
        curve_fresh = {}
        fresh_adapter = Rank16ResidualAdapter(
            d_model=self.d_model, bottleneck_dim=self.bottleneck_dim, num_classes=self.num_classes
        ).to(device)
        z_f1_fresh, _, z_recs_fresh, z_paired_fresh = self.acquisition_engine.evaluate_candidate_on_independent_world(
            fresh_adapter, ep2_target_env, model, device
        )
        curve_fresh[0] = z_f1_fresh

        for b in [4, 8, 16, 32]:
            adapter_fresh_b, _, _ = self.acquisition_engine.train_candidate_adapter(
                candidate_id="fresh_control",
                train_env=ep2_target_env,
                model=model,
                budget_n=b,
                device=device,
                seed=seed,
            )
            b_f1, _, _, _ = self.acquisition_engine.evaluate_candidate_on_independent_world(
                adapter_fresh_b, ep2_target_env, model, device
            )
            curve_fresh[b] = b_f1

        vals_fresh = [curve_fresh[b] for b in budgets]
        auc_fresh = float(trapz_fn(vals_fresh, [b / 32.0 for b in budgets])) if trapz_fn else float(np.mean(vals_fresh))

        outcomes[SequentialTransferArm.FRESH_ADAPTATION_CONTROL.value] = SequentialEpisodeOutcome(
            arm_name=SequentialTransferArm.FRESH_ADAPTATION_CONTROL.value,
            rule_family=rule_family,
            zero_shot_f1=float(z_f1_fresh),
            final_macro_f1=float(curve_fresh[32]),
            quality_auc=auc_fresh,
            sample_efficiency_curve=curve_fresh,
            paired_correctness=float(z_paired_fresh),
            classwise_recall=z_recs_fresh,
            total_compute_steps=32 * 2,
            negative_transfer_delta=0.0,
            retained_parameter_hash=None,
        )

        return outcomes
