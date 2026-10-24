"""Phase 4: 9-Arm Causal Plan-Selection Benchmark Suite.

Evaluates:
- Arm A: AUTONOMOUS_PLAN_SEARCH (Inferred sequence from demonstrations)
- Arm B: PROVIDED_CORRECT_PLAN (Ground-truth plan provided; execution ceiling)
- Arm C: FIXED_ORDER (Static ordering: Reflect -> Shift)
- Arm D: RANDOM_VALID_PLAN (Uniform random valid composition)
- Arm E: SHUFFLED_DEMONSTRATIONS (Target-scrambled demonstrations)
- Arm F: SINGLE_SKILL (Best single primitive only)
- Arm G: WRONG_ORDER (Deliberately inverted sequence)
- Arm H: FRESH_COMPOUND_ADAPTER (50-step monolithic adapter training)
- Arm I: REFERENCE_ORACLE (Deterministic symbolic calculator diagnostic)
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.neural_plan_discovery.identifiability_tasks import (
    IdentifiableTaskDataset,
    PlanIdentifiabilityGenerator,
)
from adaptive_mini_gpt.neural_plan_discovery.plan_discovery_engine import (
    AutonomousPlanDiscoveryEngine,
    PlanSearchResult,
)
from adaptive_mini_gpt.neural_plan_discovery.plan_space import CandidatePlan, PlanSpace, PlanType
from adaptive_mini_gpt.neural_skill_composition.primitive_skills import (
    PrimitiveSkillType,
    QualifiedPrimitiveRecord,
)
from adaptive_mini_gpt.neural_weight_learning.pathway_instrumentation import (
    NeuralPathwayInstrumenter,
)
from adaptive_mini_gpt.neural_weight_learning.procedural_tasks import (
    ProceduralExample,
    ProceduralTaskDataset,
)
from adaptive_mini_gpt.neural_weight_learning.sequence_adapter import Rank16SequenceAdapter

logger = logging.getLogger(__name__)


@dataclass
class PlanArmResult:
    """Evaluation metrics for a single plan-selection arm."""

    arm_name: str
    description: str
    chosen_plan_stages: tuple[str, ...]
    is_plan_correct: bool
    is_abstained: bool
    plan_search_latency_ms: float
    forward_passes_used: int
    training_steps_incurred: int
    additional_parameters_allocated: int
    sealed_exact_match: float
    sealed_token_accuracy: float
    sealed_loss: float
    is_reference_oracle: bool = False


class CausalPlanBenchmark:
    """Executes the 9-arm causal plan discovery and execution benchmark."""

    @classmethod
    def execute_plan_on_test_set(
        cls,
        model: MiniGPT,
        plan: CandidatePlan,
        test_examples: list[Any],
        primitives: dict[PrimitiveSkillType, QualifiedPrimitiveRecord],
        device: torch.device,
    ) -> tuple[float, float, float]:
        """Execute a plan on the sealed test partition using real neural inference."""
        if plan.is_empty or len(test_examples) == 0:
            return 0.0, 0.0, 99.9

        inputs = [ex.input_tokens for ex in test_examples]
        targets = torch.tensor([list(ex.target_tokens) for ex in test_examples], dtype=torch.long, device=device)

        # Stage 1
        prim1 = primitives[PrimitiveSkillType(plan.stages[0])]
        logits1, preds1 = AutonomousPlanDiscoveryEngine.execute_single_primitive(model, prim1, inputs, device)

        if plan.depth == 1:
            final_logits = logits1
            final_preds = preds1
        elif plan.depth == 2:
            stage1_outputs = [tuple(p.tolist()) for p in preds1]
            prim2 = primitives[PrimitiveSkillType(plan.stages[1])]
            logits2, preds2 = AutonomousPlanDiscoveryEngine.execute_single_primitive(
                model, prim2, stage1_outputs, device
            )
            final_logits = logits2
            final_preds = preds2
        else:
            raise ValueError(f"Depth {plan.depth} not supported")

        token_acc = float((final_preds == targets).float().mean().item())
        seq_em = float((final_preds == targets).all(dim=-1).float().mean().item())
        loss = float(F.cross_entropy(final_logits.view(-1, 16), targets.view(-1)).item())

        return round(seq_em, 4), round(token_acc, 4), round(loss, 4)

    @classmethod
    def evaluate_all_arms(
        cls,
        model: MiniGPT,
        task: IdentifiableTaskDataset,
        primitives: dict[PrimitiveSkillType, QualifiedPrimitiveRecord],
        device: torch.device,
        seed: int = 42,
    ) -> dict[str, PlanArmResult]:
        """Evaluate task across all 9 causal plan-selection arms."""
        all_candidate_plans = PlanSpace.generate_candidate_plans(
            available_primitives=list(primitives.keys()),
            max_depth=2,
            include_abstain=True,
        )
        gt_stages = task.ground_truth_plan_stages

        results: dict[str, PlanArmResult] = {}

        # ---------------------------------------------------------------------
        # Arm A: AUTONOMOUS_PLAN_SEARCH (Demonstration-based inferred sequence)
        # ---------------------------------------------------------------------
        search_res = AutonomousPlanDiscoveryEngine.discover_plan(
            model=model,
            task_id=task.task_id,
            demonstrations=task.demonstrations,
            primitives=primitives,
            device=device,
            candidate_plans=all_candidate_plans,
        )
        em_a, acc_a, loss_a = cls.execute_plan_on_test_set(
            model=model,
            plan=search_res.selected_plan,
            test_examples=task.sealed_test_examples,
            primitives=primitives,
            device=device,
        )
        is_plan_corr_a = (search_res.selected_plan.stages == gt_stages)
        results["A_AUTONOMOUS_PLAN_SEARCH"] = PlanArmResult(
            arm_name="A_AUTONOMOUS_PLAN_SEARCH",
            description="Autonomous plan discovery from observable demonstrations",
            chosen_plan_stages=search_res.selected_plan.stages,
            is_plan_correct=is_plan_corr_a,
            is_abstained=search_res.is_abstained,
            plan_search_latency_ms=search_res.search_latency_ms,
            forward_passes_used=search_res.total_forward_passes,
            training_steps_incurred=0,
            additional_parameters_allocated=0,
            sealed_exact_match=em_a,
            sealed_token_accuracy=acc_a,
            sealed_loss=loss_a,
        )

        # ---------------------------------------------------------------------
        # Arm B: PROVIDED_CORRECT_PLAN (Ground-truth provided execution ceiling)
        # ---------------------------------------------------------------------
        provided_plan = next(
            (p for p in all_candidate_plans if p.stages == gt_stages),
            CandidatePlan("PLAN_ABSTAIN", PlanType.ABSTAIN, (), 0, "Abstain"),
        )
        em_b, acc_b, loss_b = cls.execute_plan_on_test_set(
            model=model,
            plan=provided_plan,
            test_examples=task.sealed_test_examples,
            primitives=primitives,
            device=device,
        )
        results["B_PROVIDED_CORRECT_PLAN"] = PlanArmResult(
            arm_name="B_PROVIDED_CORRECT_PLAN",
            description="Provided ground-truth plan (M76 execution ceiling)",
            chosen_plan_stages=provided_plan.stages,
            is_plan_correct=True,
            is_abstained=provided_plan.is_empty,
            plan_search_latency_ms=0.0,
            forward_passes_used=0,
            training_steps_incurred=0,
            additional_parameters_allocated=0,
            sealed_exact_match=em_b,
            sealed_token_accuracy=acc_b,
            sealed_loss=loss_b,
        )

        # ---------------------------------------------------------------------
        # Arm C: FIXED_ORDER (Static ordering: Reflect -> Shift)
        # ---------------------------------------------------------------------
        fixed_plan = next(
            p for p in all_candidate_plans
            if p.stages == (PrimitiveSkillType.COMPLEMENT_REFLECT.value, PrimitiveSkillType.CAESAR_SHIFT_MOD3.value)
        )
        em_c, acc_c, loss_c = cls.execute_plan_on_test_set(
            model=model,
            plan=fixed_plan,
            test_examples=task.sealed_test_examples,
            primitives=primitives,
            device=device,
        )
        results["C_FIXED_ORDER"] = PlanArmResult(
            arm_name="C_FIXED_ORDER",
            description="Fixed default ordering (Reflect -> Shift)",
            chosen_plan_stages=fixed_plan.stages,
            is_plan_correct=(fixed_plan.stages == gt_stages),
            is_abstained=False,
            plan_search_latency_ms=0.0,
            forward_passes_used=0,
            training_steps_incurred=0,
            additional_parameters_allocated=0,
            sealed_exact_match=em_c,
            sealed_token_accuracy=acc_c,
            sealed_loss=loss_c,
        )

        # ---------------------------------------------------------------------
        # Arm D: RANDOM_VALID_PLAN (Uniform random valid composition)
        # ---------------------------------------------------------------------
        two_stage_plans = [p for p in all_candidate_plans if p.depth == 2]
        rng = random.Random(seed + 101)
        rand_plan = rng.choice(two_stage_plans)
        em_d, acc_d, loss_d = cls.execute_plan_on_test_set(
            model=model,
            plan=rand_plan,
            test_examples=task.sealed_test_examples,
            primitives=primitives,
            device=device,
        )
        results["D_RANDOM_VALID_PLAN"] = PlanArmResult(
            arm_name="D_RANDOM_VALID_PLAN",
            description="Uniform random 2-stage composition",
            chosen_plan_stages=rand_plan.stages,
            is_plan_correct=(rand_plan.stages == gt_stages),
            is_abstained=False,
            plan_search_latency_ms=0.0,
            forward_passes_used=0,
            training_steps_incurred=0,
            additional_parameters_allocated=0,
            sealed_exact_match=em_d,
            sealed_token_accuracy=acc_d,
            sealed_loss=loss_d,
        )

        # ---------------------------------------------------------------------
        # Arm E: SHUFFLED_DEMONSTRATIONS (Target-scrambled demonstrations)
        # ---------------------------------------------------------------------
        shuffled_demos = PlanIdentifiabilityGenerator.create_shuffled_demonstrations(
            task.demonstrations, seed=seed + 202
        )
        shuff_res = AutonomousPlanDiscoveryEngine.discover_plan(
            model=model,
            task_id=f"shuffled_{task.task_id}",
            demonstrations=shuffled_demos,
            primitives=primitives,
            device=device,
            candidate_plans=all_candidate_plans,
        )
        em_e, acc_e, loss_e = cls.execute_plan_on_test_set(
            model=model,
            plan=shuff_res.selected_plan,
            test_examples=task.sealed_test_examples,
            primitives=primitives,
            device=device,
        )
        results["E_SHUFFLED_DEMONSTRATIONS"] = PlanArmResult(
            arm_name="E_SHUFFLED_DEMONSTRATIONS",
            description="Plan discovery under scrambled input-output demonstrations",
            chosen_plan_stages=shuff_res.selected_plan.stages,
            is_plan_correct=(shuff_res.selected_plan.stages == gt_stages),
            is_abstained=shuff_res.is_abstained,
            plan_search_latency_ms=shuff_res.search_latency_ms,
            forward_passes_used=shuff_res.total_forward_passes,
            training_steps_incurred=0,
            additional_parameters_allocated=0,
            sealed_exact_match=em_e,
            sealed_token_accuracy=acc_e,
            sealed_loss=loss_e,
        )

        # ---------------------------------------------------------------------
        # Arm F: SINGLE_SKILL (Best single primitive only)
        # ---------------------------------------------------------------------
        single_stage_plans = [p for p in all_candidate_plans if p.depth == 1]
        first_stage = gt_stages[0] if len(gt_stages) >= 1 else PrimitiveSkillType.COMPLEMENT_REFLECT.value
        single_plan = next((p for p in single_stage_plans if p.stages == (first_stage,)), single_stage_plans[0])
        em_f, acc_f, loss_f = cls.execute_plan_on_test_set(
            model=model,
            plan=single_plan,
            test_examples=task.sealed_test_examples,
            primitives=primitives,
            device=device,
        )
        results["F_SINGLE_SKILL"] = PlanArmResult(
            arm_name="F_SINGLE_SKILL",
            description="Single primitive execution only",
            chosen_plan_stages=single_plan.stages,
            is_plan_correct=(single_plan.stages == gt_stages),
            is_abstained=False,
            plan_search_latency_ms=0.0,
            forward_passes_used=0,
            training_steps_incurred=0,
            additional_parameters_allocated=0,
            sealed_exact_match=em_f,
            sealed_token_accuracy=acc_f,
            sealed_loss=loss_f,
        )

        # ---------------------------------------------------------------------
        # Arm G: WRONG_ORDER (Deliberately inverted sequence)
        # ---------------------------------------------------------------------
        if len(gt_stages) == 2:
            reversed_stages = (gt_stages[1], gt_stages[0])
            wrong_plan = next(
                (p for p in all_candidate_plans if p.stages == reversed_stages),
                CandidatePlan("PLAN_WRONG", PlanType.COMPOSED_TWO_STAGE, reversed_stages, 2, "Wrong order"),
            )
        else:
            wrong_plan = next(p for p in all_candidate_plans if p.depth == 2 and p.stages != gt_stages)

        em_g, acc_g, loss_g = cls.execute_plan_on_test_set(
            model=model,
            plan=wrong_plan,
            test_examples=task.sealed_test_examples,
            primitives=primitives,
            device=device,
        )
        results["G_WRONG_ORDER"] = PlanArmResult(
            arm_name="G_WRONG_ORDER",
            description="Deliberately inverted operator sequence",
            chosen_plan_stages=wrong_plan.stages,
            is_plan_correct=(wrong_plan.stages == gt_stages),
            is_abstained=False,
            plan_search_latency_ms=0.0,
            forward_passes_used=0,
            training_steps_incurred=0,
            additional_parameters_allocated=0,
            sealed_exact_match=em_g,
            sealed_token_accuracy=acc_g,
            sealed_loss=loss_g,
        )

        # ---------------------------------------------------------------------
        # Arm H: FRESH_COMPOUND_ADAPTER (50-step monolithic adapter training)
        # ---------------------------------------------------------------------
        # Convert demonstrations + dev examples into procedural training set
        train_examples = [
            ProceduralExample(
                example_id=f"mono_train_{i}",
                rule_family="COMPOUND_MONOLITHIC",
                input_tokens=ex.input_tokens,
                target_tokens=ex.target_tokens,
                input_text=ex.input_text,
                target_text=ex.target_text,
            )
            for i, ex in enumerate(task.demonstrations + task.development_examples)
        ]
        test_proc_examples = [
            ProceduralExample(
                example_id=f"mono_test_{i}",
                rule_family="COMPOUND_MONOLITHIC",
                input_tokens=ex.input_tokens,
                target_tokens=ex.target_tokens,
                input_text=ex.input_text,
                target_text=ex.target_text,
            )
            for i, ex in enumerate(task.sealed_test_examples)
        ]
        proc_dataset = ProceduralTaskDataset(
            task_id=f"mono_{task.task_id}",
            rule_family="COMPOUND_MONOLITHIC",
            seed=seed,
            train_examples=train_examples,
            dev_examples=train_examples[:8],
            qualification_examples=train_examples[:8],
            sealed_test_examples=test_proc_examples,
        )

        mono_adapter = Rank16SequenceAdapter(
            d_model=model.config.d_model, bottleneck_dim=16, num_symbols=16
        ).to(device)
        trained_mono, _ = NeuralPathwayInstrumenter.train_and_instrument(
            model=model,
            adapter=mono_adapter,
            task=proc_dataset,
            adapter_id=f"mono_{task.task_id}",
            device=device,
            steps=50,
            lr=3e-3,
        )

        test_h, test_y = NeuralPathwayInstrumenter.extract_symbol_hidden_states(
            model, test_proc_examples, device
        )
        trained_mono.eval()
        with torch.no_grad():
            m_logits = trained_mono(test_h)
            m_preds = torch.argmax(m_logits, dim=-1)
            em_h = float((m_preds == test_y).all(dim=-1).float().mean().item())
            acc_h = float((m_preds == test_y).float().mean().item())
            loss_h = float(F.cross_entropy(m_logits.view(-1, 16), test_y.view(-1)).item())

        results["H_FRESH_COMPOUND_ADAPTER"] = PlanArmResult(
            arm_name="H_FRESH_COMPOUND_ADAPTER",
            description="Fresh monolithic adapter trained 50 steps on task",
            chosen_plan_stages=("MONOLITHIC_ADAPTER",),
            is_plan_correct=False,  # Not a modular plan
            is_abstained=False,
            plan_search_latency_ms=0.0,
            forward_passes_used=50 * len(train_examples),
            training_steps_incurred=50,
            additional_parameters_allocated=mono_adapter.count_parameters(),
            sealed_exact_match=round(em_h, 4),
            sealed_token_accuracy=round(acc_h, 4),
            sealed_loss=round(loss_h, 4),
        )

        # ---------------------------------------------------------------------
        # Arm I: REFERENCE_ORACLE (Deterministic symbolic calculator)
        # ---------------------------------------------------------------------
        results["I_REFERENCE_ORACLE"] = PlanArmResult(
            arm_name="I_REFERENCE_ORACLE",
            description="Deterministic symbolic calculator diagnostic",
            chosen_plan_stages=gt_stages,
            is_plan_correct=True,
            is_abstained=False,
            plan_search_latency_ms=0.0,
            forward_passes_used=0,
            training_steps_incurred=0,
            additional_parameters_allocated=0,
            sealed_exact_match=1.0,
            sealed_token_accuracy=1.0,
            sealed_loss=0.0,
            is_reference_oracle=True,
        )

        return results
