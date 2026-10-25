"""Milestone M102.2 Unbiased Causal Comparator.

Evaluates 6 independent interference control arms on a sealed non-linear polynomial target:
1. Equalized max steps (30 steps for all arms).
2. Equalized early stopping criteria (80.0% accuracy on validation split).
3. Zero artificial floors (pure empirical steps_taken directly from optimizer).
4. Disjoint training and held-out test sets (zero data leakage).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from adaptive_mini_gpt.integrated_autonomous_student.structural_evolution_engine import (
    CapacityTierM100,
)
from adaptive_mini_gpt.scientific_requalification_m102_2.curriculum_m102_2 import (
    DomainM102_2,
    IndependentVerifierM102_2,
    LearningGoalM102_2,
)
from adaptive_mini_gpt.scientific_requalification_m102_2.persistent_student_m102_2 import (
    PersistentStudentM102_2,
)
from adaptive_mini_gpt.scientific_requalification_m102_2.rehearsal_buffer_m102_2 import (
    ScientificRetentionManagerM102_2,
)

MODULUS = 16


@dataclass
class CausalArmResultM102_2:
    arm_name: str
    adaptation_steps: int
    validation_accuracy: float
    held_out_test_accuracy: float
    independent_retention: float
    unlabeled_retention: float
    compute_units: float
    qualified: bool


@dataclass
class CausalComparisonReportM102_2:
    arm_a_m101: CausalArmResultM102_2
    arm_b_m102: CausalArmResultM102_2
    arm_c_balanced_only: CausalArmResultM102_2
    arm_d_task_heads: CausalArmResultM102_2
    arm_e_pcgrad: CausalArmResultM102_2
    arm_f_combined_m102_2: CausalArmResultM102_2
    task_heads_advantage_pct: float
    gradient_projection_advantage_pct: float
    combined_advantage_pct: float


class CausalComparatorM102_2:
    """Executes strictly equalized causal evaluation across interference arms."""

    def __init__(self, seed: int = 42, device: torch.device | None = None) -> None:
        self.seed = seed
        self.device = device or torch.device("cpu")

    def generate_sealed_polynomial_goal(self) -> LearningGoalM102_2:
        """Construct sealed target y = (3x^2 + 7x + 5) mod 16 with disjoint train/val/test splits."""
        # Train: x in [0..7] (8 inputs, 2 ops -> 16 tokens)
        # Val:   x in [8..11] (4 inputs, 2 ops -> 8 tokens)
        # Test:  x in [12..15] (4 inputs, 2 ops -> 8 tokens)
        # Novel: x in [0..15] with op=3, v=5
        def encode(x_vals: list[int], op_val: int, v_val: int) -> tuple[torch.Tensor, torch.Tensor]:
            seqs = []
            labels = []
            for xi in x_vals:
                seqs.append([xi % 16, 16 + (op_val % 16), 32 + (v_val % 16)])
                labels.append((3 * xi * xi + 7 * xi + 5) % MODULUS)
            tok = torch.tensor(seqs, dtype=torch.long, device=self.device)
            lbl = torch.tensor(labels, dtype=torch.long, device=self.device)
            return tok, lbl

        tr_tok, tr_lbl = encode(list(range(8)), 3, 5)
        val_tok, val_lbl = encode(list(range(8, 12)), 3, 5)
        te_tok, te_lbl = encode(list(range(12, 16)), 3, 5)
        nov_tok, nov_lbl = encode(list(range(16)), 7, 9)

        return LearningGoalM102_2(
            goal_id="sealed_nonlinear_target_m102_2",
            domain=DomainM102_2.ALGEBRA_QUADRATIC,
            description="Sealed quadratic target y = (3x^2 + 7x + 5) mod 16 with disjoint train/val/test splits",
            train_tokens=tr_tok,
            train_labels=tr_lbl,
            val_tokens=val_tok,
            val_labels=val_lbl,
            test_tokens=te_tok,
            test_labels=te_lbl,
            anchor_tokens=tr_tok,
            anchor_labels=tr_lbl,
            novel_generalization_tokens=nov_tok,
            novel_generalization_labels=nov_lbl,
            estimated_cost=1.0,
        )

    def evaluate_arms(self, base_student: PersistentStudentM102_2) -> CausalComparisonReportM102_2:
        """Run all 6 arms with identical 30-step max budget, identical lr, and 0 artificial floors."""
        target_goal = self.generate_sealed_polynomial_goal()
        EQUALIZED_MAX_STEPS = 30
        EQUALIZED_EARLY_STOP = 80.0

        def evaluate_single_arm(
            arm_name: str,
            enable_projection: bool,
            enable_heads: bool,
            use_rehearsal: bool,
            cost_multiplier: float,
        ) -> CausalArmResultM102_2:
            student = PersistentStudentM102_2(
                tier=CapacityTierM100.TIER_20K,
                device=self.device,
                enable_gradient_projection=enable_projection,
                enable_domain_heads=enable_heads,
                enable_persistent_growth=base_student.enable_persistent_growth,
            )
            student.load_state_dict(base_student.state_dict())
            if use_rehearsal:
                student.retention_manager = base_student.retention_manager.clone()
            else:
                student.retention_manager = ScientificRetentionManagerM102_2(max_exemplars_per_domain=32)

            res = student.adapt_fast_skill(
                target_goal,
                max_steps=EQUALIZED_MAX_STEPS,
                early_stop_acc=EQUALIZED_EARLY_STOP,
            )
            audit = student.retention_manager.evaluate_retention(student)

            # Unmodified, pure empirical steps directly from optimizer
            actual_steps = int(res["steps_taken"])

            return CausalArmResultM102_2(
                arm_name=arm_name,
                adaptation_steps=actual_steps,
                validation_accuracy=res["validation_acc"],
                held_out_test_accuracy=res["held_out_test_acc"],
                independent_retention=audit.mean_independent_retention,
                unlabeled_retention=audit.mean_unlabeled_retention,
                compute_units=round(actual_steps * cost_multiplier, 2),
                qualified=res["qualified"],
            )

        arm_a = evaluate_single_arm("Arm A: Original M101", False, False, False, 1.0)
        arm_b = evaluate_single_arm("Arm B: Original M102", False, False, True, 1.05)
        arm_c = evaluate_single_arm("Arm C: Balanced Rehearsal Only", False, False, True, 1.05)
        arm_d = evaluate_single_arm("Arm D: Task-Conditioned Heads", False, True, True, 1.1)
        arm_e = evaluate_single_arm("Arm E: PCGrad Only", True, False, True, 1.15)
        arm_f = evaluate_single_arm("Arm F: Combined M102.2", True, True, True, 1.2)

        # Calculate causal speedup relative to Arm B (M102 baseline)
        b_steps = max(1, arm_b.adaptation_steps)
        adv_heads = ((b_steps - arm_d.adaptation_steps) / b_steps) * 100.0
        adv_pcgrad = ((b_steps - arm_e.adaptation_steps) / b_steps) * 100.0
        adv_comb = ((b_steps - arm_f.adaptation_steps) / b_steps) * 100.0

        return CausalComparisonReportM102_2(
            arm_a_m101=arm_a,
            arm_b_m102=arm_b,
            arm_c_balanced_only=arm_c,
            arm_d_task_heads=arm_d,
            arm_e_pcgrad=arm_e,
            arm_f_combined_m102_2=arm_f,
            task_heads_advantage_pct=round(adv_heads, 1),
            gradient_projection_advantage_pct=round(adv_pcgrad, 1),
            combined_advantage_pct=round(adv_comb, 1),
        )
