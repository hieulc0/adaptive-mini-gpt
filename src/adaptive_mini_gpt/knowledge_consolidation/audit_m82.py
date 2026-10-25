"""Milestone M83 Phase 0: Forensic Audit and Reconciliation of Milestone M82.

Scientific Audit Objectives:
1. Reconcile M82 findings: step reduction (-33.3%), paired t(4) = 9.798 (p = 0.0006), controller parameter budget (1,835 params).
2. Audit Issue 1: Reconcile meta-training and validation task family overlap and enforce strictly disjoint family partitions in M83.
3. Audit Issue 2: Perform rigorous paired uncertainty analysis between Learned Controller and Random Policy across all 5 seeds, acknowledging high random variance.
4. Audit Issue 3: Address limited evaluation breadth by establishing 5 distinct relation regimes in M83.
5. Formalize acceptance criteria and research gates for Milestone M83 (Autonomous Neural Knowledge Consolidation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.learning_to_learn.statistics import (
    compute_paired_t_test,
    student_t_two_tailed_p_value,
)

logger = logging.getLogger(__name__)


@dataclass
class M82ForensicAuditReport:
    """Structured scientific audit report reconciling Milestone M82."""

    status: str
    verdict: str
    m82_reported_metrics: dict[str, Any]
    statistical_verification_m81_comparison: dict[str, Any]
    random_policy_paired_uncertainty_audit: dict[str, Any]
    curriculum_family_overlap_reconciliation: dict[str, Any]
    evaluation_breadth_gap_analysis: dict[str, Any]
    m83_target_gates: dict[str, Any]


class M82ForensicAuditor:
    """Forensic auditor validating M82 claims and defining M83 research gates."""

    M82_M81_EXP_STEPS = [45, 40, 60, 40, 60]  # Median: 45
    M82_LEARNED_STEPS = [30, 30, 45, 30, 50]  # Median: 30
    M82_RANDOM_STEPS = [70, 50, 70, 10, 10]   # Median: 50
    M82_FIXED_STEPS = [40, 55, 60, 55, 45]    # Median: 55

    @classmethod
    def execute_audit(cls) -> M82ForensicAuditReport:
        """Run forensic validation of M82 measurements and formalize M83 gates."""
        logger.info("Executing M82 Scientific Audit...")

        # 1. Step reduction M82 Learned vs M81 Experienced
        m81_median = 45
        m82_median = 30
        reduction_pct = (m81_median - m82_median) / m81_median * 100.0  # 33.33%

        # 2. Paired t-test vs M81 Experienced (df = 5 - 1 = 4)
        paired_m81 = compute_paired_t_test(cls.M82_M81_EXP_STEPS, cls.M82_LEARNED_STEPS)

        # 3. Audit Issue 2: Rigorous Paired Analysis vs Random Policy
        # Random - Learned differences: [+40, +20, +25, -20, -40]
        paired_random = compute_paired_t_test(cls.M82_RANDOM_STEPS, cls.M82_LEARNED_STEPS)

        # 4. Audit Issue 1: Task family overlap reconciliation
        curriculum_family_reconciliation = {
            "m82_overlap_observation": (
                "In M82's early draft, MODULAR_NONLINEAR was listed in both meta-train and meta-val task definitions. "
                "Although the held-out meta-test family (TWO_STAGE_CASCADE) was completely separated, the meta-val set "
                "did not satisfy strict mathematical disjointness from meta-train."
            ),
            "m83_enforcement": (
                "Milestone M83 strictly partitions task families into mutually exclusive mathematical sets:\n"
                "- Consolidation Source Families (Arithmetic Affine, Modular Scaling, Parity Logic)\n"
                "- Partial-Overlap Families (Compound Affine-Logic Cascades)\n"
                "- Unrelated Families (Symbolic Cyclic Shifts and Block Transpositions)\n"
                "- Conflicting Families (Opposing Polarity Branching Maps)\n"
                "- Future Transfer Families (Unseen Cubic Polynomial Transformations)"
            ),
        }

        # 5. Audit Issue 3: Evaluation breadth gap analysis
        evaluation_breadth_gap = {
            "m82_limitation": (
                "M82 evaluated a single held-out family (TWO_STAGE_CASCADE). While demanding, evaluation on a single "
                "composite rule cannot establish general-purpose adaptive competence."
            ),
            "m83_expansion": (
                "M83 evaluates consolidation and distillation across five distinct relational regimes: "
                "Highly Related, Partially Overlapping, Unrelated, Conflicting, and Future Skill Learning."
            ),
        }

        # 6. M83 Target Gates
        m83_target_gates = {
            "gate_parameter_compression": (
                "At least 20% reduction in total stored adapter parameters on a redundant-skill curriculum."
            ),
            "gate_skill_retention": (
                "At least 95% of baseline performance preserved across consolidated skills, "
                "with an absolute floor of >= 85% exact match on every individual skill."
            ),
            "gate_compound_composition": (
                "Consolidated skills must participate in multi-stage sequential compositions with no material regression."
            ),
            "gate_future_learning_efficiency": (
                "Consolidated representations must not degrade subsequent skill acquisition speed "
                "(median training steps <= M82 baseline)."
            ),
            "gate_learned_policy_superiority": (
                "Learned consolidation policy must statistically outperform direct weight averaging, "
                "random merging, and naive low-rank SVD compression."
            ),
            "gate_bit_exact_rollback": (
                "Bit-exact parameter restoration when consolidation proposals fail qualification gates."
            ),
            "gate_lineage_preservation": (
                "Base model parameters strictly 100% frozen (0 growth, SHA-256 hash invariant preserved) "
                "with 100% retention on core arithmetic carry (20/20) and operational logic (8/8)."
            ),
        }

        return M82ForensicAuditReport(
            status="M82_AUDIT_COMPLETED",
            verdict="PASS_RECONCILED_WITH_M83_CORRECTIONS",
            m82_reported_metrics={
                "m81_experienced_median_steps": m81_median,
                "m82_learned_controller_median_steps": m82_median,
                "relative_step_reduction_pct": round(reduction_pct, 2),
                "controller_parameters": 1835,
                "regression_suite_tests": 345,
            },
            statistical_verification_m81_comparison={
                "t_stat": round(paired_m81.t_stat, 4),
                "df": paired_m81.df,
                "p_value": round(paired_m81.p_value, 4),
                "ci_95": (round(paired_m81.ci_95_lower, 2), round(paired_m81.ci_95_upper, 2)),
                "is_significant_01": paired_m81.is_significant_01,
            },
            random_policy_paired_uncertainty_audit={
                "m82_steps_per_seed": cls.M82_LEARNED_STEPS,
                "random_steps_per_seed": cls.M82_RANDOM_STEPS,
                "mean_step_difference_random_minus_learned": round(paired_random.delta_mean, 2),
                "std_step_difference": round(paired_random.std_diff, 2),
                "t_stat": round(paired_random.t_stat, 4),
                "df": paired_random.df,
                "p_value": round(paired_random.p_value, 4),
                "ci_95": (round(paired_random.ci_95_lower, 2), round(paired_random.ci_95_upper, 2)),
                "is_significant_05": paired_random.is_significant_05,
                "audit_verdict": (
                    "CONFIRMED_AUDIT_FINDING: While Learned Controller exhibits superior worst-case stability "
                    "(0% timeout failure rate vs 40% for Random) and a substantially lower median (30 vs 50), "
                    "the paired Student-t test against Random Policy over 5 seeds is NOT statistically significant "
                    f"(p = {round(paired_random.p_value, 4)}) due to extreme random exploration variance."
                ),
            },
            curriculum_family_overlap_reconciliation=curriculum_family_reconciliation,
            evaluation_breadth_gap_analysis=evaluation_breadth_gap,
            m83_target_gates=m83_target_gates,
        )
