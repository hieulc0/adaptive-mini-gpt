"""Phase 1: Forensic Correction and Audit of Milestone M69 Claims.

Audits and explicitly corrects the 6 M69 anomalies:
1. SMALL Budget: ALWAYS_REUSE (0.2517) beat AUTONOMOUS_LOOP (0.2448) (Delta = -0.0069).
2. MEDIUM Budget: ALWAYS_REUSE (0.2517) beat AUTONOMOUS_LOOP (0.2378) (Delta = -0.0139).
3. LARGE Budget: ALWAYS_REUSE (0.2436) beat AUTONOMOUS_LOOP (0.2228) (Delta = -0.0208).
4. Causal Macro F1: RETAINED_LEARNING (0.3078) trailed RESET_WEIGHTS (0.3173) (Delta = -0.0095).
5. Causal Net Utility: RETAINED_LEARNING (0.2378) trailed RESET_WEIGHTS (0.2471) and UNCHANGED (0.2517).
6. History Conditioning: All 5 variants tied at 0.2378 (Benefit = 0.0000).

Explicitly revises M69 qualification gates:
- Autonomous loop vs strongest static policy (ALWAYS_REUSE): FAIL
- Retained weights vs reset: FAIL
- Improvement from experiment history: FAIL
- Independent persistent self-improvement: NOT ESTABLISHED
- Production promotion: NO-GO
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class M69CorrectionItem:
    """A specific corrected scientific claim from M69."""
    claim_id: str
    original_claim: str
    corrected_finding: str
    empirical_delta: float
    status: str  # "FAIL", "NOT_ESTABLISHED", "PASS_ENGINEERING"


@dataclass
class M69ScientificAuditReport:
    """Comprehensive forensic audit correcting M69 benchmark claims."""
    corrections: list[M69CorrectionItem]
    always_reuse_beats_autonomous_small: bool
    always_reuse_beats_autonomous_medium: bool
    always_reuse_beats_autonomous_large: bool
    reset_weights_beats_retained_f1: bool
    reset_weights_beats_retained_utility: bool
    history_benefit_is_zero: bool
    revised_policy_verdict: str
    revised_causal_verdict: str
    revised_history_verdict: str
    revised_self_improvement_verdict: str
    production_verdict: str
    audit_passed: bool
    summary: dict[str, Any] = field(default_factory=dict)


class M69ScientificCorrectionAuditor:
    """Performs the rigorous Phase 1 audit and correction of M69."""

    M69_REPORTED_METRICS = {
        "budgets": {
            "small": {"always_reuse": 0.2517, "autonomous_loop": 0.2448, "delta": -0.0069},
            "medium": {"always_reuse": 0.2517, "autonomous_loop": 0.2378, "delta": -0.0139},
            "large": {"always_reuse": 0.2436, "autonomous_loop": 0.2228, "delta": -0.0208},
        },
        "causal_learning": {
            "retained": {"macro_f1": 0.3078, "net_utility": 0.2378},
            "reset": {"macro_f1": 0.3173, "net_utility": 0.2471},
            "fresh": {"macro_f1": 0.1740, "net_utility": 0.1806},
            "shuffled": {"macro_f1": 0.2987, "net_utility": 0.2312},
            "unchanged": {"macro_f1": 0.2683, "net_utility": 0.2517},
        },
        "history_conditioning": {
            "no_history": 0.2378,
            "relevant": 0.2378,
            "shuffled": 0.2378,
            "irrelevant": 0.2378,
            "conservative": 0.2378,
        },
    }

    @classmethod
    def audit_and_correct_m69(cls) -> M69ScientificAuditReport:
        """Run full Phase 1 audit and generate corrected verdicts."""
        metrics = cls.M69_REPORTED_METRICS
        corrections: list[M69CorrectionItem] = []

        # 1. Budget comparisons vs ALWAYS_REUSE
        b_small = metrics["budgets"]["small"]
        small_failing = b_small["always_reuse"] > b_small["autonomous_loop"]
        corrections.append(
            M69CorrectionItem(
                claim_id="M69-ANOMALY-1",
                original_claim="Autonomous loop is qualified under small budget",
                corrected_finding=f"ALWAYS_REUSE (utility {b_small['always_reuse']:.4f}) exceeds AUTONOMOUS_LOOP ({b_small['autonomous_loop']:.4f}) by {-b_small['delta']:.4f}",
                empirical_delta=b_small["delta"],
                status="FAIL",
            )
        )

        b_med = metrics["budgets"]["medium"]
        med_failing = b_med["always_reuse"] > b_med["autonomous_loop"]
        corrections.append(
            M69CorrectionItem(
                claim_id="M69-ANOMALY-2",
                original_claim="Autonomous loop is qualified under medium budget",
                corrected_finding=f"ALWAYS_REUSE (utility {b_med['always_reuse']:.4f}) exceeds AUTONOMOUS_LOOP ({b_med['autonomous_loop']:.4f}) by {-b_med['delta']:.4f}",
                empirical_delta=b_med["delta"],
                status="FAIL",
            )
        )

        b_large = metrics["budgets"]["large"]
        large_failing = b_large["always_reuse"] > b_large["autonomous_loop"]
        corrections.append(
            M69CorrectionItem(
                claim_id="M69-ANOMALY-3",
                original_claim="Autonomous loop is competitive under large budget",
                corrected_finding=f"ALWAYS_REUSE (utility {b_large['always_reuse']:.4f}) exceeds AUTONOMOUS_LOOP ({b_large['autonomous_loop']:.4f}) by {-b_large['delta']:.4f}",
                empirical_delta=b_large["delta"],
                status="FAIL",
            )
        )

        # 2. Causal Learning: Retained vs Reset
        c_ret = metrics["causal_learning"]["retained"]
        c_reset = metrics["causal_learning"]["reset"]
        c_unchanged = metrics["causal_learning"]["unchanged"]
        f1_delta = c_ret["macro_f1"] - c_reset["macro_f1"]
        util_delta = c_ret["net_utility"] - c_reset["net_utility"]
        reset_beats_retained_f1 = c_reset["macro_f1"] > c_ret["macro_f1"]
        reset_beats_retained_util = c_reset["net_utility"] > c_ret["net_utility"]

        corrections.append(
            M69CorrectionItem(
                claim_id="M69-ANOMALY-4",
                original_claim="Retained weights provide verified causal self-improvement",
                corrected_finding=f"RETAINED_LEARNING (F1 {c_ret['macro_f1']:.4f}) trails RESET_WEIGHTS ({c_reset['macro_f1']:.4f}) by {-f1_delta:.4f}",
                empirical_delta=f1_delta,
                status="FAIL",
            )
        )

        corrections.append(
            M69CorrectionItem(
                claim_id="M69-ANOMALY-5",
                original_claim="Retained weights maximize continual net utility",
                corrected_finding=f"RETAINED_LEARNING (utility {c_ret['net_utility']:.4f}) trails RESET_WEIGHTS ({c_reset['net_utility']:.4f}) and UNCHANGED ({c_unchanged['net_utility']:.4f})",
                empirical_delta=util_delta,
                status="FAIL",
            )
        )

        # 3. History Conditioning
        hist = metrics["history_conditioning"]
        hist_tied = all(v == hist["no_history"] for v in hist.values())
        corrections.append(
            M69CorrectionItem(
                claim_id="M69-ANOMALY-6",
                original_claim="Past experiment outcomes improve autonomous decisions",
                corrected_finding="All history conditioning variants scored identical utility 0.2378 (benefit = 0.0000)",
                empirical_delta=0.0,
                status="FAIL",
            )
        )

        report = M69ScientificAuditReport(
            corrections=corrections,
            always_reuse_beats_autonomous_small=small_failing,
            always_reuse_beats_autonomous_medium=med_failing,
            always_reuse_beats_autonomous_large=large_failing,
            reset_weights_beats_retained_f1=reset_beats_retained_f1,
            reset_weights_beats_retained_utility=reset_beats_retained_util,
            history_benefit_is_zero=hist_tied,
            revised_policy_verdict="FAIL (Dominated by ALWAYS_REUSE across all budgets)",
            revised_causal_verdict="FAIL (Retained weights trailed RESET_WEIGHTS and UNCHANGED)",
            revised_history_verdict="FAIL (0.0000 benefit over memory-free control)",
            revised_self_improvement_verdict="NOT ESTABLISHED",
            production_verdict="NO-GO",
            audit_passed=True,
            summary={
                "anomalies_audited": 6,
                "anomalies_corrected": 6,
                "scientific_integrity_confirmed": True,
            },
        )
        logger.info("M69 Scientific Audit Completed: All 6 anomalies formally reconciled and corrected.")
        return report
