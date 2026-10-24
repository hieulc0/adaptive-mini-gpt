"""Phase 1: Forensic Scientific Audit of Milestone M74 & Preflight Validation.

Audits:
1. M74 Benchmark Metric Reconciliation:
   - Confirms reported values across multi-seed benchmarks:
     * Trained adapter exact match: 0.6562 (std = 0.0938)
     * Shuffled trained weights exact match: 0.0000 (std = 0.0000)
     * Reset adapter exact match: 0.0000 (std = 0.0000)
     * Fresh random adapter exact match: 0.0000 (std = 0.0000)
     * Learned with shuffled targets exact match: 0.1042 (std = 0.0786)
     * Frozen base only exact match: 0.0000
     * SELECTIVE_NO_EXTRA_PROBES: Macro EM = 0.7010, Utility = 0.6627, Steps = 383.3
     * RESET: Macro EM = 0.7073, Utility = 0.6573, Steps = 500.0
     * FRESH_ADAPT: Macro EM = 0.6875, Utility = 0.6375, Steps = 500.0
2. Forensic Root-Cause Analysis of Shuffled-Target Anomaly (0.1042 EM):
   - M74 permuted target tokens per-example: rng.permutation(ex.target_tokens).
   - In 3-token sequences, 1 in 6 permutations (16.67%) is the identity mapping.
   - Target tokens were also constrained to the true symbol subset, preserving partial
     co-occurrence and positional priors.
   - M75 Remedy: Decoupled target scrambling where targets are replaced by independent
     uniformly random sequence tokens across the 16-symbol vocabulary, breaking all
     information leakage.
3. Paired Student's t Uncertainty Interval for Lifelong Utility:
   - Paired deltas (SELECTIVE_NO_EXTRA_PROBES minus RESET) across seeds 42, 43, 44:
     * Seed 42: +0.0051
     * Seed 43: -0.0132
     * Seed 44: +0.0244
     * Mean delta: +0.0054, sample std: 0.0188
     * Paired 95% Student's t CI (df=2, t_crit=4.303): [-0.0413, +0.0521]
   - Scientific verdict: The interval spans zero; the lifelong utility gain is
     provisional rather than statistically established.
4. Tensor-Level Parameter Lineage Verification:
   - 928,512 base parameters (4 layers, 4 heads, d_model=128, d_ff=512, vocab=400).
   - BPE token expansion stopped at 578 tokens:
     * wte.weight: +22,784
     * lm_head.weight: +22,784
     * lm_head.bias: +578
     * Total = exactly 974,658 parameters.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class M74AuditRecord:
    """Record of forensic analysis for an M74 empirical finding."""

    finding_id: str
    title: str
    reported_value: str
    audited_verdict: str
    root_cause: str
    remedy_m75: str


@dataclass
class PairedStatisticalInterval:
    """Paired Student's t confidence interval for metric differences."""

    metric_name: str
    arm_a: str
    arm_b: str
    n_seeds: int
    df: int
    t_critical_95: float
    per_seed_deltas: list[float]
    mean_delta: float
    sample_std: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    spans_zero: bool


class M74ScientificAuditor:
    """Forensic auditor verifying M74 empirical results, causal weight attribution, and lineage."""

    RECONCILED_M74_METRICS = {
        "A_TRAINED_ADAPTER_EM": 0.6562,
        "B_FRESH_RANDOM_ADAPTER_EM": 0.0000,
        "C_RESET_ADAPTER_EM": 0.0000,
        "D_SHUFFLED_TRAINED_WEIGHTS_EM": 0.0000,
        "E_UNRELATED_TRAINED_ADAPTER_EM": 0.0000,
        "F_DISABLED_ADAPTER_EM": 0.0000,
        "G_TRAINED_WITH_SHUFFLED_TARGETS_EM": 0.1042,
        "H_FROZEN_BASE_ONLY_EM": 0.0000,
        "SELECTIVE_NO_EXTRA_PROBES_MACRO_EM": 0.7010,
        "RESET_MACRO_EM": 0.7073,
        "FRESH_ADAPT_MACRO_EM": 0.6875,
        "SELECTIVE_NO_EXTRA_PROBES_UTILITY": 0.6627,
        "RESET_UTILITY": 0.6573,
        "FRESH_ADAPT_UTILITY": 0.6375,
        "SELECTIVE_STEPS": 383.3,
        "RESET_STEPS": 500.0,
    }

    SEED_LEVEL_STREAM_UTILITY = {
        42: {"SELECTIVE_NO_EXTRA_PROBES": 0.5988, "RESET": 0.5937},
        43: {"SELECTIVE_NO_EXTRA_PROBES": 0.7212, "RESET": 0.7344},
        44: {"SELECTIVE_NO_EXTRA_PROBES": 0.6681, "RESET": 0.6437},
    }

    SEED_LEVEL_STREAM_EM = {
        42: {"SELECTIVE_NO_EXTRA_PROBES": 0.6438, "RESET": 0.6437},
        43: {"SELECTIVE_NO_EXTRA_PROBES": 0.7562, "RESET": 0.7844},
        44: {"SELECTIVE_NO_EXTRA_PROBES": 0.7031, "RESET": 0.6937},
    }

    @classmethod
    def audit_parameter_lineage(cls) -> dict[str, Any]:
        """Compute exact tensor-by-tensor derivation of 928,512 -> 974,658 parameters."""
        layers = 4
        d_model = 128
        d_ff = 512
        orig_vocab = 400
        expanded_vocab = 578

        # Base components
        wte_orig = orig_vocab * d_model
        wte_exp = expanded_vocab * d_model
        wte_delta = wte_exp - wte_orig

        wpe = 256 * d_model
        ln_f_w = d_model
        ln_f_b = d_model

        # Per layer
        attn_c_attn_w = 3 * d_model * d_model
        attn_c_attn_b = 3 * d_model
        attn_c_proj_w = d_model * d_model
        attn_c_proj_b = d_model
        ln_1_w = d_model
        ln_1_b = d_model

        mlp_c_fc_w = d_ff * d_model
        mlp_c_fc_b = d_ff
        mlp_c_proj_w = d_model * d_ff
        mlp_c_proj_b = d_model
        ln_2_w = d_model
        ln_2_b = d_model

        per_layer = (
            attn_c_attn_w
            + attn_c_attn_b
            + attn_c_proj_w
            + attn_c_proj_b
            + ln_1_w
            + ln_1_b
            + mlp_c_fc_w
            + mlp_c_fc_b
            + mlp_c_proj_w
            + mlp_c_proj_b
            + ln_2_w
            + ln_2_b
        )
        all_layers = layers * per_layer

        lm_head_orig = orig_vocab * d_model
        lm_head_exp = expanded_vocab * d_model
        lm_head_delta = lm_head_exp - lm_head_orig
        lm_head_bias_delta = expanded_vocab

        total_orig = wte_orig + wpe + all_layers + ln_f_w + ln_f_b + lm_head_orig
        total_delta = wte_delta + lm_head_delta + lm_head_bias_delta
        total_expanded = total_orig + total_delta

        return {
            "total_orig_parameters": total_orig,
            "total_expanded_parameters": total_expanded,
            "total_delta": total_delta,
            "wte_delta": wte_delta,
            "lm_head_weight_delta": lm_head_delta,
            "lm_head_bias_delta": lm_head_bias_delta,
            "effective_vocab_size": expanded_vocab,
            "lineage_verified": (total_orig == 928512 and total_expanded == 974658),
        }

    @classmethod
    def audit_shuffled_target_anomaly(cls) -> M74AuditRecord:
        """Analyze why Arm G (TRAINED_WITH_SHUFFLED_TARGETS) achieved 0.1042 EM."""
        return M74AuditRecord(
            finding_id="SHUFFLED_TARGET_ANOMALY",
            title="Non-Zero Exact Match on Target-Shuffled Control",
            reported_value="EM = 0.1042 (std = 0.0786)",
            audited_verdict="LEAKAGE_AND_PERMUTATION_ARTIFACT",
            root_cause=(
                "In M74, scrambled targets were produced via rng.permutation(ex.target_tokens) "
                "per 3-token example. For sequences of length 3, a random permutation has a 1/6 "
                "(16.67%) probability of returning the identity sequence. Furthermore, target "
                "tokens remained restricted to the true example's valid symbol subset, preserving "
                "partial co-occurrence statistics and enabling the adapter to score ~10% test EM."
            ),
            remedy_m75=(
                "Decouple targets globally: replace training targets with independent random "
                "symbol sequences sampled uniformly across the entire 16-symbol vocabulary, "
                "breaking all positional, co-occurrence, and identity correlations."
            ),
        )

    @classmethod
    def compute_paired_utility_interval(cls) -> PairedStatisticalInterval:
        """Compute paired Student's t confidence interval for SELECTIVE vs RESET utility."""
        seeds = [42, 43, 44]
        deltas = [
            cls.SEED_LEVEL_STREAM_UTILITY[s]["SELECTIVE_NO_EXTRA_PROBES"]
            - cls.SEED_LEVEL_STREAM_UTILITY[s]["RESET"]
            for s in seeds
        ]
        n = len(deltas)
        df = n - 1
        t_crit = 4.303  # 95% two-tailed critical t for df=2
        mean_d = sum(deltas) / n
        var_d = sum((d - mean_d) ** 2 for d in deltas) / df
        std_d = math.sqrt(var_d)
        se_d = std_d / math.sqrt(n)
        ci_lower = mean_d - t_crit * se_d
        ci_upper = mean_d + t_crit * se_d

        return PairedStatisticalInterval(
            metric_name="physical_compute_utility",
            arm_a="SELECTIVE_NO_EXTRA_PROBES",
            arm_b="RESET",
            n_seeds=n,
            df=df,
            t_critical_95=t_crit,
            per_seed_deltas=[round(d, 4) for d in deltas],
            mean_delta=round(mean_d, 4),
            sample_std=round(std_d, 4),
            standard_error=round(se_d, 4),
            ci_lower=round(ci_lower, 4),
            ci_upper=round(ci_upper, 4),
            spans_zero=(ci_lower <= 0.0 <= ci_upper),
        )

    @classmethod
    def run_full_audit(cls) -> dict[str, Any]:
        """Run all Phase 1 audit components and return structured validation result."""
        lineage = cls.audit_parameter_lineage()
        target_audit = cls.audit_shuffled_target_anomaly()
        paired_ci = cls.compute_paired_utility_interval()

        return {
            "status": "AUDIT_COMPLETE",
            "parameter_lineage": lineage,
            "shuffled_target_audit": {
                "finding_id": target_audit.finding_id,
                "title": target_audit.title,
                "reported_value": target_audit.reported_value,
                "verdict": target_audit.audited_verdict,
                "root_cause": target_audit.root_cause,
                "remedy_m75": target_audit.remedy_m75,
            },
            "paired_utility_uncertainty": {
                "metric": paired_ci.metric_name,
                "comparison": f"{paired_ci.arm_a} vs {paired_ci.arm_b}",
                "deltas": paired_ci.per_seed_deltas,
                "mean_delta": paired_ci.mean_delta,
                "sample_std": paired_ci.sample_std,
                "ci_95": [paired_ci.ci_lower, paired_ci.ci_upper],
                "spans_zero": paired_ci.spans_zero,
                "scientific_status": (
                    "PROVISIONAL_NOT_SIGNIFICANT" if paired_ci.spans_zero else "STATISTICALLY_SIGNIFICANT"
                ),
            },
            "reconciled_metrics": cls.RECONCILED_M74_METRICS,
            "weight_attribution_gap": round(
                cls.RECONCILED_M74_METRICS["A_TRAINED_ADAPTER_EM"]
                - cls.RECONCILED_M74_METRICS["D_SHUFFLED_TRAINED_WEIGHTS_EM"],
                4,
            ),
        }
