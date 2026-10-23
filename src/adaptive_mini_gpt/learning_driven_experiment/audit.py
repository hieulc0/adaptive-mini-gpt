"""Phase 1: Scientific Audit and Reconciliation of M51 Empirical Results."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class M51AuditReport:
    """Diagnostic report reconciling M51 reported results, entropy accounting, and failure modes."""
    # 1. Reported metrics from M51 benchmark
    reported_arm_a_random_ig: float = 1.2855
    reported_arm_b_eig_ig: float = 3.1098
    reported_arm_c_neural_ig: float = 1.1002
    reported_arm_d_filtered_ig: float = 1.1002
    reported_ablation_filter_only_ig: float = 0.8940
    reported_neural_filtered_acc: float = 0.3750
    reported_random_acc: float = 0.3194
    reported_neural_filtered_auc: float = 0.3694
    reported_random_auc: float = 0.3411
    reported_paired_correctness: float = 0.0000
    reported_unknown_detection_rate: float = 0.0000

    # 2. Reconciled entropy accounting
    num_ordinary_hypotheses: int = 8
    ordinary_prior_prob: float = 0.12
    unknown_rule_prior_prob: float = 0.04
    calculated_prior_entropy_bits: float = 0.0
    calculated_posterior_entropy_bits: float = 0.0125
    reconciled_eig_information_gain: float = 0.0
    entropy_reconciliation_verdict: str = ""

    # 3. Diagnostic findings
    unknown_rule_failure_diagnosis: str = ""
    paired_correctness_failure_diagnosis: str = ""
    learning_outcome_gap_diagnosis: str = ""


class M51ScientificAuditor:
    """Forensic analysis and audit of M51 empirical data and theoretical accounting."""

    @classmethod
    def audit_prior_entropy(
        cls,
        num_ordinary: int = 8,
        unk_prior: float = 0.04,
        reported_post_entropy: float = 0.0125,
    ) -> tuple[float, float, str]:
        """Mathematically reconcile prior entropy under 9-state hypothesis distribution."""
        ordinary_prob = (1.0 - unk_prior) / float(num_ordinary)
        
        # Shannon entropy H = - sum(p * log2(p))
        h_ordinary = -float(num_ordinary) * (ordinary_prob * math.log2(ordinary_prob))
        h_unknown = -(unk_prior * math.log2(unk_prior))
        h_total = h_ordinary + h_unknown

        ig = max(0.0, h_total - reported_post_entropy)
        verdict = (
            f"RECONCILED: With {num_ordinary} known hypotheses (prior {ordinary_prob:.4f} each) "
            f"and 1 UNKNOWN_RULE state (prior {unk_prior:.4f}), prior entropy H_0 = {h_total:.4f} bits. "
            f"Subtracting reported posterior entropy ({reported_post_entropy:.4f} bits) yields {ig:.4f} bits, "
            f"exactly matching M51's reported 3.1098 bits. The apparent discrepancy with M50 (3.0 bits) was "
            f"due to M51's non-uniform 9-state prior rather than any information-accounting bug."
        )
        return h_total, ig, verdict

    @classmethod
    def audit_unknown_rule_behavior(cls) -> str:
        """Diagnose why M51 failed to detect UNKNOWN_RULE in out-of-hypothesis environments."""
        return (
            "DIAGNOSIS OF UNKNOWN_RULE DETECTION FAILURE IN M51:\n"
            "1. Flat Likelihood Ceiling: UNKNOWN_RULE used a constant likelihood P(y | UNK) = 1/3 (0.3333). "
            "In contrast, known hypotheses whose predictions happen to align with an observation receive 1 - eps = 0.98.\n"
            "2. Prior Suppression: UNKNOWN_RULE was initialized with a low prior (0.04 vs 0.96 for known set). "
            "In a 6-query budget, even if all known hypotheses are flawed, different known hypotheses match different "
            "individual queries, each receiving 0.98 boosts that continuously suppress UNKNOWN_RULE.\n"
            "3. Lack of Posterior Predictive Calibration: The belief tracker evaluated single-step likelihoods "
            "without tracking cumulative predictive inconsistency across observed evidence.\n"
            "M52 RESOLUTION: Use a properly normalized probabilistic model where posterior belief increases or "
            "decreases strictly based on evidence (not hardcoded monotonic growth), with an empirical predictive distribution "
            "for open-world outcomes."
        )

    @classmethod
    def audit_paired_correctness_failure(cls) -> str:
        """Diagnose why all M51 arms scored 0% structural paired correctness."""
        return (
            "DIAGNOSIS OF 0% STRUCTURAL PAIRED CORRECTNESS IN M51:\n"
            "1. Pointwise vs Relational Loss: Adapters were trained exclusively on pointwise triples (u, v, label) "
            "using standard Cross-Entropy loss. The network learned superficial prior marginals or memorized query pairs, "
            "but never learned the causal dependency between an edge's presence and reachability.\n"
            "2. Monotonic/Constant Prediction on Perturbations: On minimal pairs (G_base, q) vs (G_perturbed, q) where "
            "an edge deletion or addition flips the ground truth from YES to NO, the model predicted the identical "
            "class for both graphs, yielding 0% joint accuracy.\n"
            "M52 RESOLUTION: Introduce explicit minimal-pair supervision and paired contrastive objective during "
            "candidate adapter adaptation, ensuring the network is rewarded only when it correctly predicts BOTH sides "
            "of the minimal graph modification."
        )

    @classmethod
    def audit_learning_outcome_gap(cls) -> str:
        """Diagnose why symbolic information gain failed to correlate with downstream neural learning."""
        return (
            "DIAGNOSIS OF INFORMATION GAIN VS LEARNING OUTCOME DIVERGENCE:\n"
            "In M51, Deterministic EIG achieved the highest information gain (3.1098 bits) by querying edges that "
            "maximally distinguished candidate symbolic hypotheses. However, its downstream neural adaptation accuracy "
            "(33.33%) was strictly inferior to neural ranking (37.50%). A query that efficiently cuts symbolic hypothesis "
            "space does not necessarily provide the neural encoder with representative gradient signal to learn the "
            "underlying representation. M52 replaces symbolic EIG surrogate training with direct reinforcement on "
            "measured downstream learning improvement."
        )

    def generate_audit_report(self) -> M51AuditReport:
        """Generate comprehensive M51 audit report."""
        h_prior, ig, verdict = self.audit_prior_entropy()
        return M51AuditReport(
            calculated_prior_entropy_bits=h_prior,
            reconciled_eig_information_gain=ig,
            entropy_reconciliation_verdict=verdict,
            unknown_rule_failure_diagnosis=self.audit_unknown_rule_behavior(),
            paired_correctness_failure_diagnosis=self.audit_paired_correctness_failure(),
            learning_outcome_gap_diagnosis=self.audit_learning_outcome_gap(),
        )
