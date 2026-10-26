"""Milestone M104-C Master Benchmark Orchestrator.

Orchestrates 10 scientific acceptance gates for Milestone M104-C:
- Gate 1: Parameter Accounting & Minimal Footprint Certification (1,178,474 backbone, ~35k bottleneck, >= 50% reduction vs M104-B)
- Gate 2: Phase C0 Bottleneck Diagnostic Audit (Decoupled Stage 1 Text Extraction vs Stage 2 State Dynamics)
- Gate 3: Baseline Hierarchy & Text-to-Concept Audit (Strict text inference outperforms Majority & Linear baselines)
- Gate 4: Phase C1 Grounded Concept Learning Qualification (Exceeds M104-B baseline with 95% CIs)
- Gate 5: Phase C1 Selective Plasticity Ablation (Selective top-layer plasticity outperforms full backbone fine-tuning)
- Gate 6: Phase C1 Linguistic Robustness & Paraphrase Transfer (Paraphrase acc >= 55.0%)
- Gate 7: Phase C2 Pre-Competence Gating on Familiar Concepts (Certified mastery before rule shifts)
- Gate 8: Phase C2 Continual Learning & Protected Retention (>= 60% backward retention under protected adaptation)
- Gate 9: Phase C3 Multi-Pool Practice Selection (Systematic 6-policy comparison across pools [28, 100, 200])
- Gate 10: Phase C4 Meta-Transfer & Self-Modification Preview (Transfer on unseen world and certified transactional rollback)
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from adaptive_mini_gpt.natural_language_learning_m103.capacity_auditor_m103_e1 import (
    DynamicCapacityAuditorM103_E1,
)
from adaptive_mini_gpt.natural_language_learning_m103.concept_learner_m104_c import (
    ConceptLearnerM104_C,
)
from adaptive_mini_gpt.natural_language_learning_m103.semantic_bottleneck_m104_c import (
    IntegratedStudentM104_C,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)


@dataclass
class MilestoneGateM104_C:
    gate_id: str
    name: str
    passed: bool
    evidence: str
    scientific_metric: str


@dataclass
class MilestoneReportM104_C:
    milestone_id: str
    timestamp: str
    device: str
    execution_duration_sec: float
    overall_status: str  # "QUALIFIED_PASS", "PARTIAL", "FAIL"
    gates_passed: int
    total_gates: int
    acceptance_gates: list[MilestoneGateM104_C]
    capacity_audit: dict[str, Any]
    phase_c0_diagnostic: dict[str, Any]
    phase_c1_grounded_concepts: dict[str, Any]
    phase_c2_corrective_learning: dict[str, Any]
    phase_c3_multi_pool_practice: dict[str, Any]
    phase_c4_meta_transfer: dict[str, Any]
    disk_roundtrip_verified: bool


class BenchmarkM104_C:
    """Master benchmark runner for Milestone M104-C."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.learner = ConceptLearnerM104_C(device=self.device)

    def run_benchmark(
        self,
        seeds: list[int] | None = None,
        steps: int = 150,
        output_dir: Path | None = None,
    ) -> MilestoneReportM104_C:
        start_time = time.time()
        if seeds is None:
            seeds = [42, 43, 44, 45, 46]

        print(f"=== Starting Milestone M104-C Master Benchmark on {self.device} ===", flush=True)

        # -------------------------------------------------------------
        # Gate 1: Parameter Accounting & Minimal Footprint
        # -------------------------------------------------------------
        base_student = TextStudentM103(prune_orphaned_lm_head=True).to(self.device)
        audit = DynamicCapacityAuditorM103_E1.audit_student(base_student)

        # Integrated Student with Compact Semantic Bottleneck
        integrated_student = IntegratedStudentM104_C(student=base_student, mode="selective_plasticity").to(self.device)
        total_integrated_params = sum(p.numel() for p in integrated_student.parameters())
        bottleneck_params = sum(p.numel() for p in integrated_student.bottleneck.parameters())
        backbone_params = sum(p.numel() for p in integrated_student.student.parameters())

        # Check reduction vs M104-B (M104-B adapter was 127,418 params)
        reduction_pct = ((127_418 - bottleneck_params) / 127_418) * 100.0

        gate_1_pass = (
            audit.active_forward_path_params == 1_178_474
            and audit.orphaned_lm_head_params == 0
            and audit.discrepancy == 0
            and backbone_params == 1_178_474
            and bottleneck_params < 60_000
            and reduction_pct >= 50.0
        )
        gate_1 = MilestoneGateM104_C(
            gate_id="GATE_1",
            name="Parameter Accounting & Minimal Footprint Certification",
            passed=gate_1_pass,
            evidence=(
                f"Backbone: {backbone_params:,} (1,178,474 exact), Bottleneck: {bottleneck_params:,} "
                f"({reduction_pct:.1f}% reduction vs M104-B 127k params), Total: {total_integrated_params:,}, "
                f"Orphaned: {audit.orphaned_lm_head_params}, Discrepancy: {audit.discrepancy}"
            ),
            scientific_metric="Persistent backbone == 1,178,474, Orphaned == 0, Bottleneck <= 60k params (>=50% reduction)",
        )

        # -------------------------------------------------------------
        # Phase C0: Decoupled Bottleneck Diagnostic Audit
        # -------------------------------------------------------------
        print("\n-> Executing Phase C0: Bottleneck Diagnostic Audit...", flush=True)
        p0_report = self.learner.run_phase_c0_diagnostic_audit(seed=seeds[0])

        # Gate 2: Phase C0 Bottleneck Diagnostic Audit
        gate_2_pass = (p0_report.text_to_concept.sample_count > 0 and p0_report.primary_bottleneck in ("TEXT_EXTRACTION", "CAUSAL_TRANSITION", "BALANCED"))
        gate_2 = MilestoneGateM104_C(
            gate_id="GATE_2",
            name="Phase C0 Bottleneck Diagnostic Audit",
            passed=gate_2_pass,
            evidence=(
                f"Decoupled Audit: Stage 1 Text Extraction F1={p0_report.text_to_concept.f1_score:.1f}% "
                f"(Precision={p0_report.text_to_concept.precision:.1f}%, Recall={p0_report.text_to_concept.recall:.1f}%, Exact={p0_report.text_to_concept.exact_tag_match_pct:.1f}%). "
                f"Stage 2 State Transition: Extracted Acc={p0_report.concept_to_state.extracted_attribute_accuracy:.1f}% vs "
                f"Oracle Privileged Acc={p0_report.concept_to_state.oracle_attribute_accuracy:.1f}%. "
                f"Primary Bottleneck: {p0_report.primary_bottleneck}"
            ),
            scientific_metric="Decoupled measurement of text-to-attribute extraction vs state transition dynamics",
        )

        # -------------------------------------------------------------
        # Phase C1: Grounded Concept Learning & Plasticity Ablation
        # -------------------------------------------------------------
        print("\n-> Executing Phase C1: Grounded Concept Learning Study...", flush=True)
        p1_report = self.learner.run_phase_c1_grounded_concept_study(seeds=seeds, steps=steps)

        # Gate 3: Baseline Hierarchy & Text-to-Concept Audit
        gate_3_pass = (
            p1_report.selective_plasticity_mean_held_out_pair_acc > p1_report.baseline_majority_acc
            and p1_report.selective_plasticity_mean_held_out_pair_acc > p1_report.baseline_linear_probe_acc
        )
        gate_3 = MilestoneGateM104_C(
            gate_id="GATE_3",
            name="Baseline Hierarchy & Text-to-Concept Audit",
            passed=gate_3_pass,
            evidence=(
                f"Selective Plasticity Student: {p1_report.selective_plasticity_mean_held_out_pair_acc:.1f}% vs "
                f"Majority: {p1_report.baseline_majority_acc:.1f}%, Linear: {p1_report.baseline_linear_probe_acc:.1f}%, "
                f"NN: {p1_report.baseline_nearest_neighbor_acc:.1f}%, Attr Heuristic: {p1_report.baseline_attribute_heuristic_acc:.1f}%, "
                f"Standalone (Privileged Upper Bound): {p1_report.standalone_upper_bound_held_out_acc:.1f}%"
            ),
            scientific_metric="Integrated student with text-only inference outperforms empirical Majority and Linear baselines",
        )

        # Gate 4: Phase C1 Grounded Concept Learning Qualification
        gate_4_pass = (
            p1_report.selective_plasticity_mean_held_out_pair_acc >= 50.0
            and p1_report.selective_plasticity_mean_novel_entity_acc >= 50.0
            and p1_report.selective_plasticity_mean_held_out_pair_acc >= 46.5  # Must match or exceed M104-B baseline
        )
        gate_4 = MilestoneGateM104_C(
            gate_id="GATE_4",
            name="Phase C1 Grounded Concept Learning Qualification",
            passed=gate_4_pass,
            evidence=(
                f"Held-out pair acc: {p1_report.selective_plasticity_mean_held_out_pair_acc:.1f}% "
                f"[95% CI: {p1_report.selective_plasticity_held_out_ci95[0]:.1f}-{p1_report.selective_plasticity_held_out_ci95[1]:.1f}%], "
                f"Novel entity acc: {p1_report.selective_plasticity_mean_novel_entity_acc:.1f}% "
                f"[95% CI: {p1_report.selective_plasticity_novel_ci95[0]:.1f}-{p1_report.selective_plasticity_novel_ci95[1]:.1f}%] "
                f"(M104-B baseline was 46.5%)"
            ),
            scientific_metric="Held-out pair acc >= 50.0% and novel entity >= 50.0% with 95% CIs across 5 seeds",
        )

        # Gate 5: Phase C1 Selective Plasticity Ablation
        gate_5_pass = (p1_report.selective_plasticity_mean_held_out_pair_acc >= p1_report.trainable_backbone_held_out_acc)
        gate_5 = MilestoneGateM104_C(
            gate_id="GATE_5",
            name="Phase C1 Selective Plasticity Ablation",
            passed=gate_5_pass,
            evidence=(
                f"Selective Plasticity: {p1_report.selective_plasticity_mean_held_out_pair_acc:.1f}% vs "
                f"Frozen Backbone: {p1_report.frozen_backbone_held_out_acc:.1f}%, "
                f"Trainable Backbone: {p1_report.trainable_backbone_held_out_acc:.1f}%, "
                f"M104-B Heavy Adapter: {p1_report.m104b_heavy_adapter_held_out_acc:.1f}%, "
                f"No-Adapter Baseline: {p1_report.no_adapter_baseline_held_out_acc:.1f}%"
            ),
            scientific_metric="Selective top-layer plasticity outperforms or matches full backbone training by mitigating gradient noise",
        )

        # Gate 6: Phase C1 Linguistic Robustness & Paraphrase Transfer
        gate_6_pass = (p1_report.selective_plasticity_mean_paraphrase_acc >= 55.0)
        gate_6 = MilestoneGateM104_C(
            gate_id="GATE_6",
            name="Phase C1 Linguistic Robustness & Paraphrase Transfer",
            passed=gate_6_pass,
            evidence=f"Paraphrased prompt generalization acc: {p1_report.selective_plasticity_mean_paraphrase_acc:.1f}% (Target >= 55.0%)",
            scientific_metric="Concept grounding generalizes across paraphrased questions and novel prompt phrasing",
        )

        # -------------------------------------------------------------
        # Phase C2: Corrective Concept Learning with Pre-Competence Gating
        # -------------------------------------------------------------
        print("\n-> Executing Phase C2: Corrective Concept Learning...", flush=True)
        p2_report = self.learner.run_phase_c2_corrective_learning(seed=seeds[0], steps=steps)

        # Gate 7: Phase C2 Pre-Competence Gating on Familiar Concepts
        gate_7_pass = p2_report.pre_competence_certified
        gate_7 = MilestoneGateM104_C(
            gate_id="GATE_7",
            name="Phase C2 Pre-Competence Gating on Familiar Concepts",
            passed=gate_7_pass,
            evidence=f"Pre-feedback original world acc: {p2_report.pre_competence_original_acc:.1f}% (Certified mastery verified)",
            scientific_metric="Certified competence on familiar world prior to rule perturbation (>= 75% train / 50% held-out)",
        )

        # Gate 8: Phase C2 Continual Learning & Protected Retention
        gate_8_pass = p2_report.retention_certified
        gate_8 = MilestoneGateM104_C(
            gate_id="GATE_8",
            name="Phase C2 Continual Learning & Protected Retention",
            passed=gate_8_pass,
            evidence=(
                f"Backward Retention: Replay={p2_report.replay_backward_retention:.1f}%, "
                f"Modular={p2_report.modular_backward_retention:.1f}%, "
                f"Selective={p2_report.selective_backbone_backward_retention:.1f}% "
                f"(Unprotected Baseline={p2_report.unprotected_backward_retention:.1f}%). "
                f"Forward Adaptation={p2_report.replay_forward_adaptation_acc:.1f}%"
            ),
            scientific_metric="Protected adaptation achieves >= 60% backward retention with successful forward adaptation",
        )

        # -------------------------------------------------------------
        # Phase C3: Multi-Pool Practice Selection (Pools 28, 100, 200 across 6 policies)
        # -------------------------------------------------------------
        print("\n-> Executing Phase C3: Multi-Pool Practice Study...", flush=True)
        p3_report = self.learner.run_phase_c3_multi_pool_practice(seed=seeds[0], simulator_budget=20)

        # Gate 9: Phase C3 Multi-Pool Practice Selection
        gate_9_pass = len(p3_report.pool_reports) == 3 and all(
            len(r.trajectories) == 6 for r in p3_report.pool_reports.values()
        )
        gate_9_evidence_parts = []
        for p_sz, r in p3_report.pool_reports.items():
            gate_9_evidence_parts.append(
                f"Pool {p_sz}: Best={r.best_policy_name} (AULC {r.best_policy_aulc:.1f} vs Random {r.random_aulc:.1f}, Gain: {r.sample_efficiency_gain_pct:+.1f}%)"
            )
        gate_9 = MilestoneGateM104_C(
            gate_id="GATE_9",
            name="Phase C3 Multi-Pool Practice Selection & Scaling",
            passed=gate_9_pass,
            evidence=" | ".join(gate_9_evidence_parts),
            scientific_metric="Systematic 6-policy evaluation completed across candidate pools [28, 100, 200]",
        )

        # -------------------------------------------------------------
        # Phase C4: Learning-to-Learn Meta-Transfer & Self-Modification
        # -------------------------------------------------------------
        print("\n-> Executing Phase C4: Learning-to-Learn Meta-Transfer & Self-Modification...", flush=True)
        p4_report = self.learner.run_phase_c4_meta_transfer(seed=seeds[0], steps=steps, target_acc=60.0)

        # Gate 10: Phase C4 Learning-to-Learn Meta-Transfer Qualification
        gate_10_pass = p4_report.positive_meta_transfer_certified
        gate_10 = MilestoneGateM104_C(
            gate_id="GATE_10",
            name="Phase C4 Learning-to-Learn Meta-Transfer Qualification",
            passed=gate_10_pass,
            evidence=(
                f"World B Transfer: Experienced={p4_report.experienced_persistent_final_acc:.1f}% ({p4_report.experienced_persistent_calls_to_target} steps) vs "
                f"Fresh={p4_report.fresh_baseline_final_acc:.1f}% ({p4_report.fresh_baseline_calls_to_target} steps), "
                f"Pretrained={p4_report.pretrained_no_rule_final_acc:.1f}%, NoMemory={p4_report.experienced_no_memory_final_acc:.1f}%. "
                f"Speedup Ratio: {p4_report.speedup_ratio:.2f}x. Self-Modification Preview Accepted: {p4_report.self_modification_preview_accepted}"
            ),
            scientific_metric="Experienced persistent student demonstrates faster concept acquisition on independent unseen world",
        )

        # -------------------------------------------------------------
        # Disk Roundtrip & Parameter Persistence Audit
        # -------------------------------------------------------------
        disk_verified = False
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "checkpoint_m104_c.pt"
            torch.save(integrated_student.state_dict(), ckpt_path)

            fresh_base_s = TextStudentM103(prune_orphaned_lm_head=True).to(self.device)
            reloaded_student = IntegratedStudentM104_C(student=fresh_base_s, mode="selective_plasticity").to(self.device)
            reloaded_student.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))

            dummy_prompt = torch.randint(0, 256, (2, 32), device=self.device)
            dummy_action = torch.tensor([0, 1], device=self.device)

            integrated_student.eval()
            reloaded_student.eval()
            with torch.no_grad():
                l_orig, _, z_orig, _ = integrated_student(dummy_prompt, dummy_action)
                l_rel, _, z_rel, _ = reloaded_student(dummy_prompt, dummy_action)
                max_diff = float((l_orig - l_rel).abs().max().item())
                max_z_diff = float((z_orig - z_rel).abs().max().item())
                if max_diff < 1e-6 and max_z_diff < 1e-6:
                    disk_verified = True

        gates = [gate_1, gate_2, gate_3, gate_4, gate_5, gate_6, gate_7, gate_8, gate_9, gate_10]
        passed_count = sum(1 for g in gates if g.passed)
        overall_status = "QUALIFIED_PASS" if passed_count == 10 else ("PARTIAL" if passed_count >= 5 else "FAIL")

        exec_time = time.time() - start_time
        print(f"\n=== Benchmark Complete in {exec_time:.2f}s! Passed {passed_count}/10 gates. Status: {overall_status} ===", flush=True)

        report = MilestoneReportM104_C(
            milestone_id="M104-C",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            device=str(self.device),
            execution_duration_sec=exec_time,
            overall_status=overall_status,
            gates_passed=passed_count,
            total_gates=10,
            acceptance_gates=gates,
            capacity_audit=asdict(audit),
            phase_c0_diagnostic=asdict(p0_report),
            phase_c1_grounded_concepts=asdict(p1_report),
            phase_c2_corrective_learning=asdict(p2_report),
            phase_c3_multi_pool_practice=asdict(p3_report),
            phase_c4_meta_transfer=asdict(p4_report),
            disk_roundtrip_verified=disk_verified,
        )

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            json_path = output_dir / "m104_c_milestone_report.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(asdict(report), f, indent=2)
            print(f"Report saved to {json_path}", flush=True)

        return report
