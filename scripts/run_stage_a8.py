"""REMOTE ONLY: Stage A.8 Neural Learning Mechanism Investigation & Foundation Repair.

Covers:
- Task 1: Frozen-Feature Representation Diagnosis (probe_representations_a8.py)
- Task 2: Optimization Forensics & 4-Arm Contrast (diagnose_optimization_a8.py)
- Task 3: Single Evidence-Driven Neural Correction (Branch A vs Branch B)
- Task 4: Qualification Gates & Learning-to-Learn Readiness Decision
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from annotate_frame_a2_diagnostics import signature
from diagnose_optimization_a8 import build_optimizer_and_clipper, run_optimization_diagnostics
from frame_a2_common import ROOT_HASH, ROOT_TENSOR_HASH, fork_owner
from probe_representations_a8 import run_probe_diagnosis
from run_attribute_learning import encode, evaluate, file_hash
from run_foundational_audit import digest
from run_generalization_a6_training import prepare as a6_prepare
from run_stable_attributes import compact, enrich

from adaptive_mini_gpt.data.attribute_learning import TAG_IDS
from adaptive_mini_gpt.data.exposure_a7 import repeated_plan
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103
from adaptive_mini_gpt.training.learning_state import LearningState
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda

SPEC_OPTIMIZATION = Path("data/specs/optimization_a8_v1.json")


def prepare_a8_data():
    """Builds identical 756-case coverage data and shared evaluation sets from A.7."""
    training, newcounts, blocks, shared, _ = a6_prepare()
    spec = json.loads(SPEC_OPTIMIZATION.read_text())
    plans = {"coverage": repeated_plan(blocks["coverage"], spec["steps"])}
    return training, newcounts, plans, shared


def evaluate_qualification_gates(
    metrics_6800: dict[str, Any],
    metrics_7000: dict[str, Any],
    metrics_7200: dict[str, Any],
) -> tuple[dict[str, bool], bool]:
    """Evaluates Binding Acceptance Gates 1-3 simultaneously across late checkpoints."""
    late_metrics = [metrics_6800, metrics_7000, metrics_7200]

    # Gate 1: Expanded training exact match >= 99.0% at all three checkpoints
    gate_1 = all(m["training"]["exact"] >= 0.99 for m in late_metrics)

    # Gate 2: Prior-frame retention >= 95.0% for both old_known and old_renamed
    gate_2 = all(
        m["old_known"]["exact"] >= 0.95 and m["old_renamed"]["exact"] >= 0.95 for m in late_metrics
    )

    # Gate 3: Unfamiliar grammar development exact >= 85.0% (positive, negated, absent recall each >= 80.0%)
    m_final = metrics_7200
    dev_fam_acc = m_final["development_familiar"]["exact"]
    dev_nov_acc = m_final["development_novel"]["exact"]
    gate_3_acc = dev_fam_acc >= 0.85 and dev_nov_acc >= 0.85

    recalls_fam = m_final["development_familiar"].get("state_recall", {})
    recalls_nov = m_final["development_novel"].get("state_recall", {})
    gate_3_recall = (
        (
            all(recalls_fam.get(s, 0.0) >= 0.80 for s in ("positive", "negated", "absent"))
            and all(recalls_nov.get(s, 0.0) >= 0.80 for s in ("positive", "negated", "absent"))
        )
        if recalls_fam and recalls_nov
        else False
    )

    gate_3 = bool(gate_3_acc and gate_3_recall)

    gates = {
        "gate_1_stable_expanded_acquisition": bool(gate_1),
        "gate_2_prior_frame_retention": bool(gate_2),
        "gate_3_unfamiliar_grammar_generalization": bool(gate_3),
    }
    all_pass = bool(gate_1 and gate_2 and gate_3)
    return gates, all_pass


def run_stage_a8(root_path: Path, out_dir: Path) -> dict[str, Any]:
    require_remote_cuda()
    if file_hash(root_path) != ROOT_HASH:
        raise ValueError(
            f"Parent checkpoint does not match ROOT_HASH: {file_hash(root_path)} != {ROOT_HASH}"
        )

    parent = LearningState.load(root_path)
    if digest(parent.model.state_dict()) != ROOT_TENSOR_HASH:
        raise ValueError("Parent tensor state digest does not match ROOT_TENSOR_HASH")

    training, newcounts, plans, shared = prepare_a8_data()
    opt_spec = json.loads(SPEC_OPTIMIZATION.read_text())

    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tokenizer = ByteTextTokenizerM103()
    context = parent.model.student.base_model.config.context_length

    batches = {}
    sequences = {}
    shared_batches = {}
    for k, rs in training.items():
        batches[k], sequences[k] = encode(rs, tokenizer, context)
    for k, rs in shared.items():
        shared_batches[k], sequences[k] = encode(rs, tokenizer, context)

    results: dict[str, Any] = {
        "stage": "Stage A.8",
        "git_sha": source_sha,
        "platform": require_remote_cuda(),
        "parent_sha256": ROOT_HASH,
        "parent_tensor_digest": ROOT_TENSOR_HASH,
        "tasks": {},
    }

    # ---------------------------------------------------------
    # STAGE 2 (Task 2): Optimization Instability Forensics & 4-Arm Contrast
    # ---------------------------------------------------------
    print("=== Running Stage 2 (Task 2): Optimization Diagnostics ===")
    opt_results = run_optimization_diagnostics(
        root_path, training, shared, plans, batches, shared_batches, out_dir
    )
    results["tasks"]["task_2_optimization"] = opt_results

    # ---------------------------------------------------------
    # STAGE 1 (Task 1): Frozen-Feature Representation Diagnosis
    # ---------------------------------------------------------
    print("=== Running Stage 1 (Task 1): Representation Probes ===")
    ckpt_paths: dict[str, Path] = {
        "parent": root_path,
    }
    possible_1200 = [
        *list(Path("/kaggle/input").rglob("coverage_1200.pt")),
        out_dir / "coverage_1200.pt",
        Path("runs/kaggle-stage-a7-exposure/output-v1/a7-exposure/coverage_1200.pt"),
    ]
    if any(p.exists() for p in possible_1200):
        ckpt_paths["coverage_1200"] = next(p for p in possible_1200 if p.exists())

    if (out_dir / "coverage_5800.pt").exists():
        ckpt_paths["coverage_5800"] = out_dir / "coverage_5800.pt"

    possible_7200 = [
        out_dir / "opt_0_baseline_7200.pt",
        *list(Path("/kaggle/input").rglob("coverage_7200.pt")),
        out_dir / "coverage_7200.pt",
        Path("runs/kaggle-stage-a7-exposure/output-v1/a7-exposure/coverage_7200.pt"),
    ]
    if any(p.exists() for p in possible_7200):
        ckpt_paths["coverage_7200"] = next(p for p in possible_7200 if p.exists())

    probe_results = run_probe_diagnosis(ckpt_paths, training["coverage"], shared, out_dir)
    results["tasks"]["task_1_probes"] = probe_results

    # ---------------------------------------------------------
    # STAGE 3 (Task 3): Single Evidence-Driven Neural Correction
    # ---------------------------------------------------------
    print("=== Running Stage 3 (Task 3): Branch Selection & Neural Correction ===")
    target_probe_model = "coverage_7200" if "coverage_7200" in probe_results["models"] else "parent"
    target_probes = probe_results["models"][target_probe_model]["probes"]

    h_pool_dev = target_probes["ridge_h_pool"]["metrics"]["development_novel"]["exact"]
    z_concept_dev = target_probes["ridge_z_concept"]["metrics"]["development_novel"]["exact"]
    h_stat_dev = target_probes["ridge_h_tokens_stat"]["metrics"]["development_novel"]["exact"]

    print(
        f"Probe dev exacts: h_pool={h_pool_dev:.4f}, z_concept={z_concept_dev:.4f}, h_stat={h_stat_dev:.4f}"
    )

    if h_pool_dev >= 0.85 or z_concept_dev >= 0.85:
        branch = "BRANCH_A_READOUT_OPTIMIZATION"
        repair_option = "none"
        pooling_mode = "mean"
        plasticity_mode = "selective_plasticity"
    else:
        branch = "BRANCH_B_REPRESENTATION_REPAIR"
        if h_stat_dev > h_pool_dev:
            repair_option = "B1_SPAN_RELATIVE_MAX_MEAN_POOLING"
            pooling_mode = "span_relative_max_mean"
            plasticity_mode = "selective_plasticity"
        else:
            repair_option = "B2_SELECTIVE_TOP2_BLOCKS"
            pooling_mode = "mean"
            plasticity_mode = "selective_top2_blocks"

    print(f"Selected: {branch} with repair: {repair_option}")

    # Determine winning optimizer from Stage 2
    arm_metrics = {
        arm: opt_results["arms"][arm]["final_metrics"]["training"]["exact"]
        for arm in opt_results["arms"]
    }
    winning_arm = max(arm_metrics.keys(), key=lambda k: arm_metrics[k])
    if "OPT_2_DECOUPLED_TIMESCALE" in arm_metrics and arm_metrics[
        "OPT_2_DECOUPLED_TIMESCALE"
    ] >= arm_metrics.get("OPT_0_BASELINE", 0.0):
        winning_arm = "OPT_2_DECOUPLED_TIMESCALE"
    if "OPT_3_LOGIT_BOUNDED_BCE" in arm_metrics and arm_metrics[
        "OPT_3_LOGIT_BOUNDED_BCE"
    ] >= arm_metrics.get("OPT_2_DECOUPLED_TIMESCALE", 0.0):
        winning_arm = "OPT_3_LOGIT_BOUNDED_BCE"

    print(f"Running neural correction using optimizer {winning_arm} and {repair_option}")

    student_owner = fork_owner(parent, 42)
    student_owner.model.pooling_mode = pooling_mode
    if plasticity_mode != "selective_plasticity":
        student_owner.set_mode(plasticity_mode)

    optimizer, scheduler, clipper = build_optimizer_and_clipper(
        student_owner.model, winning_arm, opt_spec
    )
    student_owner.optimizer = optimizer
    student_owner.scheduler = scheduler

    arm_cfg = opt_spec["arms"][winning_arm]
    logit_pen = arm_cfg.get("logit_penalty", 0.0)
    st_weights = arm_cfg.get("balanced_weights", None)

    eval_sets = {
        "training": training["coverage"][:756],
        "synthetic": training["coverage"][756:],
        **shared,
    }
    eval_batches = {
        "training": tuple(v[:756] for v in batches["coverage"]),
        "synthetic": tuple(v[756:] for v in batches["coverage"]),
        **shared_batches,
    }

    eval_snapshots: dict[str, Any] = {}
    plan = plans["coverage"]
    batch = batches["coverage"]

    for step in range(1, opt_spec["steps"] + 1):
        cursor = (step - 1) % len(plan)
        ix = plan[cursor]
        student_owner.step(
            *[v[ix] for v in batch],
            attribute_ids=TAG_IDS,
            clipper=clipper,
            logit_penalty=logit_pen,
            state_weights=st_weights,
        )

        if step in (6800, 7000, 7200):
            snap_metrics = {
                k: enrich(evaluate(student_owner.model, eval_batches[k], rs))
                for k, rs in eval_sets.items()
            }
            eval_snapshots[str(step)] = snap_metrics
            snap_path = out_dir / f"student_repair_{step}.pt"
            student_owner.save(snap_path)

    final_repair_path = out_dir / "student_repair_7200.pt"
    results["tasks"]["task_3_correction"] = {
        "branch": branch,
        "repair_option": repair_option,
        "optimizer": winning_arm,
        "pooling_mode": pooling_mode,
        "plasticity_mode": plasticity_mode,
        "snapshots": {
            step: {k: compact(v) for k, v in m.items()} for step, m in eval_snapshots.items()
        },
    }

    # ---------------------------------------------------------
    # STAGE 4 (Task 4): Qualification Gates & Learning-to-Learn Readiness Decision
    # ---------------------------------------------------------
    print("=== Running Stage 4 (Task 4): Qualification Gates ===")
    gates, all_pass = evaluate_qualification_gates(
        eval_snapshots["6800"],
        eval_snapshots["7000"],
        eval_snapshots["7200"],
    )

    resume_child_cmd = [
        sys.executable,
        __file__,
        "--resume-check",
        str(final_repair_path),
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(resume_child_cmd, check=True, timeout=120)
    expected_resume = json.loads((out_dir / "resume_child_a8.json").read_text())

    reloaded_owner = LearningState.load(final_repair_path)
    pred_reload = evaluate(reloaded_owner.model, batches["coverage"], training["coverage"])
    cursor = 7200 % len(plan)
    ix = plan[cursor]
    reloaded_owner.step(*[v[ix] for v in batch], attribute_ids=TAG_IDS)

    logit_diff = max(
        abs(a - b)
        for row_a, row_b in zip(
            pred_reload["logits"], expected_resume["predictions"]["logits"], strict=True
        )
        for a, b in zip(row_a, row_b, strict=True)
    )
    gate_4 = bool(logit_diff == 0.0)
    gates["gate_4_deterministic_continuation"] = gate_4

    results["tasks"]["task_4_qualification"] = {
        "gates": gates,
        "all_gates_pass": all_pass and gate_4,
        "max_logit_diff": logit_diff,
    }

    if all_pass and gate_4:
        results["verdict"] = "STAGE_A8_QUALIFIED_PASS"
        print(">>> ALL GATES PASSED! Proceeding to sealed qualification across seeds 42, 43, 44.")
    else:
        results["verdict"] = "ARCHITECTURAL_REDESIGN_REQUIRED"
        print(">>> GATES FAILED. Stopping promotion; recording Architectural Redesign Decision.")

    (out_dir / "stage_a8_final_manifest.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--resume-check", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require_remote_cuda()

    if args.resume_check:
        owner = LearningState.load(args.resume_check)
        training, _, plans, _ = prepare_a8_data()
        tokenizer = ByteTextTokenizerM103()
        context = owner.model.student.base_model.config.context_length
        batch, _ = encode(training["coverage"], tokenizer, context)
        pred = evaluate(owner.model, batch, training["coverage"])
        plan = plans["coverage"]
        cursor = 7200 % len(plan)
        ix = plan[cursor]
        loss = owner.step(*[v[ix] for v in batch], attribute_ids=TAG_IDS)
        child_info = {
            "predictions": pred,
            "loss": loss,
            "signature": signature(owner),
            "steps": owner.metadata["steps"],
        }
        (args.output_dir / "resume_child_a8.json").write_text(json.dumps(child_info))
        return

    if not args.root:
        raise ValueError("--root is required when not in resume-check mode")

    run_stage_a8(args.root, args.output_dir)


if __name__ == "__main__":
    main()
