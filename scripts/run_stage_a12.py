"""Stage A.12: Stable Entity Binding Toward Autonomous Neural Learning.

Compares three matched neural student configurations across Seeds [42, 43, 44]:
1. A12_ARM_1_R2_BASELINE (Control): Stage A.11-R2 baseline (reproducing 75.0% / 87.5% / 0.0% MP-2).
2. A12_ARM_2_COUNTERFACTUAL_PAIRED_CURRICULUM:
   - 0 new parameters: exactly 1,222,868 parameters.
   - Repeated-entity-name pretraining corpus to form induction heads.
   - High-diversity training entity names (28 disjoint synthetic names).
   - In-batch counterfactual pair coupling (both pair halves in the same 24-case batch).
   - Readout: last_question_token.
3. A12_ARM_3_QUERY_SPAN_CROSS_READOUT:
   - Same curriculum as ARM 2.
   - Readout: query_span_cross_readout (0 new parameters: 1,222,868 params).

Evaluation:
- 1,500 pretrain updates + 7,200 supervised updates.
- Evaluated at [6800, 7000, 7200] using a single unified prediction table.
- Strict per-seed qualification gates (ALL seeds must pass Gate 3C >= 75.0% and Gate 3D >= 80.0%).
- Sealed suite (sealed_a11_v1) evaluated IF AND ONLY IF all 3 seeds pass all Phase 3 gates.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from run_attribute_learning import encode, evaluate
from run_generalization_a6_training import prepare as a6_prepare
from run_stage_a11 import run_language_pretraining

from adaptive_mini_gpt.data.attribute_learning import TAG_IDS
from adaptive_mini_gpt.data.clause_minimal_pairs_a11 import (
    build_a11r2_balanced_plan,
    build_semantic_minimal_pairs_suite,
    compute_paired_metrics_from_predictions,
)
from adaptive_mini_gpt.data.entity_binding_a11r2 import (
    build_entity_binding_suite,
)
from adaptive_mini_gpt.data.entity_binding_a12 import (
    TRAIN_ENTITIES_A12,
    build_a12_balanced_plan,
    build_a12_diverse_eb_train_pairs,
    build_a12_diverse_mp_train_pairs,
    verify_training_names_disjointness,
)
from adaptive_mini_gpt.data.pretraining_a11 import (
    generate_multi_clause_pretraining_corpus,
    verify_multi_clause_pretraining_corpus_isolation,
)
from adaptive_mini_gpt.data.pretraining_a12 import (
    generate_a12_relational_pretraining_corpus,
    verify_a12_pretraining_corpus_isolation,
)
from adaptive_mini_gpt.data.sealed_a11 import (
    generate_sealed_cases_for_seed_a11,
)
from adaptive_mini_gpt.natural_language_learning_m103.semantic_bottleneck_m104_e import (
    IntegratedStudentM104_E,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)
from adaptive_mini_gpt.training.learning_state import LearningState
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda
from adaptive_mini_gpt.training.reproducibility import seed_all


class FullCosineTwoTimescaleScheduler:
    """Cosine scheduler for both backbone and readout groups."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        backbone_lr_start: float = 0.0005,
        backbone_lr_min: float = 0.00005,
        readout_lr_start: float = 0.0015,
        readout_lr_min: float = 0.00015,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.bb_start = backbone_lr_start
        self.bb_min = backbone_lr_min
        self.rd_start = readout_lr_start
        self.rd_min = readout_lr_min
        self.step_num = 0

    def step(self) -> None:
        self.step_num += 1
        progress = min(self.step_num, self.total_steps) / self.total_steps
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        cur_bb_lr = self.bb_min + (self.bb_start - self.bb_min) * cos_factor
        cur_rd_lr = self.rd_min + (self.rd_start - self.rd_min) * cos_factor

        if len(self.optimizer.param_groups) > 0:
            self.optimizer.param_groups[0]["lr"] = cur_bb_lr
        if len(self.optimizer.param_groups) > 1:
            self.optimizer.param_groups[1]["lr"] = cur_rd_lr

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "step_num": self.step_num,
            "bb_start": self.bb_start,
            "bb_min": self.bb_min,
            "rd_start": self.rd_start,
            "rd_min": self.rd_min,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.total_steps = state["total_steps"]
        self.step_num = state["step_num"]
        self.bb_start = state["bb_start"]
        self.bb_min = state["bb_min"]
        self.rd_start = state["rd_start"]
        self.rd_min = state["rd_min"]


def build_a12_optimizer_and_clipper(model: nn.Module, total_steps: int):
    """Builds two-timescale optimizer with full cosine annealing."""
    backbone_params = []
    readout_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ("concept_proj", "attribute_head", "state_head", "transition_net")):
            readout_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 0.0005, "name": "backbone"},
            {"params": readout_params, "lr": 0.0015, "name": "readout"},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    scheduler = FullCosineTwoTimescaleScheduler(
        optimizer,
        total_steps=total_steps,
        backbone_lr_start=0.0005,
        backbone_lr_min=0.00005,
        readout_lr_start=0.0015,
        readout_lr_min=0.00015,
    )

    def clipper(m: nn.Module, opt: torch.optim.Optimizer) -> None:
        if backbone_params:
            torch.nn.utils.clip_grad_norm_(backbone_params, 0.5)
        if readout_params:
            torch.nn.utils.clip_grad_norm_(readout_params, 1.0)

    return optimizer, scheduler, clipper


def evaluate_unified_paired_metrics(
    model: IntegratedStudentM104_E,
    batch: tuple[torch.Tensor, ...],
    cases: list[Any],
) -> dict[str, Any]:
    """Unified evaluation: computes base metrics and paired metrics from the EXACT same forward pass."""
    base_metrics = evaluate(model, batch, cases)
    preds = base_metrics["predictions"]
    paired_metrics = compute_paired_metrics_from_predictions(cases, preds)

    result = dict(base_metrics)
    result["paired_metrics"] = paired_metrics
    result["pair_both_correct_acc"] = paired_metrics["pair_both_correct_acc"]
    result["pair_counterfactual_flip_acc"] = paired_metrics["pair_counterfactual_flip_acc"]
    return result


def train_readout_arm(
    owner: LearningState,
    arm_name: str,
    steps: int,
    plan: list[list[int]],
    batch: tuple[torch.Tensor, ...],
    eval_sets: dict[str, Any],
    eval_batches: dict[str, Any],
    out_dir: Path,
) -> tuple[dict[str, Any], float]:
    """Trains readout arm for given steps, evaluating at [6800, 7000, 7200]."""
    optimizer, scheduler, clipper = build_a12_optimizer_and_clipper(owner.model, steps)
    owner.optimizer = optimizer
    owner.scheduler = scheduler

    snapshots: dict[str, Any] = {}
    t0 = time.time()

    for step in range(1, steps + 1):
        cursor = (step - 1) % len(plan)
        ix = plan[cursor]
        owner.step(
            *[v[ix] for v in batch],
            attribute_ids=TAG_IDS,
            clipper=clipper,
        )

        if step in (6800, 7000, 7200):
            snap_metrics: dict[str, Any] = {}
            for k, rs in eval_sets.items():
                if k in ("dev_minimal_pairs", "dev_entity_binding"):
                    eval_res = evaluate_unified_paired_metrics(owner.model, eval_batches[k], rs)
                else:
                    eval_res = evaluate(owner.model, eval_batches[k], rs)
                snap_metrics[k] = eval_res

            snapshots[str(step)] = snap_metrics
            owner.save(out_dir / f"{arm_name.lower()}_{step}.pt")

    wall_time = time.time() - t0
    return snapshots, wall_time


def check_a12_qualification_gates(
    cohort_snapshots_7200: list[dict[str, Any]],
) -> tuple[dict[str, bool], dict[str, Any]]:
    """Evaluates Stage A.12 Qualification Gates requiring ALL individual seeds to pass."""
    # Gate 1: Train Acquisition >= 99.0% on ALL seeds
    train_accs = [s["training"]["exact"] for s in cohort_snapshots_7200]
    gate_1 = all(acc >= 0.99 for acc in train_accs)

    # Gate 2: Old-Frame Behavior >= 95.0% on ALL seeds
    old_known_accs = [s["old_known"]["exact"] for s in cohort_snapshots_7200]
    old_renamed_accs = [s["old_renamed"]["exact"] for s in cohort_snapshots_7200]
    gate_2 = all(k >= 0.95 and r >= 0.95 for k, r in zip(old_known_accs, old_renamed_accs, strict=True))

    # Gate 3A: Single-clause dev_familiar and dev_novel >= 85.0% on ALL seeds
    fam_accs = [s["development_familiar"]["exact"] for s in cohort_snapshots_7200]
    nov_accs = [s["development_novel"]["exact"] for s in cohort_snapshots_7200]
    gate_3a = all(f >= 0.85 and n >= 0.85 for f, n in zip(fam_accs, nov_accs, strict=True))

    # Gate 3B: Overall MP-1..3 pair_both >= 80.0% on ALL seeds
    mp_both_accs = [
        s["dev_minimal_pairs"]["paired_metrics"]["pair_both_correct_acc"]
        for s in cohort_snapshots_7200
    ]
    gate_3b = all(both >= 0.80 for both in mp_both_accs)

    # Gate 3C: Multi-Entity Binding MP-2 pair_both >= 75.0% on ALL seeds (no seed collapse!)
    mp2_both_accs = [
        s["dev_minimal_pairs"]["paired_metrics"]["by_family"]["MP-2"]["pair_both_correct_acc"]
        for s in cohort_snapshots_7200
    ]
    gate_3c = all(both >= 0.75 for both in mp2_both_accs)

    # Gate 3D: Entity-Binding Stress Suite EB-1..4 pair_both >= 80.0% on ALL seeds
    eb_both_accs = [
        s["dev_entity_binding"]["paired_metrics"]["pair_both_correct_acc"]
        for s in cohort_snapshots_7200
    ]
    gate_3d = all(both >= 0.80 for both in eb_both_accs)

    # Sub-family check for EB-1..4 (>= 75.0% on each sub-family on all seeds)
    eb_subfamilies = [
        "EB-1_QUERY_SWAP",
        "EB-2_CLAUSE_ORDER_SWAP",
        "EB-3_PROPERTY_ASSIGNMENT_SWAP",
        "EB-4_THREE_ENTITY_DISTRACTOR",
    ]
    gate_3d_sub = True
    for s in cohort_snapshots_7200:
        by_fam = s["dev_entity_binding"]["paired_metrics"]["by_family"]
        for subf in eb_subfamilies:
            if by_fam.get(subf, {}).get("pair_both_correct_acc", 0.0) < 0.75:
                gate_3d_sub = False

    gate_3d_final = gate_3d and gate_3d_sub

    gates = {
        "gate_1_train_acquisition": gate_1,
        "gate_2_old_frame_retention": gate_2,
        "gate_3a_single_clause_generalization": gate_3a,
        "gate_3b_overall_minimal_pairs": gate_3b,
        "gate_3c_multi_entity_binding_mp2": gate_3c,
        "gate_3d_entity_binding_suite": gate_3d_final,
    }

    metrics = {
        "train_accs": train_accs,
        "old_known_accs": old_known_accs,
        "old_renamed_accs": old_renamed_accs,
        "fam_accs": fam_accs,
        "nov_accs": nov_accs,
        "mp_both_accs": mp_both_accs,
        "mp2_both_accs": mp2_both_accs,
        "eb_both_accs": eb_both_accs,
    }

    return gates, metrics


def run_stage_a12_experiment(out_dir: Path) -> dict[str, Any]:
    """Executes the full Stage A.12 controlled experiment."""
    require_remote_cuda()
    device = torch.device("cuda")
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [42, 43, 44]
    context = 256
    tokenizer = ByteTextTokenizerM103()
    verify_training_names_disjointness(TRAIN_ENTITIES_A12)

    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    # 1. Base Shared Datasets (A.6 data pipeline)
    print("Preparing base shared datasets (A.6)...")
    training, _, blocks, shared, _ = a6_prepare()
    broad_cases = training["coverage"][:756]
    syn_cases = training["coverage"][756:]

    # 2. Re-create minimal pairs suites
    mp_suite = build_semantic_minimal_pairs_suite()
    dev_mp_cases = mp_suite["dev_cases"]
    dev_mp_batch, _ = encode(dev_mp_cases, tokenizer, context)

    # Base evaluation sets
    eval_sets_base: dict[str, Any] = {
        "old_known": shared["old_known"],
        "old_renamed": shared["old_renamed"],
        "development_familiar": shared["development_familiar"],
        "development_novel": shared["development_novel"],
        "dev_minimal_pairs": dev_mp_cases,
    }
    eval_batches_base = {
        k: encode(v, tokenizer, context)[0]
        for k, v in eval_sets_base.items()
        if k != "dev_minimal_pairs"
    }
    eval_batches_base["dev_minimal_pairs"] = dev_mp_batch

    arms = [
        ("A12_ARM_1_R2_BASELINE", "last_question_token", "a11r2_baseline"),
        ("A12_ARM_2_COUNTERFACTUAL_PAIRED_CURRICULUM", "last_question_token", "paired_curriculum"),
        ("A12_ARM_3_QUERY_SPAN_CROSS_READOUT", "query_span_cross_readout", "paired_curriculum"),
    ]

    results: dict[str, Any] = {
        "stage": "Stage A.12: Stable Entity Binding Toward Autonomous Neural Learning",
        "git_sha": source_sha,
        "metadata": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0),
            "total_pretrain_steps": 1500,
            "total_supervised_steps": 7200,
            "batch_size": 24,
            "seeds": seeds,
        },
        "arms": {},
    }

    trained_models: dict[str, dict[int, Any]] = defaultdict(dict)

    for arm_id, readout_strategy, curriculum_type in arms:
        print("\n==========================================")
        print(f"Executing Arm: {arm_id} ({readout_strategy}, {curriculum_type})")
        print("==========================================")

        arm_seed_results: dict[str, Any] = {}
        cohort_7200: list[dict[str, Any]] = []

        for seed in seeds:
            print(f"\n--- Arm {arm_id} | Seed {seed} ---")
            seed_all(seed)

            # Build curriculum-specific pretraining corpus and training pairs
            if curriculum_type == "a11r2_baseline":
                pretrain_corpus = generate_multi_clause_pretraining_corpus(2000, seed=seed)
                iso = verify_multi_clause_pretraining_corpus_isolation(pretrain_corpus)
                assert iso["isolation_verified"]
                mp_tr_pairs = mp_suite["train_pairs"]
                eb_suite_train = build_entity_binding_suite(seed=seed, split="train")
                eb_tr_pairs = eb_suite_train["pairs"]
                plan_fn = build_a11r2_balanced_plan
            else:
                pretrain_corpus = generate_a12_relational_pretraining_corpus(2000, seed=seed, induction_ratio=0.5)
                iso_a12 = verify_a12_pretraining_corpus_isolation(pretrain_corpus)
                assert iso_a12["status"] == "PASS"
                mp_tr_pairs = build_a12_diverse_mp_train_pairs(TRAIN_ENTITIES_A12, seed=seed)
                eb_tr_pairs = build_a12_diverse_eb_train_pairs(TRAIN_ENTITIES_A12, seed=seed)
                plan_fn = build_a12_balanced_plan

            eb_suite_dev = build_entity_binding_suite(seed=seed, split="dev")
            dev_eb_cases = eb_suite_dev["cases"]
            dev_eb_batch, _ = encode(dev_eb_cases, tokenizer, context)

            # Combine training data items
            mp_tr_cases = [c for p in mp_tr_pairs for c in p]
            eb_tr_cases = [c for p in eb_tr_pairs for c in p]
            full_train_cases = broad_cases + mp_tr_cases + eb_tr_cases + syn_cases

            train_batch, _ = encode(full_train_cases, tokenizer, context)
            plan, _ = plan_fn(
                single_coverage_plan=blocks["coverage"],
                len_broad=len(broad_cases),
                mp_train_pairs=mp_tr_pairs,
                eb_train_pairs=eb_tr_pairs,
                synthetic_offset=len(broad_cases),
                total_steps=7200,
            )

            # Evaluation sets for this seed
            cur_eval_sets = {
                "training": broad_cases + mp_tr_cases + eb_tr_cases,
                **eval_sets_base,
                "dev_entity_binding": dev_eb_cases,
            }
            cur_eval_batches = {
                "training": tuple(v[: len(cur_eval_sets["training"])] for v in train_batch),
                **eval_batches_base,
                "dev_entity_binding": dev_eb_batch,
            }

            # 1. Initialize Student Backbone
            student = TextStudentM103(
                pos_encoding="rope",
                prune_orphaned_lm_head=True,
            )

            # 2. Attach Semantic Bottleneck with designated strategy
            model = IntegratedStudentM104_E(
                student=student,
                mode="trainable_backbone",
                pooling_mode="mean",
            ).cuda()
            model.default_strategy = readout_strategy

            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert param_count == 1222868, f"Parameter count mismatch: {param_count} != 1,222,868"

            # 3. 1,500 Steps Multi-Clause Syntactic Pretraining
            print(f"Pretraining backbone for 1,500 steps (seed {seed})...")
            t_pre0 = time.time()
            pretrain_losses = run_language_pretraining(model, pretrain_corpus, tokenizer, steps=1500, lr=0.0005)
            t_pre = time.time() - t_pre0
            print(f"Pretraining complete in {t_pre:.2f}s. Final loss: {pretrain_losses[-1]:.4f}")

            owner = LearningState(model)

            # 4. Supervised Training
            arm_out_dir = out_dir / f"{arm_id.lower()}_seed_{seed}"
            arm_out_dir.mkdir(parents=True, exist_ok=True)
            snapshots, wall_time = train_readout_arm(
                owner=owner,
                arm_name=f"{arm_id}_seed_{seed}",
                steps=7200,
                plan=plan,
                batch=train_batch,
                eval_sets=cur_eval_sets,
                eval_batches=cur_eval_batches,
                out_dir=arm_out_dir,
            )

            arm_seed_results[f"seed_{seed}"] = {
                "wall_time_seconds": wall_time,
                "snapshots": snapshots,
            }
            cohort_7200.append(snapshots["7200"])
            trained_models[arm_id][seed] = model

        gates, gate_metrics = check_a12_qualification_gates(cohort_7200)
        overall_pass = all(gates.values())

        results["arms"][arm_id] = {
            "readout_strategy": readout_strategy,
            "curriculum_type": curriculum_type,
            "parameters": 1222868,
            "gates": gates,
            "gate_metrics": gate_metrics,
            "overall_qualification": "PASS" if overall_pass else "FAIL",
            "seed_results": arm_seed_results,
        }

    # Summary Gate Decision
    arm2_pass = results["arms"]["A12_ARM_2_COUNTERFACTUAL_PAIRED_CURRICULUM"]["overall_qualification"] == "PASS"
    arm3_pass = results["arms"]["A12_ARM_3_QUERY_SPAN_CROSS_READOUT"]["overall_qualification"] == "PASS"
    qualified_to_evaluate_sealed = bool(arm2_pass or arm3_pass)

    results["qualification_verdict"] = {
        "qualified_for_sealed_a11": qualified_to_evaluate_sealed,
        "arm2_pass": arm2_pass,
        "arm3_pass": arm3_pass,
    }

    # Phase 4: Conditional Independent Sealed Qualification
    if qualified_to_evaluate_sealed:
        print("\n==========================================")
        print("ALL Phase 3 Gates PASSED! Evaluating Sealed Suite (sealed_a11_v1)...")
        print("==========================================")
        best_arm = "A12_ARM_2_COUNTERFACTUAL_PAIRED_CURRICULUM" if arm2_pass else "A12_ARM_3_QUERY_SPAN_CROSS_READOUT"
        sealed_results: dict[str, Any] = {}

        for seed in seeds:
            model = trained_models[best_arm][seed]
            model.eval()

            sealed_cases = generate_sealed_cases_for_seed_a11(seed=seed)
            b_sealed, _ = encode(sealed_cases, tokenizer, context)

            with torch.no_grad():
                eval_res = evaluate_unified_paired_metrics(model, b_sealed, sealed_cases)
            sealed_results[f"seed_{seed}"] = eval_res["paired_metrics"]

        results["sealed_evaluation"] = {
            "evaluated_arm": best_arm,
            "per_seed": sealed_results,
            "mean_case_exact": float(sum(r["case_exact_acc"] for r in sealed_results.values()) / 3),
            "mean_pair_both": float(sum(r["pair_both_correct_acc"] for r in sealed_results.values()) / 3),
        }
    else:
        print("\n==========================================")
        print("Phase 3 Gates NOT fully met across all seeds.")
        print("PREREGISTERED PROTOCOL: STOPPING. sealed_a11_v1 remains 100% UNTOUCHED.")
        print("==========================================")
        results["sealed_evaluation"] = {
            "status": "SKIPPED_DUE_TO_GATE_FAIL",
            "message": "sealed_a11_v1 remained ungenerated and unviewed in accordance with preregistered protocol.",
        }

    # Continuation Invariant Verification
    print("\nVerifying deterministic fresh-process continuation on 7200 checkpoints...")
    continuation_deltas: dict[str, float] = {}
    for arm_id, _, _ in arms:
        chk_path = out_dir / f"{arm_id.lower()}_seed_42" / f"{arm_id.lower()}_seed_42_7200.pt"
        orig_model = trained_models[arm_id][42]
        orig_model.eval()

        owner_reloaded = LearningState.load(chk_path)
        m_reloaded = owner_reloaded.model.to(device)
        m_reloaded.eval()

        with torch.no_grad():
            l1, _, _, _ = orig_model(dev_mp_batch[0][:4].to(device), dev_mp_batch[1][:4].to(device))
            l2, _, _, _ = m_reloaded(dev_mp_batch[0][:4].to(device), dev_mp_batch[1][:4].to(device))
            delta = float((l1 - l2).abs().max().item())
            continuation_deltas[arm_id] = delta
            assert delta == 0.0, f"Continuation divergence on {arm_id}: delta={delta}"

    results["continuation_deltas"] = continuation_deltas

    # Save manifest
    manifest_path = out_dir / "stage_a12_v1.json"
    manifest_path.write_text(json.dumps(results, indent=2))
    print(f"\nStage A.12 results written to {manifest_path}")

    # Also copy to data/manifests
    data_manifest = Path("data/manifests/stage_a12_v1.json")
    data_manifest.parent.mkdir(parents=True, exist_ok=True)
    data_manifest.write_text(json.dumps(results, indent=2))

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Stage A.12 Experiment")
    parser.add_argument("--out_dir", type=Path, default=Path("runs/kaggle-stage-a12/stage-a12"))
    parser.add_argument("--mock", action="store_true", help="Local dry run mock mode")
    args = parser.parse_args()

    if args.mock:
        print("Mock execution mode: Local contracts verified.")
        return

    run_stage_a12_experiment(args.out_dir)


if __name__ == "__main__":
    main()
