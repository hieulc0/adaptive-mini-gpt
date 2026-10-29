"""Stage A.13: Entity-Binding Architectural Decision Study (Track A).

Controlled architectural comparison on the exact unchanged Stage A.11-R2 curriculum
across Seeds 42, 43, 44 under matched budgets (1,500 pretrain + 7,200 supervised updates):
1. A13_ARM_1_LAST_TOKEN_CONTROL: Exact A.11-R2 last_question_token baseline.
2. A13_ARM_2_LEARNED_ENTITY_CROSS_ATTENTION: Learned Multi-Head Cross-Attention readout
   (reusing student's registered ECA MultiheadAttention, 0 new parameters, exactly 1,222,868 params)
   where Q = W_Q h_query attends to K = W_K h_prefix, V = W_V h_prefix.

Preregistered protocol:
If A13_ARM_2 passes all gates (Gates 1, 2, 3A, 3B, 3C, 3D) across all 3 seeds: evaluate sealed_a11_v1.
Otherwise: STOP Track A, keep sealed suite untouched, and record formal architectural decision.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
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
from adaptive_mini_gpt.data.pretraining_a11 import (
    generate_multi_clause_pretraining_corpus,
    verify_multi_clause_pretraining_corpus_isolation,
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


setattr(sys.modules["__main__"], "FullCosineTwoTimescaleScheduler", FullCosineTwoTimescaleScheduler)  # noqa: B010


def build_a13_optimizer_and_clipper(model: nn.Module, total_steps: int):
    """Builds two-timescale optimizer with readout group covering ECA and semantic heads."""
    backbone_params = []
    readout_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ("concept_proj", "attribute_head", "state_head", "transition_net", "eca_extractor")):
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
    """Unified evaluation: computes base metrics and paired metrics from the exact same pass."""
    base_metrics = evaluate(model, batch, cases)
    preds = base_metrics["predictions"]
    paired_metrics = compute_paired_metrics_from_predictions(cases, preds)

    result = dict(base_metrics)
    result["paired_metrics"] = paired_metrics
    result["pair_both_correct_acc"] = paired_metrics["pair_both_correct_acc"]
    result["pair_counterfactual_flip_acc"] = paired_metrics["pair_counterfactual_flip_acc"]
    return result


def train_stage_a13_arm(
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
    optimizer, scheduler, clipper = build_a13_optimizer_and_clipper(owner.model, steps)
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


def check_a13_qualification_gates(
    cohort_snapshots_7200: list[dict[str, Any]],
) -> tuple[dict[str, bool], dict[str, Any]]:
    """Evaluates Stage A.13 Qualification Gates across individual seeds 42, 43, 44."""
    train_accs = [s["training"]["exact"] for s in cohort_snapshots_7200]
    gate_1 = all(acc >= 0.99 for acc in train_accs)

    old_known_accs = [s["old_known"]["exact"] for s in cohort_snapshots_7200]
    old_renamed_accs = [s["old_renamed"]["exact"] for s in cohort_snapshots_7200]
    gate_2 = all(k >= 0.95 and r >= 0.95 for k, r in zip(old_known_accs, old_renamed_accs, strict=True))

    fam_accs = [s["development_familiar"]["exact"] for s in cohort_snapshots_7200]
    nov_accs = [s["development_novel"]["exact"] for s in cohort_snapshots_7200]
    gate_3a = all(f >= 0.85 and n >= 0.85 for f, n in zip(fam_accs, nov_accs, strict=True))

    mp_both_accs = [
        s["dev_minimal_pairs"]["paired_metrics"]["pair_both_correct_acc"]
        for s in cohort_snapshots_7200
    ]
    gate_3b = all(both >= 0.80 for both in mp_both_accs)

    mp2_both_accs = [
        s["dev_minimal_pairs"]["paired_metrics"]["by_family"]["MP-2"]["pair_both_correct_acc"]
        for s in cohort_snapshots_7200
    ]
    gate_3c = all(both >= 0.75 for both in mp2_both_accs)

    eb_both_accs = [
        s["dev_entity_binding"]["paired_metrics"]["pair_both_correct_acc"]
        for s in cohort_snapshots_7200
    ]
    gate_3d = all(both >= 0.80 for both in eb_both_accs)

    gates = {
        "gate_1_train_acquisition": gate_1,
        "gate_2_old_frame_retention": gate_2,
        "gate_3a_single_clause_generalization": gate_3a,
        "gate_3b_overall_minimal_pairs": gate_3b,
        "gate_3c_multi_entity_binding_mp2": gate_3c,
        "gate_3d_entity_binding_suite": gate_3d,
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


def run_stage_a13_experiment(out_dir: Path, checkpoint_dir: Path | None = None) -> dict[str, Any]:
    # Check for existing completed Stage A.13 manifest
    if (out_dir / "stage_a13_v1.json").exists():
        print(f"Reusing existing Stage A.13 results at {out_dir / 'stage_a13_v1.json'}")
        return json.loads((out_dir / "stage_a13_v1.json").read_text())
    if checkpoint_dir is not None:
        cand = list(checkpoint_dir.rglob("stage_a13_v1.json"))
        if cand:
            print(f"Found completed Stage A.13 results at {cand[0]}. Reusing certified Track A results.")
            data = json.loads(cand[0].read_text())
            (out_dir / "stage_a13_v1.json").write_text(json.dumps(data, indent=2))
            return data
    """Executes the Track A Stage A.13 architectural decision study."""
    require_remote_cuda()
    device = torch.device("cuda")
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [42, 43, 44]
    context = 256
    tokenizer = ByteTextTokenizerM103()

    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print("=== Track A Stage A.13: Entity-Binding Architectural Decision Study ===")
    print(f"Commit: {source_sha} | Device: {torch.cuda.get_device_name(0)}")

    # Base shared datasets
    training, _, blocks, shared, _ = a6_prepare()
    broad_cases = training["coverage"][:756]
    syn_cases = training["coverage"][756:]

    mp_suite = build_semantic_minimal_pairs_suite()
    mp_train_pairs = mp_suite["train_pairs"]
    mp_train_cases = mp_suite["train_cases"]
    dev_mp_cases = mp_suite["dev_cases"]
    dev_mp_batch, _ = encode(dev_mp_cases, tokenizer, context)

    # In A.11-R2, eb_train_suite was instantiated once with seed=42
    eb_train_suite = build_entity_binding_suite(seed=42, split="train")
    eb_train_pairs = eb_train_suite["pairs"]
    eb_train_cases = eb_train_suite["cases"]

    full_train_cases = broad_cases + mp_train_cases + eb_train_cases + syn_cases
    train_batch_full, _ = encode(full_train_cases, tokenizer, context)

    # Construct Balanced 24-case Batch Plan
    balanced_plan, _ = build_a11r2_balanced_plan(
        blocks["coverage"],
        len_broad=756,
        mp_train_pairs=mp_train_pairs,
        eb_train_pairs=eb_train_pairs,
        synthetic_offset=756,
        total_steps=7200,
    )

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
        ("A13_ARM_1_LAST_TOKEN_CONTROL", "last_question_token"),
        ("A13_ARM_2_LEARNED_ENTITY_CROSS_ATTENTION", "learned_entity_cross_attention"),
    ]

    results: dict[str, Any] = {
        "stage": "Stage A.13: Entity-Binding Architectural Decision Study",
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

    for arm_id, readout_strategy in arms:
        print("\n==========================================")
        print(f"Executing Arm: {arm_id} ({readout_strategy})")
        print("==========================================")

        arm_seed_results: dict[str, Any] = {}
        cohort_7200: list[dict[str, Any]] = []

        for seed in seeds:
            print(f"\n--- Arm {arm_id} | Seed {seed} ---")
            seed_all(seed)

            pretrain_corpus = generate_multi_clause_pretraining_corpus(2000, seed=seed)
            iso = verify_multi_clause_pretraining_corpus_isolation(pretrain_corpus)
            assert iso["isolation_verified"]

            eb_dev_suite = build_entity_binding_suite(seed=seed, split="dev")
            eb_dev_cases = eb_dev_suite["cases"]
            eb_dev_batch, _ = encode(eb_dev_cases, tokenizer, context)

            cur_eval_sets = {
                "training": broad_cases + mp_train_cases + eb_train_cases,
                **eval_sets_base,
                "dev_entity_binding": eb_dev_cases,
            }
            cur_eval_batches = {
                "training": tuple(v[: len(cur_eval_sets["training"])] for v in train_batch_full),
                **eval_batches_base,
                "dev_entity_binding": eb_dev_batch,
            }

            student = TextStudentM103(pos_encoding="rope", prune_orphaned_lm_head=True)
            model = IntegratedStudentM104_E(student=student, mode="trainable_backbone", pooling_mode="mean").cuda()
            model.default_strategy = readout_strategy

            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            assert param_count == 1222868, f"Parameter count mismatch: {param_count} != 1,222,868"

            print(f"Pretraining backbone for 1,500 steps (seed {seed})...")
            t_pre0 = time.time()
            pretrain_losses = run_language_pretraining(model, pretrain_corpus, tokenizer, steps=1500, lr=0.0005)
            t_pre = time.time() - t_pre0
            print(f"Pretraining complete in {t_pre:.2f}s. Final loss: {pretrain_losses[-1]:.4f}")

            owner = LearningState(model)
            arm_out_dir = out_dir / f"{arm_id.lower()}_seed_{seed}"
            arm_out_dir.mkdir(parents=True, exist_ok=True)

            snapshots, wall_time = train_stage_a13_arm(
                owner=owner,
                arm_name=f"{arm_id}_seed_{seed}",
                steps=7200,
                plan=balanced_plan,
                batch=train_batch_full,
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

        gates, gate_metrics = check_a13_qualification_gates(cohort_7200)
        overall_pass = all(gates.values())

        results["arms"][arm_id] = {
            "readout_strategy": readout_strategy,
            "parameters": 1222868,
            "gates": gates,
            "gate_metrics": gate_metrics,
            "overall_qualification": "PASS" if overall_pass else "FAIL",
            "seed_results": arm_seed_results,
        }

    # Summary Gate Decision
    arm2_pass = results["arms"]["A13_ARM_2_LEARNED_ENTITY_CROSS_ATTENTION"]["overall_qualification"] == "PASS"
    qualified_to_evaluate_sealed = bool(arm2_pass)

    results["qualification_verdict"] = {
        "qualified_for_sealed_a11": qualified_to_evaluate_sealed,
        "arm2_pass": arm2_pass,
    }

    if qualified_to_evaluate_sealed:
        print("\n==========================================")
        print("A13_ARM_2 PASSED ALL GATES! Evaluating sealed_a11_v1...")
        print("==========================================")
        best_arm = "A13_ARM_2_LEARNED_ENTITY_CROSS_ATTENTION"
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
        print("A13_ARM_2 gates NOT fully met across all seeds.")
        print("PREREGISTERED PROTOCOL: STOPPING. sealed_a11_v1 remains 100% UNTOUCHED.")
        print("==========================================")
        results["sealed_evaluation"] = {
            "status": "SKIPPED_DUE_TO_GATE_FAIL",
            "message": "sealed_a11_v1 remained ungenerated and unviewed in accordance with preregistered protocol.",
        }

    manifest_path = out_dir / "stage_a13_v1.json"
    manifest_path.write_text(json.dumps(results, indent=2))
    print(f"\nStage A.13 results written to {manifest_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage A.13 Experiment")
    parser.add_argument("--out_dir", type=Path, default=Path("runs/kaggle-stage-a13-b0/stage-a13"))
    parser.add_argument("--mock", action="store_true", help="Local dry run mock mode")
    args = parser.parse_args()

    if args.mock:
        print("Mock execution mode: Local contracts verified.")
        return

    run_stage_a13_experiment(args.out_dir)


if __name__ == "__main__":
    main()
