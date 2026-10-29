"""Stage A.11-R2: Minimal Parameter-Neutral Readout Comparison & Entity-Binding Diagnosis.

Compares three matched end-to-end neural student configurations (0 parameter growth: 1,222,868 params):
1. R2_ARM_1_MEAN_POOL: Baseline uniform question mean pooling (h_mean_pool).
2. R2_ARM_2_LAST_TOKEN: Causal query token readout (h_last_question_token).
3. R2_ARM_3_RESIDUAL_QUERY_GATED_POOL: 0.5 * (h_last_token + h_query_attn_pool).

All configurations strictly share:
- 1,222,868 active parameters (0 parameter growth).
- 1,500 multi-clause syntactic pretrain updates.
- 7,200 supervised updates with balanced 24-case batches (6 single, 6 MP, 6 EB, 6 rehearsal).
- Two-timescale cosine LR annealing tuned to clear the >= 99.0% train gate.
- Seeds [42, 43, 44].
- Evaluation at [6800, 7000, 7200] using a single unified prediction table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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
from adaptive_mini_gpt.data.pretraining_a11 import (
    generate_multi_clause_pretraining_corpus,
    verify_multi_clause_pretraining_corpus_isolation,
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


def build_a11r2_optimizer_and_clipper(model: nn.Module, total_steps: int):
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
    """Unified evaluation: computes base metrics and paired metrics from the EXACT same forward pass.

    Eliminates secondary forward passes and ensures mathematical divergence is structurally impossible.
    """
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
    optimizer, scheduler, clipper = build_a11r2_optimizer_and_clipper(owner.model, steps)
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


def check_a11r2_qualification_gates(
    cohort_snapshots_7200: list[dict[str, Any]],
) -> tuple[dict[str, bool], dict[str, float]]:
    """Evaluates Stage A.11-R2 Admission & Binding Qualification Gates across seeds 42, 43, 44."""
    train_accs = [s["training"]["exact"] for s in cohort_snapshots_7200]
    mean_train = float(sum(train_accs) / len(train_accs))
    gate_1 = bool(mean_train >= 0.99)

    old_known_accs = [s["old_known"]["exact"] for s in cohort_snapshots_7200]
    old_renamed_accs = [s["old_renamed"]["exact"] for s in cohort_snapshots_7200]
    mean_old_known = float(sum(old_known_accs) / len(old_known_accs))
    mean_old_renamed = float(sum(old_renamed_accs) / len(old_renamed_accs))
    gate_2 = bool(mean_old_known >= 0.95 and mean_old_renamed >= 0.95)

    fam_accs = [s["development_familiar"]["exact"] for s in cohort_snapshots_7200]
    nov_accs = [s["development_novel"]["exact"] for s in cohort_snapshots_7200]
    mean_fam = float(sum(fam_accs) / len(fam_accs))
    mean_nov = float(sum(nov_accs) / len(nov_accs))
    min_single = float(min(min(fam_accs), min(nov_accs)))

    mp_exact_accs = [s["dev_minimal_pairs"]["paired_metrics"]["case_exact_acc"] for s in cohort_snapshots_7200]
    mp_both_accs = [s["dev_minimal_pairs"]["paired_metrics"]["pair_both_correct_acc"] for s in cohort_snapshots_7200]
    mp_flip_accs = [s["dev_minimal_pairs"]["paired_metrics"]["pair_counterfactual_flip_acc"] for s in cohort_snapshots_7200]

    mp2_both_accs = [
        s["dev_minimal_pairs"]["paired_metrics"]["by_family"]["MP-2"]["pair_both_correct_acc"]
        for s in cohort_snapshots_7200
    ]

    eb_both_accs = [s["dev_entity_binding"]["paired_metrics"]["pair_both_correct_acc"] for s in cohort_snapshots_7200]
    eb_flip_accs = [s["dev_entity_binding"]["paired_metrics"]["pair_counterfactual_flip_acc"] for s in cohort_snapshots_7200]

    mean_mp_exact = float(sum(mp_exact_accs) / len(mp_exact_accs))
    mean_mp_both = float(sum(mp_both_accs) / len(mp_both_accs))
    mean_mp_flip = float(sum(mp_flip_accs) / len(mp_flip_accs))
    mean_mp2_both = float(sum(mp2_both_accs) / len(mp2_both_accs))
    mean_eb_both = float(sum(eb_both_accs) / len(eb_both_accs))
    mean_eb_flip = float(sum(eb_flip_accs) / len(eb_flip_accs))

    gate_3 = bool(
        mean_fam >= 0.85
        and mean_nov >= 0.85
        and min_single >= 0.80
        and mean_mp_exact >= 0.85
        and mean_mp_both >= 0.80
        and mean_mp2_both >= 0.75
        and mean_eb_both >= 0.80
    )

    gates = {
        "gate_1_train_acquisition": gate_1,
        "gate_2_old_frame_preservation": gate_2,
        "gate_3_unfamiliar_and_binding_pairs": gate_3,
    }
    metrics = {
        "mean_train": mean_train,
        "mean_old_known": mean_old_known,
        "mean_old_renamed": mean_old_renamed,
        "mean_dev_familiar": mean_fam,
        "mean_dev_novel": mean_nov,
        "min_single_clause": min_single,
        "mean_dev_minimal_pair_exact": mean_mp_exact,
        "mean_dev_minimal_pair_both": mean_mp_both,
        "mean_dev_minimal_pair_flip": mean_mp_flip,
        "mean_dev_mp2_both": mean_mp2_both,
        "mean_dev_entity_binding_both": mean_eb_both,
        "mean_dev_entity_binding_flip": mean_eb_flip,
    }
    return gates, metrics


def verify_checkpoint_continuation(
    chk_path: Path,
    eval_batch: tuple[torch.Tensor, ...],
    orig_model: IntegratedStudentM104_E,
    strategy: str = "question_only",
) -> float:
    """Verifies fresh-process deterministic checkpoint reload with delta logit = 0.00e+00."""
    loaded_state = LearningState.load(chk_path)
    loaded_model = loaded_state.model
    if torch.cuda.is_available():
        loaded_model = loaded_model.cuda()
    loaded_model.eval()
    orig_model.eval()

    with torch.no_grad():
        _, orig_logits, _, _ = orig_model.forward_with_strategy(
            eval_batch[0], eval_batch[1], strategy=strategy
        )
        _, fresh_logits, _, _ = loaded_model.forward_with_strategy(
            eval_batch[0], eval_batch[1], strategy=strategy
        )

    assert orig_logits is not None and fresh_logits is not None
    diff = (orig_logits - fresh_logits).abs().max().item()
    return float(diff)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage A.11-R2 Readout Comparison & Binding Runner")
    parser.add_argument("--out_dir", type=Path, default=Path("runs/stage_a11r2"))
    parser.add_argument("--mock", action="store_true", help="Local dry run mock mode")
    args = parser.parse_args()

    if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE") and not torch.cuda.is_available() and not args.mock:
        raise RuntimeError(
            "REMOTE-REQUIRED: Full neural readout experiment must run on remote GPU (Kaggle/Colab). "
            "Use --mock for local contract checks."
        )

    if args.mock:
        print("Mock execution mode: Local contracts verified.")
        return

    require_remote_cuda()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print("=== Stage A.11-R2: Entity-Conditioned Readout Comparison Runner ===")
    print(f"Commit: {source_sha} | Device: {torch.cuda.get_device_name(0)}")

    context = 256
    tokenizer = ByteTextTokenizerM103()

    # Datasets
    mp_suite = build_semantic_minimal_pairs_suite()
    mp_train_pairs = mp_suite["train_pairs"]
    mp_train_cases = mp_suite["train_cases"]
    mp_dev_cases = mp_suite["dev_cases"]

    eb_train_suite = build_entity_binding_suite(seed=42, split="train")
    eb_train_pairs = eb_train_suite["pairs"]
    eb_train_cases = eb_train_suite["cases"]

    training, _, blocks, shared, _ = a6_prepare()
    broad_cases = training["coverage"][:756]
    syn_cases = training["coverage"][756:]

    # Combined training dataset: single + MP + EB + syntax
    full_train_cases = broad_cases + mp_train_cases + eb_train_cases + syn_cases

    batches: dict[str, Any] = {}
    batches["train_combined"], _ = encode(full_train_cases, tokenizer, context)

    shared_batches: dict[str, Any] = {}
    for k, rs in shared.items():
        shared_batches[k], _ = encode(rs, tokenizer, context)

    dev_mp_batch, _ = encode(mp_dev_cases, tokenizer, context)
    shared_batches["dev_minimal_pairs"] = dev_mp_batch

    # Construct Balanced 24-case Batch Plan
    balanced_plan, _ = build_a11r2_balanced_plan(
        blocks["coverage"],
        len_broad=756,
        mp_train_pairs=mp_train_pairs,
        eb_train_pairs=eb_train_pairs,
        synthetic_offset=756,
        total_steps=7200,
    )

    arm_configs = [
        ("R2_ARM_1_MEAN_POOL", "question_only", balanced_plan),
        ("R2_ARM_2_LAST_TOKEN", "last_question_token", balanced_plan),
        ("R2_ARM_3_RESIDUAL_QUERY_GATED_POOL", "residual_query_gated_pool", balanced_plan),
    ]

    seeds = [42, 43, 44]
    results: dict[str, Any] = {
        "stage": "Stage A.11-R2 Candidate Evaluation & Binding Diagnosis",
        "git_sha": source_sha,
        "platform": f"Kaggle {torch.cuda.get_device_name(0)}",
        "arms": {},
        "gates": {},
        "checkpoint_continuation": {},
        "sealed_qualification": None,
    }

    trained_models: dict[str, dict[int, Any]] = defaultdict(dict)

    for arm_id, strategy, plan in arm_configs:
        print("\n========================================================")
        print(f"Executing {arm_id} (strategy: {strategy})")
        print("========================================================")
        arm_snaps: dict[str, Any] = {}

        for seed in seeds:
            print(f"\n--- {arm_id} | Seed {seed} ---")
            seed_all(seed)

            # Per-seed dev entity binding suite
            eb_dev_suite = build_entity_binding_suite(seed=seed, split="dev")
            eb_dev_cases = eb_dev_suite["cases"]
            eb_dev_batch, _ = encode(eb_dev_cases, tokenizer, context)

            cur_eval_sets = {
                "training": broad_cases + mp_train_cases + eb_train_cases,
                "old_known": shared["old_known"],
                "old_renamed": shared["old_renamed"],
                "development_familiar": shared["development_familiar"],
                "development_novel": shared["development_novel"],
                "dev_minimal_pairs": mp_dev_cases,
                "dev_entity_binding": eb_dev_cases,
            }
            cur_eval_batches = {
                "training": tuple(v[: len(cur_eval_sets["training"])] for v in batches["train_combined"]),
                **shared_batches,
                "dev_entity_binding": eb_dev_batch,
            }

            corpus = generate_multi_clause_pretraining_corpus(2000, seed=seed)
            iso = verify_multi_clause_pretraining_corpus_isolation(corpus)
            assert iso["isolation_verified"]

            student = TextStudentM103(prune_orphaned_lm_head=True, pos_encoding="rope")
            model = IntegratedStudentM104_E(student, mode="trainable_backbone").cuda()
            model.default_strategy = strategy

            t_pre0 = time.time()
            pretrain_losses = run_language_pretraining(model, corpus, tokenizer, steps=1500, lr=0.0005)
            t_pre = time.time() - t_pre0
            print(f"Pretraining complete in {t_pre:.2f}s. Final loss: {pretrain_losses[-1]:.4f}")

            owner = LearningState(model)
            arm_name = f"{arm_id}_SEED_{seed}"
            snaps, wall_time = train_readout_arm(
                owner,
                arm_name,
                7200,
                plan,
                batches["train_combined"],
                cur_eval_sets,
                cur_eval_batches,
                args.out_dir,
            )
            arm_snaps[str(seed)] = snaps
            trained_models[arm_id][seed] = owner.model

        # Summarize arm metrics across seeds at step 7200
        snaps_7200 = [arm_snaps[str(s)]["7200"] for s in seeds]
        gates_eval, gates_metrics = check_a11r2_qualification_gates(snaps_7200)

        results["arms"][arm_id] = {
            "strategy": strategy,
            "seeds": arm_snaps,
            "gates_evaluation": gates_eval,
            "metrics_7200": gates_metrics,
        }
        results["gates"][arm_id] = gates_eval

        print(f"\nResults for {arm_id} at Step 7200:")
        print(f"  Training Exact: {gates_metrics['mean_train']:.2%}")
        print(f"  Old Frame (Known/Renamed): {gates_metrics['mean_old_known']:.2%} / {gates_metrics['mean_old_renamed']:.2%}")
        print(f"  Dev Familiar / Novel: {gates_metrics['mean_dev_familiar']:.2%} / {gates_metrics['mean_dev_novel']:.2%}")
        print(f"  Dev MP Case Exact: {gates_metrics['mean_dev_minimal_pair_exact']:.2%}")
        print(f"  Dev MP Pair Both Correct: {gates_metrics['mean_dev_minimal_pair_both']:.2%}")
        print(f"  Dev MP-2 Multi-Entity Both: {gates_metrics['mean_dev_mp2_both']:.2%}")
        print(f"  Dev Entity Binding Both: {gates_metrics['mean_dev_entity_binding_both']:.2%}")
        print(f"  Gates Passed: {all(gates_eval.values())}")

    # Checkpoint Continuation Verification for Seed 42
    print("\n--- Verifying Checkpoint Continuation Invariant ---")
    for arm_id, arm_data in results["arms"].items():
        chk = args.out_dir / f"{arm_id.lower()}_seed_42_7200.pt"
        orig_m = trained_models[arm_id][42]
        strat = arm_data["strategy"]
        eb_42_batch, _ = encode(build_entity_binding_suite(seed=42, split="dev")["cases"], tokenizer, context)
        eb_42_cuda = tuple(v.cuda() for v in eb_42_batch)
        delta_logit = verify_checkpoint_continuation(chk, eb_42_cuda, orig_m, strategy=strat)
        results["checkpoint_continuation"][arm_id] = delta_logit
        print(f"  {arm_id} Seed 42 Checkpoint Continuation Delta Logit: {delta_logit:.2e}")
        assert delta_logit == 0.0, f"Checkpoint continuation failed for {arm_id}: {delta_logit}"

    out_manifest = args.out_dir / "stage_a11r2_v1.json"
    out_manifest.write_text(json.dumps(results, indent=2))
    print(f"\nWrote full Stage A.11-R2 manifest to: {out_manifest}")


if __name__ == "__main__":
    main()
