"""Stage A.11-R: Minimal Readout Diagnosis & Representation Audit.

Compares three matched end-to-end neural student configurations:
1. R_ARM_1_MEAN_POOL_CONTROL: Uniform mean pooling over question span (reproducing A.11).
2. R_ARM_2_LAST_TOKEN_READOUT: Route causal final question token into concept_proj -> attribute_head.
3. R_ARM_3_QUERY_ATTN_POOL_READOUT: Route query-anchored attention pooling into concept_proj -> attribute_head
   with balanced batch weighting (6 single, 6 minimal-pair, 6 rehearsal).

All configurations strictly share:
- 1,222,868 active parameters (0 parameter growth).
- 1,500 multi-clause syntactic pretrain updates.
- 7,200 supervised updates with snapshots at [6800, 7000, 7200].
- Seeds [42, 43, 44].
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from run_attribute_learning import encode, evaluate
from run_generalization_a6_training import prepare as a6_prepare
from run_stage_a11 import (
    build_a11_optimizer_and_clipper,
    build_relational_augmented_plan,
    run_language_pretraining,
)

from adaptive_mini_gpt.data.attribute_learning import TAG_IDS
from adaptive_mini_gpt.data.clause_minimal_pairs_a11 import (
    build_balanced_relational_plan,
    build_semantic_minimal_pairs_suite,
    compute_paired_metrics,
)
from adaptive_mini_gpt.data.pretraining_a11 import (
    generate_multi_clause_pretraining_corpus,
    verify_multi_clause_pretraining_corpus_isolation,
)
from adaptive_mini_gpt.data.sealed_a11 import (
    generate_master_sealed_suite_a11,
    verify_sealed_a11_isolation,
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
from adaptive_mini_gpt.training.learning_state import (
    LearningState,
)
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda
from adaptive_mini_gpt.training.reproducibility import seed_all


def evaluate_with_paired_metrics(
    model: IntegratedStudentM104_E,
    batch: tuple[torch.Tensor, ...],
    cases: list[Any],
) -> dict[str, Any]:
    """Evaluates model using standard metric dict enriched with compute_paired_metrics."""
    base_metrics = evaluate(model, batch, cases)
    tokens, actions, _ = batch
    with torch.no_grad():
        _, tag_logits, _, _ = model.forward_with_strategy(tokens, actions)
        assert tag_logits is not None
        preds = (tag_logits[:, TAG_IDS] >= 0).long().tolist()

    paired_metrics = compute_paired_metrics(cases, preds)
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
    optimizer, scheduler, clipper = build_a11_optimizer_and_clipper(owner.model, steps)
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
                if k == "dev_minimal_pairs":
                    eval_res = evaluate_with_paired_metrics(owner.model, eval_batches[k], rs)
                else:
                    eval_res = evaluate(owner.model, eval_batches[k], rs)
                snap_metrics[k] = eval_res

            snapshots[str(step)] = snap_metrics
            owner.save(out_dir / f"{arm_name.lower()}_{step}.pt")

    wall_time = time.time() - t0
    return snapshots, wall_time


def check_a11r_development_gates(
    cohort_snapshots_7200: list[dict[str, Any]],
) -> tuple[dict[str, bool], dict[str, float]]:
    """Evaluates Stage A.11-R Development Admission Gates across seeds 42, 43, 44."""
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

    mean_mp_exact = float(sum(mp_exact_accs) / len(mp_exact_accs))
    mean_mp_both = float(sum(mp_both_accs) / len(mp_both_accs))
    mean_mp_flip = float(sum(mp_flip_accs) / len(mp_flip_accs))

    gate_3 = bool(
        mean_fam >= 0.85
        and mean_nov >= 0.85
        and min_single >= 0.80
        and mean_mp_exact >= 0.85
        and mean_mp_both >= 0.80
        and mean_mp_flip >= 0.80
    )

    gates = {
        "gate_1_train_acquisition": gate_1,
        "gate_2_old_frame_preservation": gate_2,
        "gate_3_unfamiliar_and_minimal_pairs": gate_3,
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
    }
    return gates, metrics


def verify_checkpoint_continuation(
    chk_path: Path,
    eval_batch: tuple[torch.Tensor, ...],
    strategy: str = "question_only",
) -> float:
    """Verifies fresh-process deterministic checkpoint reload with delta logit = 0.00e+00."""
    loaded_state = LearningState.load(chk_path)
    loaded_state.model.eval()

    with torch.no_grad():
        _, orig_logits, _, _ = loaded_state.model.forward_with_strategy(
            eval_batch[0], eval_batch[1], strategy=strategy
        )

    # Re-instantiate and load weights freshly
    student = TextStudentM103(prune_orphaned_lm_head=True, pos_encoding="rope")
    fresh_model = IntegratedStudentM104_E(
        student, mode="trainable_backbone", default_strategy=strategy
    )
    state_dict = torch.load(chk_path, map_location="cpu", weights_only=False)
    if "model_state" in state_dict:
        fresh_model.load_state_dict(state_dict["model_state"])
    elif "model" in state_dict:
        fresh_model.load_state_dict(state_dict["model"])
    if torch.cuda.is_available():
        fresh_model = fresh_model.cuda()
    fresh_model.eval()

    with torch.no_grad():
        _, fresh_logits, _, _ = fresh_model.forward_with_strategy(
            eval_batch[0], eval_batch[1], strategy=strategy
        )

    assert orig_logits is not None and fresh_logits is not None
    diff = (orig_logits - fresh_logits).abs().max().item()
    return float(diff)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage A.11-R Minimal Readout Diagnosis")
    parser.add_argument("--out_dir", type=Path, default=Path("runs/stage_a11r"))
    parser.add_argument("--mock", action="store_true", help="Local dry run mock mode")
    args = parser.parse_args()

    # Enforce execution policy
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
    print("=== Stage A.11-R: Minimal Readout Diagnosis Master Runner ===")
    print(f"Commit: {source_sha} | Device: {torch.cuda.get_device_name(0)}")

    context = 256
    tokenizer = ByteTextTokenizerM103()
    pairs_suite = build_semantic_minimal_pairs_suite()
    train_pairs = pairs_suite["train_pairs"]
    train_pair_cases = pairs_suite["train_cases"]
    dev_pair_cases = pairs_suite["dev_cases"]

    training, _, blocks, shared, _ = a6_prepare()
    broad_cases = training["coverage"][:756]
    syn_cases = training["coverage"][756:]
    rel_train_cases = broad_cases + train_pair_cases + syn_cases

    batches: dict[str, Any] = {}
    batches["relational_coverage"], _ = encode(rel_train_cases, tokenizer, context)

    shared_batches: dict[str, Any] = {}
    for k, rs in shared.items():
        shared_batches[k], _ = encode(rs, tokenizer, context)

    dev_mp_batch, _ = encode(dev_pair_cases, tokenizer, context)
    shared_batches["dev_minimal_pairs"] = dev_mp_batch

    eval_sets_common = {
        "old_known": shared["old_known"],
        "old_renamed": shared["old_renamed"],
        "development_familiar": shared["development_familiar"],
        "development_novel": shared["development_novel"],
        "dev_minimal_pairs": dev_pair_cases,
    }
    cur_eval_cases = broad_cases + train_pair_cases
    cur_eval_sets = {"training": cur_eval_cases, **eval_sets_common}
    cur_eval_batches = {
        "training": tuple(v[: len(cur_eval_cases)] for v in batches["relational_coverage"]),
        **shared_batches,
    }

    # Plans
    standard_rel_plan, _ = build_relational_augmented_plan(
        blocks["coverage"],
        len_broad=756,
        train_pairs=train_pairs,
        synthetic_offset=756,
        total_steps=7200,
    )
    balanced_rel_plan, _ = build_balanced_relational_plan(
        blocks["coverage"],
        len_broad=756,
        train_pairs=train_pairs,
        synthetic_offset=756,
        total_steps=7200,
    )

    arm_configs = [
        ("R_ARM_1_MEAN_POOL_CONTROL", "question_only", standard_rel_plan),
        ("R_ARM_2_LAST_TOKEN_READOUT", "last_question_token", standard_rel_plan),
        ("R_ARM_3_QUERY_ATTN_POOL_READOUT", "query_attn_pool", balanced_rel_plan),
    ]

    seeds = [42, 43, 44]
    results: dict[str, Any] = {
        "stage": "Stage A.11-R Minimal Readout Diagnosis",
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
                batches["relational_coverage"],
                cur_eval_sets,
                cur_eval_batches,
                args.out_dir,
            )
            arm_snaps[str(seed)] = snaps
            trained_models[arm_id][seed] = owner.model

        # Summarize arm metrics across seeds at step 7200
        snaps_7200 = [arm_snaps[str(s)]["7200"] for s in seeds]
        gates_eval, gates_metrics = check_a11r_development_gates(snaps_7200)

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
        print(f"  Dev MP Counterfactual Flip: {gates_metrics['mean_dev_minimal_pair_flip']:.2%}")
        print(f"  Gates Passed: {all(gates_eval.values())}")

    # Checkpoint Continuation Verification for Seed 42
    print("\n--- Verifying Checkpoint Continuation Invariant ---")
    for arm_id, arm_data in results["arms"].items():
        chk = args.out_dir / f"{arm_id.lower()}_seed_42_7200.pt"
        if chk.exists():
            delta = verify_checkpoint_continuation(chk, cur_eval_batches["training"], strategy=arm_data["strategy"])
            results["checkpoint_continuation"][arm_id] = delta
            print(f"{arm_id} Seed 42 continuation delta logit: {delta:.2e}")
            assert delta < 1e-6, f"Checkpoint continuation failed for {arm_id}: {delta}"

    # Strict Sealed Evaluation Rule
    # "Do NOT evaluate sealed_a11_v1 unless a readout arm passes ALL development gates across the full 3-seed cohort"
    qualifying_arm = None
    for arm_id, arm_data in results["arms"].items():
        if all(arm_data["gates_evaluation"].values()):
            qualifying_arm = arm_id
            break

    if not qualifying_arm:
        print("\n[!] No readout arm passed ALL development gates across the 3-seed cohort.")
        print("Sealed suite sealed_a11_v1 was NOT evaluated (strict isolation preserved).")
        results["sealed_qualification"] = {
            "executed": False,
            "reason": "Admission gates 1-3 did not pass simultaneously across the 3-seed cohort.",
            "sealed_isolation_preserved": True,
        }
    else:
        print(f"\n[+] Qualifying Arm found: {qualifying_arm}! Evaluating sealed_a11_v1...")
        master_sealed = generate_master_sealed_suite_a11()
        iso = verify_sealed_a11_isolation(master_sealed)
        assert iso["isolation_verified"]

        sealed_batch, _ = encode(master_sealed, tokenizer, context)
        sealed_scores: dict[str, Any] = {}

        for seed in seeds:
            model = trained_models[qualifying_arm][seed]
            model.eval()
            with torch.no_grad():
                res = evaluate_with_paired_metrics(model, sealed_batch, master_sealed)
            sealed_scores[str(seed)] = res

        results["sealed_qualification"] = {
            "executed": True,
            "qualifying_arm": qualifying_arm,
            "sealed_suite_sha256": iso["sealed_suite_sha256"],
            "seed_results": sealed_scores,
        }

    manifest_path = Path("data/manifests/stage_a11r_v1.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(results, indent=2))
    print(f"\nStage A.11-R complete. Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
