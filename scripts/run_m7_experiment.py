"""M7: Rigorous Generalization, Retention & Architecture Verification Benchmark.

Evaluates 4 architectures across multiple seeds:
  - Arm 1: Static Dense Fine-Tuning (Control)
  - Arm 2: Modular MoE + moe_head (Candidate D)
  - Arm 3: MoE + Mixture-of-Depths (Candidate E, 50% capacity)
  - Arm 4: Compute-Matched Dense Adapter Control (Frozen Backbone + moe_head only)

Multi-Domain Retention:
  - Clean Prose (Shakespeare, 0 digits)
  - Numerical Prose (Dates, scene numbers, quantities without math operators)
  - Mixed Prose + Math (Prose ending in equations)

Fine-Grained Arithmetic:
  - ADD_NO_CARRY, ADD_CARRY, SUB_NO_BORROW, SUB_BORROW, PARITY, OOD_EXTRAPOLATION (3-digit)

Usage:
    uv run python scripts/run_m7_experiment.py \
        --seeds 42 43 44 \
        --steps-stage1 500 \
        --steps-stage2 500 \
        --device auto \
        --precision fp16 \
        --output-dir runs/m7_benchmark
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from adaptive_mini_gpt.adaptive.growth import enable_model_mod, grow_model_experts
from adaptive_mini_gpt.data.dataset import (
    TokenDataset,
    generate_arithmetic_corpus,
    load_corpus,
    split_data,
)
from adaptive_mini_gpt.data.tokenizer import CharacterTokenizer
from adaptive_mini_gpt.evaluation.evaluator import (
    evaluate_arithmetic_detailed,
    evaluate_dataset,
    evaluate_text_retention_suite,
)
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import LowRankOutputAdapter, MiniGPT
from adaptive_mini_gpt.training.config import TrainingConfig
from adaptive_mini_gpt.training.trainer import Trainer


def compute_mean_std(values: list[float]) -> dict[str, float]:
    """Compute mean and sample standard deviation (0.0 if len == 1)."""
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": round(values[0], 4), "std": 0.0}
    return {
        "mean": round(statistics.mean(values), 4),
        "std": round(statistics.stdev(values), 4),
    }


def build_unified_tokenizer(text_a: str, text_b: str) -> CharacterTokenizer:
    return CharacterTokenizer(use_base_ascii=True).fit(text_a + "\n" + text_b)


def run_single_seed(
    seed: int,
    args: argparse.Namespace,
    tokenizer: CharacterTokenizer,
    tokens_a: torch.Tensor,
    tokens_b: torch.Tensor,
    fp_a: str,
    fp_b: str,
    device: torch.device,
    unseen_token_ids: list[int],
    out_root: Path,
) -> dict[str, Any]:
    print("\n==========================================")
    print(f"=== Running M7 Evaluation for Seed: {seed} ===")
    print("==========================================")
    torch.manual_seed(seed)

    # Prepare datasets
    train_a, val_a, _ = split_data(tokens_a)
    train_b, val_b, _ = split_data(tokens_b)

    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        context_length=128,
        dropout=0.1,
        tie_word_embeddings=False,
    )

    ds_train_a = TokenDataset(train_a, context_length=cfg.context_length)
    ds_val_a = TokenDataset(val_a, context_length=cfg.context_length)
    ds_train_b = TokenDataset(
        train_b,
        context_length=cfg.context_length,
        mask_prompt_before_equals=True,
        equals_token_id=tokenizer.stoi["="],
        newline_token_id=tokenizer.stoi["\n"],
    )
    ds_val_b = TokenDataset(
        val_b,
        context_length=cfg.context_length,
        mask_prompt_before_equals=True,
        equals_token_id=tokenizer.stoi["="],
        newline_token_id=tokenizer.stoi["\n"],
    )

    # 1. Stage 1: Pretrain Dense Backbone on Task A (Shakespeare)
    t0_stage1 = time.time()
    base_model = MiniGPT(cfg).to(device)
    train_cfg_s1 = TrainingConfig(
        seed=seed,
        batch_size=16,
        gradient_accumulation=2,
        learning_rate=3e-3,
        max_steps=args.steps_stage1,
        eval_every_steps=max(1, args.steps_stage1 // 3),
        device=str(device),
        precision=args.precision,
        save_dir=str(out_root / f"seed_{seed}" / "stage1"),
    )
    trainer_s1 = Trainer(
        model=base_model,
        model_config=base_model.config,
        training_config=train_cfg_s1,
        train_dataset=ds_train_a,
        val_dataset=ds_val_a,
        tokenizer=tokenizer,
        dataset_fingerprint=fp_a,
    )
    trainer_s1.train()
    stage1_wall_clock = time.time() - t0_stage1

    eval_s1_a = evaluate_dataset(base_model, ds_val_a, batch_size=16, device=device)
    s1_retention_suite = evaluate_text_retention_suite(base_model, tokenizer, device=device)

    # 2. Arm 1: Dense Fine-Tuning Control (All parameters update on Task B)
    print(f"\n--- [Seed {seed}] Arm 1: Dense Fine-Tuning Control ---")
    t0_arm1 = time.time()
    arm1_model = copy.deepcopy(base_model)
    arm1_cfg = TrainingConfig(
        seed=seed,
        batch_size=16,
        gradient_accumulation=2,
        learning_rate=1e-3,
        max_steps=args.steps_stage2,
        eval_every_steps=max(1, args.steps_stage2 // 3),
        device=str(device),
        precision=args.precision,
        save_dir=str(out_root / f"seed_{seed}" / "arm1_dense_ft"),
    )
    trainer_arm1 = Trainer(
        model=arm1_model,
        model_config=arm1_model.config,
        training_config=arm1_cfg,
        train_dataset=ds_train_b,
        val_dataset=ds_val_b,
        tokenizer=tokenizer,
        dataset_fingerprint=fp_b,
    )
    trainer_arm1.train()
    arm1_wall_clock = time.time() - t0_arm1

    eval_arm1_b = evaluate_dataset(arm1_model, ds_val_b, batch_size=16, device=device)
    arith_arm1 = evaluate_arithmetic_detailed(arm1_model, tokenizer, device=device)
    retention_arm1 = evaluate_text_retention_suite(arm1_model, tokenizer, device=device)

    # 3. Arm 2: Modular MoE + moe_head (Candidate D)
    print(f"\n--- [Seed {seed}] Arm 2: Modular MoE + Residual Output Adapter ---")
    t0_arm2 = time.time()
    arm2_model = copy.deepcopy(base_model)
    grow_model_experts(
        arm2_model,
        add_experts=4,
        freeze_backbone=True,
        freeze_old_experts=False,
        trainable_token_ids=unseen_token_ids,
    )
    # Configure domain trigger token to '=' so prose with numbers is not disturbed
    arm2_model.config.domain_trigger_token_ids = [tokenizer.stoi["="]]

    arm2_cfg = TrainingConfig(
        seed=seed,
        batch_size=16,
        gradient_accumulation=2,
        learning_rate=1e-3,
        max_steps=args.steps_stage2,
        eval_every_steps=max(1, args.steps_stage2 // 3),
        device=str(device),
        precision=args.precision,
        save_dir=str(out_root / f"seed_{seed}" / "arm2_moe"),
    )
    trainer_arm2 = Trainer(
        model=arm2_model,
        model_config=arm2_model.config,
        training_config=arm2_cfg,
        train_dataset=ds_train_b,
        val_dataset=ds_val_b,
        tokenizer=tokenizer,
        dataset_fingerprint=fp_b,
    )
    trainer_arm2.train()
    arm2_wall_clock = time.time() - t0_arm2

    eval_arm2_b = evaluate_dataset(arm2_model, ds_val_b, batch_size=16, device=device)
    arith_arm2 = evaluate_arithmetic_detailed(arm2_model, tokenizer, device=device)
    retention_arm2 = evaluate_text_retention_suite(arm2_model, tokenizer, device=device)

    # 4. Arm 3: MoE + Mixture-of-Depths (Candidate E, 50% capacity)
    print(f"\n--- [Seed {seed}] Arm 3: MoE + Mixture-of-Depths (50% capacity) ---")
    t0_arm3 = time.time()
    arm3_model = copy.deepcopy(base_model)
    grow_model_experts(
        arm3_model,
        add_experts=4,
        freeze_backbone=False,
    )
    enable_model_mod(arm3_model, capacity_factor=0.5)
    arm3_model.freeze_backbone(
        freeze_existing_experts=False,
        trainable_token_ids=unseen_token_ids,
    )
    arm3_model.config.domain_trigger_token_ids = [tokenizer.stoi["="]]

    arm3_cfg = TrainingConfig(
        seed=seed,
        batch_size=16,
        gradient_accumulation=2,
        learning_rate=1e-3,
        max_steps=args.steps_stage2,
        eval_every_steps=max(1, args.steps_stage2 // 3),
        device=str(device),
        precision=args.precision,
        save_dir=str(out_root / f"seed_{seed}" / "arm3_mod"),
    )
    trainer_arm3 = Trainer(
        model=arm3_model,
        model_config=arm3_model.config,
        training_config=arm3_cfg,
        train_dataset=ds_train_b,
        val_dataset=ds_val_b,
        tokenizer=tokenizer,
        dataset_fingerprint=fp_b,
    )
    trainer_arm3.train()
    arm3_wall_clock = time.time() - t0_arm3

    eval_arm3_b = evaluate_dataset(arm3_model, ds_val_b, batch_size=16, device=device)
    arith_arm3 = evaluate_arithmetic_detailed(arm3_model, tokenizer, device=device)
    retention_arm3 = evaluate_text_retention_suite(arm3_model, tokenizer, device=device)

    # 5. Arm 4: Compute-Matched Dense Adapter Control (Frozen Backbone + moe_head only)
    print(f"\n--- [Seed {seed}] Arm 4: Dense Adapter Control (No-Growth MoE Baseline) ---")
    t0_arm4 = time.time()
    arm4_model = copy.deepcopy(base_model)
    arm4_model.config.adaptive_enabled = True
    arm4_model.moe_head = LowRankOutputAdapter(
        cfg.d_model, cfg.d_expert, cfg.vocab_size
    ).to(device=device)
    arm4_model.freeze_backbone(trainable_token_ids=unseen_token_ids)
    arm4_model.config.domain_trigger_token_ids = [tokenizer.stoi["="]]

    arm4_cfg = TrainingConfig(
        seed=seed,
        batch_size=16,
        gradient_accumulation=2,
        learning_rate=1e-3,
        max_steps=args.steps_stage2,
        eval_every_steps=max(1, args.steps_stage2 // 3),
        device=str(device),
        precision=args.precision,
        save_dir=str(out_root / f"seed_{seed}" / "arm4_adapter_control"),
    )
    trainer_arm4 = Trainer(
        model=arm4_model,
        model_config=arm4_model.config,
        training_config=arm4_cfg,
        train_dataset=ds_train_b,
        val_dataset=ds_val_b,
        tokenizer=tokenizer,
        dataset_fingerprint=fp_b,
    )
    trainer_arm4.train()
    arm4_wall_clock = time.time() - t0_arm4

    eval_arm4_b = evaluate_dataset(arm4_model, ds_val_b, batch_size=16, device=device)
    arith_arm4 = evaluate_arithmetic_detailed(arm4_model, tokenizer, device=device)
    retention_arm4 = evaluate_text_retention_suite(arm4_model, tokenizer, device=device)

    return {
        "seed": seed,
        "stage1_baseline": {
            "val_loss": round(eval_s1_a["loss"], 4),
            "val_ppl": round(eval_s1_a["perplexity"], 2),
            "wall_clock_s": round(stage1_wall_clock, 2),
            "retention_suite": s1_retention_suite,
        },
        "arm1_dense_finetune": {
            "total_params": arm1_model.get_num_params(),
            "active_params": arm1_model.get_active_params_per_token(),
            "trainable_params": sum(p.numel() for p in arm1_model.parameters() if p.requires_grad),
            "val_loss": round(eval_arm1_b["loss"], 4),
            "val_ppl": round(eval_arm1_b["perplexity"], 2),
            "wall_clock_s": round(arm1_wall_clock, 2),
            "arithmetic": arith_arm1,
            "retention": retention_arm1,
        },
        "arm2_m4_moe": {
            "total_params": arm2_model.get_num_params(),
            "active_params": arm2_model.get_active_params_per_token(),
            "trainable_params": sum(p.numel() for p in arm2_model.parameters() if p.requires_grad),
            "val_loss": round(eval_arm2_b["loss"], 4),
            "val_ppl": round(eval_arm2_b["perplexity"], 2),
            "wall_clock_s": round(arm2_wall_clock, 2),
            "arithmetic": arith_arm2,
            "retention": retention_arm2,
        },
        "arm3_m5_mod": {
            "total_params": arm3_model.get_num_params(),
            "active_params": arm3_model.get_active_params_per_token(),
            "trainable_params": sum(p.numel() for p in arm3_model.parameters() if p.requires_grad),
            "val_loss": round(eval_arm3_b["loss"], 4),
            "val_ppl": round(eval_arm3_b["perplexity"], 2),
            "wall_clock_s": round(arm3_wall_clock, 2),
            "arithmetic": arith_arm3,
            "retention": retention_arm3,
        },
        "arm4_adapter_control": {
            "total_params": arm4_model.get_num_params(),
            "active_params": arm4_model.get_active_params_per_token(),
            "trainable_params": sum(p.numel() for p in arm4_model.parameters() if p.requires_grad),
            "val_loss": round(eval_arm4_b["loss"], 4),
            "val_ppl": round(eval_arm4_b["perplexity"], 2),
            "wall_clock_s": round(arm4_wall_clock, 2),
            "arithmetic": arith_arm4,
            "retention": retention_arm4,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M7 Multi-Seed Architecture & Generalization Benchmark"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Random seeds for multi-seed ablation.",
    )
    parser.add_argument("--steps-stage1", type=int, default=500, help="Steps on Task A.")
    parser.add_argument("--steps-stage2", type=int, default=500, help="Steps on Task B.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--output-dir", type=str, default="runs/m7_benchmark")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Initializing M7 Benchmark across seeds: {args.seeds} on {device} ({args.precision})")

    # Load Stage 1 Shakespeare & Stage 2 Diverse Procedural Arithmetic
    text_a, fp_a = load_corpus()
    text_b, fp_b = generate_arithmetic_corpus(num_samples=10000, seed=42)
    tokenizer = build_unified_tokenizer(text_a, text_b)

    tokens_a = torch.tensor(tokenizer.encode(text_a), dtype=torch.long)
    tokens_b = torch.tensor(tokenizer.encode(text_b), dtype=torch.long)

    # Identify unseen arithmetic tokens
    seen_in_a = set(tokens_a.tolist())
    unseen_token_ids = [tid for tid in set(tokens_b.tolist()) if tid not in seen_in_a]
    unseen_token_ids.sort()

    seed_results: list[dict[str, Any]] = []
    for seed in args.seeds:
        res = run_single_seed(
            seed=seed,
            args=args,
            tokenizer=tokenizer,
            tokens_a=tokens_a,
            tokens_b=tokens_b,
            fp_a=fp_a,
            fp_b=fp_b,
            device=device,
            unseen_token_ids=unseen_token_ids,
            out_root=out_root,
        )
        seed_results.append(res)

    # Compute Aggregate Multi-Seed Statistics
    arms = ["arm1_dense_finetune", "arm2_m4_moe", "arm3_m5_mod", "arm4_adapter_control"]
    aggregate_stats: dict[str, Any] = {}

    for arm in arms:
        val_losses = [s[arm]["val_loss"] for s in seed_results]
        in_dom_accs = [s[arm]["arithmetic"]["in_domain_accuracy"] for s in seed_results]
        ood_accs = [s[arm]["arithmetic"]["ood_accuracy"] for s in seed_results]
        clean_deltas = [s[arm]["retention"]["clean_prose"]["delta_loss"] for s in seed_results]
        num_deltas = [s[arm]["retention"]["numerical_prose"]["delta_loss"] for s in seed_results]
        mixed_deltas = [s[arm]["retention"]["mixed_prose_math"]["delta_loss"] for s in seed_results]

        cat_names = [
            "ADD_NO_CARRY",
            "ADD_CARRY",
            "SUB_NO_BORROW",
            "SUB_BORROW",
            "PARITY",
            "OOD_EXTRAPOLATION",
        ]
        cat_stats = {
            cat: compute_mean_std(
                [s[arm]["arithmetic"]["accuracy_by_category"].get(cat, 0.0) for s in seed_results]
            )
            for cat in cat_names
        }

        aggregate_stats[arm] = {
            "val_loss": compute_mean_std(val_losses),
            "in_domain_accuracy": compute_mean_std(in_dom_accs),
            "ood_accuracy": compute_mean_std(ood_accs),
            "clean_prose_delta_loss": compute_mean_std(clean_deltas),
            "numerical_prose_delta_loss": compute_mean_std(num_deltas),
            "mixed_prose_delta_loss": compute_mean_std(mixed_deltas),
            "accuracy_by_category": cat_stats,
            "total_params": seed_results[0][arm]["total_params"],
            "active_params": seed_results[0][arm]["active_params"],
            "trainable_params": seed_results[0][arm]["trainable_params"],
        }

    full_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "precision": args.precision,
        "seeds": args.seeds,
        "steps_stage1": args.steps_stage1,
        "steps_stage2": args.steps_stage2,
        "unseen_token_ids_adapted": unseen_token_ids,
        "domain_trigger_token_ids": [tokenizer.stoi["="]],
        "multi_seed_aggregate": aggregate_stats,
        "per_seed_runs": seed_results,
    }

    report_path = out_root / "m7_multi_seed_report.json"
    report_path.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"\nSaved full M7 multi-seed report to {report_path}")
    print("\n=== M7 Multi-Seed Summary (Mean ± Std) ===")
    print(json.dumps(aggregate_stats, indent=2))


if __name__ == "__main__":
    main()
