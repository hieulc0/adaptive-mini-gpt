"""Automated M3/M4/M5 Continual Learning, Adaptive Growth & Mixture-of-Depths Benchmark.

Compares:
  - Baseline A (Dense Fine-Tuning): Fine-tunes all dense parameters on Task B.
  - Candidate D (M4 Grown Experts + Selective Symbol Masking): Spawns zero-init residual
    experts and trains on Task B while strictly protecting Task A backbone & vocabulary.
  - Candidate E (M5 Grown Experts + Mixture-of-Depths): Adds dynamic token-level sub-layer
    skipping (capacity_factor=0.5) to reduce active FLOPs/params per token.

Usage:
    uv run python scripts/run_m3_m4_experiment.py \
        --steps-stage1 500 --steps-stage2 500 --device auto --precision fp16
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

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
    evaluate_arithmetic_accuracy,
    evaluate_dataset,
)
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.training.config import TrainingConfig
from adaptive_mini_gpt.training.trainer import Trainer


def build_unified_tokenizer(text_a: str, text_b: str) -> CharacterTokenizer:
    """Build a deterministic ASCII vocabulary covering text and arithmetic symbols."""
    return CharacterTokenizer(use_base_ascii=True).fit(text_a + "\n" + text_b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M3/M4/M5 Continual Growth & MoD Ablation")
    parser.add_argument(
        "--steps-stage1", type=int, default=500, help="Steps on Task A (Shakespeare)"
    )
    parser.add_argument(
        "--steps-stage2", type=int, default=500, help="Steps on Task B (Arithmetic)"
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--output-dir", type=str, default="runs/m3_m4_ablation")
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=None,
        help="Optional external backup directory (e.g. Google Drive) for persistence.",
    )
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Task A (Shakespeare) and Task B (Procedural Arithmetic with disjoint holdout)
    text_a, fp_a = load_corpus(None)
    text_b, fp_b = generate_arithmetic_corpus(num_samples=3000, seed=42)
    tokenizer = build_unified_tokenizer(text_a, text_b)

    tokens_a = torch.tensor(tokenizer.encode(text_a), dtype=torch.long)
    tokens_b = torch.tensor(tokenizer.encode(text_b), dtype=torch.long)

    # Identify new domain symbols in Task B that never appeared in Task A
    unseen_token_ids = sorted(set(tokens_b.tolist()) - set(tokens_a.tolist()))

    train_a, val_a, _ = split_data(tokens_a)
    train_b, val_b, _ = split_data(tokens_b)

    ctx_len = 128
    ds_train_a = TokenDataset(train_a, context_length=ctx_len)
    ds_val_a = TokenDataset(val_a, context_length=ctx_len)

    equals_id = tokenizer.stoi.get("=")
    newline_id = tokenizer.stoi.get("\n")
    ds_train_b = TokenDataset(
        train_b,
        context_length=ctx_len,
        mask_prompt_before_equals=True,
        equals_token_id=equals_id,
        newline_token_id=newline_id,
    )
    ds_val_b = TokenDataset(
        val_b,
        context_length=ctx_len,
        mask_prompt_before_equals=True,
        equals_token_id=equals_id,
        newline_token_id=newline_id,
    )

    # 2. Stage 1: Train Dense Backbone on Task A (Shakespeare)
    print("=== Stage 1: Training Dense Backbone on Task A (Shakespeare) ===")
    base_model_cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=512,
        context_length=ctx_len,
        dropout=0.1,
        adaptive_enabled=False,
    )
    stage1_train_cfg = TrainingConfig(
        seed=42,
        batch_size=16,
        gradient_accumulation=2,
        learning_rate=3e-4,
        max_steps=args.steps_stage1,
        eval_every_steps=max(1, args.steps_stage1 // 3),
        device=args.device,
        precision=args.precision,
        save_dir=str(out_root / "stage1_base"),
        backup_dir=str(Path(args.backup_dir) / "stage1_base") if args.backup_dir else None,
    )

    base_model = MiniGPT(base_model_cfg)
    trainer_s1 = Trainer(
        model=base_model,
        model_config=base_model_cfg,
        training_config=stage1_train_cfg,
        train_dataset=ds_train_a,
        val_dataset=ds_val_a,
        tokenizer=tokenizer,
        dataset_fingerprint=fp_a,
    )
    trainer_s1.train()
    device = trainer_s1.device

    eval_s1_task_a = evaluate_dataset(base_model, ds_val_a, batch_size=16, device=device)
    eval_s1_task_b = evaluate_dataset(base_model, ds_val_b, batch_size=16, device=device)

    # 3. Arm 1 (Control): Dense Fine-Tuning on Task B
    print("\n=== Arm 1 (Control): Static Dense Fine-Tuning on Task B (Arithmetic) ===")
    dense_ft_model = copy.deepcopy(base_model)
    dense_ft_cfg = copy.deepcopy(base_model_cfg)
    arm1_train_cfg = TrainingConfig(
        seed=42,
        batch_size=16,
        gradient_accumulation=2,
        learning_rate=3e-4,
        max_steps=args.steps_stage2,
        eval_every_steps=max(1, args.steps_stage2 // 3),
        device=args.device,
        precision=args.precision,
        save_dir=str(out_root / "arm1_dense_control"),
        backup_dir=str(Path(args.backup_dir) / "arm1_dense_control") if args.backup_dir else None,
    )
    trainer_arm1 = Trainer(
        model=dense_ft_model,
        model_config=dense_ft_cfg,
        training_config=arm1_train_cfg,
        train_dataset=ds_train_b,
        val_dataset=ds_val_b,
        tokenizer=tokenizer,
        dataset_fingerprint=fp_b,
    )
    trainer_arm1.train()

    eval_arm1_task_a = evaluate_dataset(dense_ft_model, ds_val_a, batch_size=16, device=device)
    eval_arm1_task_b = evaluate_dataset(dense_ft_model, ds_val_b, batch_size=16, device=device)
    arith_arm1 = evaluate_arithmetic_accuracy(
        dense_ft_model, tokenizer, num_problems=100, device=device, ood_extrapolation=False
    )
    arith_arm1_ood = evaluate_arithmetic_accuracy(
        dense_ft_model, tokenizer, num_problems=100, device=device, ood_extrapolation=True
    )

    # 4. Arm 2 (M4 Adaptive Growth): Zero-Init Experts + Protected Backbone
    print("\n=== Arm 2 (M4 Adaptive Growth): Zero-Init Experts + Protected Backbone ===")
    grown_model = copy.deepcopy(base_model)
    grow_model_experts(
        grown_model,
        add_experts=4,
        freeze_backbone=True,
        freeze_old_experts=False,
        trainable_token_ids=unseen_token_ids,
    )

    eval_boundary_task_a = evaluate_dataset(grown_model, ds_val_a, batch_size=16, device=device)

    arm2_train_cfg = TrainingConfig(
        seed=42,
        batch_size=16,
        gradient_accumulation=2,
        learning_rate=1e-3,
        max_steps=args.steps_stage2,
        eval_every_steps=max(1, args.steps_stage2 // 3),
        device=args.device,
        precision=args.precision,
        save_dir=str(out_root / "arm2_adaptive_grown"),
        backup_dir=str(Path(args.backup_dir) / "arm2_adaptive_grown") if args.backup_dir else None,
    )
    trainer_arm2 = Trainer(
        model=grown_model,
        model_config=grown_model.config,
        training_config=arm2_train_cfg,
        train_dataset=ds_train_b,
        val_dataset=ds_val_b,
        tokenizer=tokenizer,
        dataset_fingerprint=fp_b,
    )
    trainer_arm2.train()

    eval_arm2_task_b = evaluate_dataset(grown_model, ds_val_b, batch_size=16, device=device)
    arith_arm2 = evaluate_arithmetic_accuracy(
        grown_model, tokenizer, num_problems=100, device=device, ood_extrapolation=False
    )
    arith_arm2_ood = evaluate_arithmetic_accuracy(
        grown_model, tokenizer, num_problems=100, device=device, ood_extrapolation=True
    )
    router_telemetry_arm2 = dict(grown_model.last_aux_info)

    forgetting_dense = eval_arm1_task_a["loss"] - eval_s1_task_a["loss"]
    eval_arm2_task_a = evaluate_dataset(grown_model, ds_val_a, batch_size=16, device=device)
    forgetting_grown_unconditioned = eval_arm2_task_a["loss"] - eval_s1_task_a["loss"]
    eval_arm2_task_a_bypassed = evaluate_dataset(
        grown_model, ds_val_a, batch_size=16, device=device, bypass_moe=True
    )
    task_conditioned_backbone_forgetting = (
        eval_arm2_task_a_bypassed["loss"] - eval_s1_task_a["loss"]
    )

    # 5. Arm 3 (M5 Mixture-of-Depths + Grown Experts): Dynamic 50% Token Skipping
    print("\n=== Arm 3 (M5 Mixture-of-Depths + MoE): 50% Dynamic Sub-Layer Capacity ===")
    mod_model = copy.deepcopy(base_model)
    grow_model_experts(
        mod_model,
        add_experts=4,
        freeze_backbone=False,
    )
    enable_model_mod(mod_model, capacity_factor=0.5)
    mod_model.freeze_backbone(
        freeze_existing_experts=False,
        trainable_token_ids=unseen_token_ids,
    )

    arm3_train_cfg = TrainingConfig(
        seed=42,
        batch_size=16,
        gradient_accumulation=2,
        learning_rate=1e-3,
        max_steps=args.steps_stage2,
        eval_every_steps=max(1, args.steps_stage2 // 3),
        device=args.device,
        precision=args.precision,
        save_dir=str(out_root / "arm3_m5_mod"),
        backup_dir=str(Path(args.backup_dir) / "arm3_m5_mod") if args.backup_dir else None,
    )
    trainer_arm3 = Trainer(
        model=mod_model,
        model_config=mod_model.config,
        training_config=arm3_train_cfg,
        train_dataset=ds_train_b,
        val_dataset=ds_val_b,
        tokenizer=tokenizer,
        dataset_fingerprint=fp_b,
    )
    trainer_arm3.train()

    eval_arm3_task_b = evaluate_dataset(mod_model, ds_val_b, batch_size=16, device=device)
    arith_arm3 = evaluate_arithmetic_accuracy(
        mod_model, tokenizer, num_problems=100, device=device, ood_extrapolation=False
    )
    telemetry_arm3 = dict(mod_model.last_aux_info)
    eval_arm3_task_a_bypassed = evaluate_dataset(
        mod_model, ds_val_a, batch_size=16, device=device, bypass_moe=True
    )

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "stage1_task_a_baseline": {
            "task_a_val_loss": round(eval_s1_task_a["loss"], 4),
            "task_a_val_ppl": round(eval_s1_task_a["perplexity"], 2),
            "task_b_zero_shot_loss": round(eval_s1_task_b["loss"], 4),
        },
        "arm1_dense_finetune_control": {
            "total_params": dense_ft_model.get_num_params(),
            "trainable_params": sum(
                p.numel() for p in dense_ft_model.parameters() if p.requires_grad
            ),
            "task_b_val_loss": round(eval_arm1_task_b["loss"], 4),
            "task_b_val_ppl": round(eval_arm1_task_b["perplexity"], 2),
            "task_b_exact_match_accuracy": arith_arm1["exact_match_accuracy"],
            "task_b_accuracy_by_op": arith_arm1.get("accuracy_by_op", {}),
            "task_b_ood_exact_match_accuracy": arith_arm1_ood["exact_match_accuracy"],
            "task_a_val_loss_after": round(eval_arm1_task_a["loss"], 4),
            "catastrophic_forgetting_delta_loss": round(forgetting_dense, 4),
        },
        "arm2_m4_adaptive_growth": {
            "total_params": grown_model.get_num_params(),
            "active_params_per_token": grown_model.get_active_params_per_token(),
            "trainable_params": sum(p.numel() for p in grown_model.parameters() if p.requires_grad),
            "unseen_domain_tokens_adapted": len(unseen_token_ids),
            "boundary_task_a_loss_drift": round(
                abs(eval_boundary_task_a["loss"] - eval_s1_task_a["loss"]), 6
            ),
            "task_b_val_loss": round(eval_arm2_task_b["loss"], 4),
            "task_b_val_ppl": round(eval_arm2_task_b["perplexity"], 2),
            "task_b_exact_match_accuracy": arith_arm2["exact_match_accuracy"],
            "task_b_accuracy_by_op": arith_arm2.get("accuracy_by_op", {}),
            "task_b_ood_exact_match_accuracy": arith_arm2_ood["exact_match_accuracy"],
            "task_a_val_loss_unconditioned": round(eval_arm2_task_a["loss"], 4),
            "unconditioned_forgetting_delta_loss": round(forgetting_grown_unconditioned, 4),
            "task_conditioned_backbone_loss": round(eval_arm2_task_a_bypassed["loss"], 4),
            "task_conditioned_backbone_forgetting": round(task_conditioned_backbone_forgetting, 6),
            "router_entropy": round(router_telemetry_arm2.get("router_entropy", 0.0), 4),
            "expert_utilization": router_telemetry_arm2.get("expert_utilization", []),
        },
        "arm3_m5_mixture_of_depths": {
            "total_params": mod_model.get_num_params(),
            "active_params_per_token": mod_model.get_active_params_per_token(),
            "mod_capacity_factor": mod_model.config.mod_capacity_factor,
            "mod_active_fraction_eval": telemetry_arm3.get("mod_active_fraction", 0.5),
            "task_b_val_loss": round(eval_arm3_task_b["loss"], 4),
            "task_b_val_ppl": round(eval_arm3_task_b["perplexity"], 2),
            "task_b_exact_match_accuracy": arith_arm3["exact_match_accuracy"],
            "task_b_accuracy_by_op": arith_arm3.get("accuracy_by_op", {}),
            "task_conditioned_backbone_forgetting": round(
                eval_arm3_task_a_bypassed["loss"] - eval_s1_task_a["loss"], 6
            ),
            "router_entropy": round(telemetry_arm3.get("router_entropy", 0.0), 4),
        },
    }

    report_path = out_root / "ablation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.backup_dir:
        backup_out = Path(args.backup_dir)
        backup_out.mkdir(parents=True, exist_ok=True)
        (backup_out / "ablation_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    print("\n=== M3/M4/M5 Ablation Summary ===")
    print(json.dumps(report, indent=2))
    print(f"\nSaved full experiment report to {report_path}")


if __name__ == "__main__":
    main()
