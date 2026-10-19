"""Command-line interface for Adaptive Mini-GPT (amg)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from adaptive_mini_gpt.adaptive.growth import (
    enable_model_mod,
    grow_model_depth,
    grow_model_experts,
)
from adaptive_mini_gpt.autonomous import GrowthAction, LearningController
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
    generate_sample,
)
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.training.checkpoint import CheckpointManager
from adaptive_mini_gpt.training.config import TrainingConfig
from adaptive_mini_gpt.training.trainer import Trainer, build_cosine_lr_schedule


def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    model_dict = dict(data.get("model", {}))
    # Normalize nested adaptive block if present
    if "adaptive" in data and isinstance(data["adaptive"], dict):
        adaptive_dict = data["adaptive"]
        if "enabled" in adaptive_dict:
            model_dict["adaptive_enabled"] = adaptive_dict["enabled"]
        for k in (
            "n_experts",
            "top_k",
            "d_expert",
            "routing_mode",
            "aux_loss_weight",
            "z_loss_weight",
        ):
            if k in adaptive_dict and k not in model_dict:
                model_dict[k] = adaptive_dict[k]

    return {
        "model": model_dict,
        "training": dict(data.get("training", {})),
    }


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _load_task_corpus(data_arg: str | None) -> tuple[str, str]:
    """Load corpus from file, synthetic arithmetic ('synthetic:arithmetic'), or Shakespeare."""
    if data_arg == "synthetic:arithmetic":
        return generate_arithmetic_corpus(num_samples=4000, seed=42)
    return load_corpus(data_arg)


def cmd_train(args: argparse.Namespace) -> int:
    cfg_data = _load_yaml_config(args.config)

    # Apply CLI overrides to training config
    train_dict = cfg_data["training"]
    if args.max_steps is not None:
        train_dict["max_steps"] = args.max_steps
    if args.save_dir is not None:
        train_dict["save_dir"] = args.save_dir
    if args.backup_dir is not None:
        train_dict["backup_dir"] = args.backup_dir
    if args.device is not None:
        train_dict["device"] = args.device
    if args.precision is not None:
        train_dict["precision"] = args.precision
    if getattr(args, "task_id", None) is not None:
        train_dict["task_id"] = args.task_id
    if getattr(args, "json_output", False):
        train_dict["json_output"] = True

    training_config = TrainingConfig.from_dict(train_dict)
    text, data_fp = _load_task_corpus(args.data)

    # If resuming or starting from a grown checkpoint, use its saved vocabulary
    if args.resume is not None:
        ckpt_path = CheckpointManager.find_latest_valid_checkpoint(
            args.resume,
            search_backup=Path(training_config.backup_dir) if training_config.backup_dir else None,
        )
        raw_ckpt = CheckpointManager.load_raw_checkpoint(ckpt_path)
        tokenizer = CharacterTokenizer.from_dict(raw_ckpt["tokenizer"])
        model_dict = raw_ckpt.get("model_config", cfg_data["model"])
        model_config = ModelConfig.from_dict(model_dict)
    else:
        tokenizer = CharacterTokenizer(use_base_ascii=True).fit(text)
        cfg_data["model"]["vocab_size"] = tokenizer.vocab_size
        model_config = ModelConfig.from_dict(cfg_data["model"])

    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_tokens, val_tokens, _ = split_data(tokens)

    if len(val_tokens) <= model_config.context_length:
        val_tokens = train_tokens if len(train_tokens) > model_config.context_length else tokens

    mask_prompt = (args.data == "synthetic:arithmetic") or getattr(args, "mask_prompt", False)
    equals_id = tokenizer.stoi.get("=")
    newline_id = tokenizer.stoi.get("\n")

    train_ds = TokenDataset(
        train_tokens,
        context_length=model_config.context_length,
        mask_prompt_before_equals=mask_prompt,
        equals_token_id=equals_id,
        newline_token_id=newline_id,
    )
    val_ds = TokenDataset(
        val_tokens,
        context_length=model_config.context_length,
        mask_prompt_before_equals=mask_prompt,
        equals_token_id=equals_id,
        newline_token_id=newline_id,
    )

    model = MiniGPT(model_config)
    if args.freeze_backbone:
        model.freeze_backbone(freeze_existing_experts=args.freeze_old_experts)

    trainer = Trainer(
        model=model,
        model_config=model_config,
        training_config=training_config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        tokenizer=tokenizer,
        dataset_fingerprint=data_fp,
    )

    if args.resume is not None:
        trainer.resume_from_checkpoint(
            args.resume,
            strict_dataset_match=not args.allow_new_dataset,
            allow_growth=args.allow_new_dataset,
        )
        # When transitioning to a new dataset/stage in continual learning:
        if args.allow_new_dataset:
            trainer.step = 0
            trainer.data_position = 0
            trainer.best_val_loss = float("inf")
        # Re-apply freeze flags and rebuild optimizer/scheduler for trainable subset
        if args.freeze_backbone or args.allow_new_dataset:
            if args.freeze_backbone:
                model.freeze_backbone(freeze_existing_experts=args.freeze_old_experts)
            trainer.optimizer = model.configure_optimizers(
                learning_rate=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
            )
            if training_config.lr_schedule == "cosine":
                trainer.scheduler = build_cosine_lr_schedule(
                    trainer.optimizer,
                    warmup_steps=training_config.warmup_steps,
                    max_steps=training_config.max_steps,
                    min_lr_ratio=training_config.min_lr / training_config.learning_rate,
                )

    summary = trainer.train()
    print(json.dumps(summary, indent=None if getattr(args, "json_output", False) else 2))
    return 0


def cmd_grow(args: argparse.Namespace) -> int:
    """Execute a function-preserving M4 growth transformation on a saved checkpoint."""
    ckpt_path = CheckpointManager.find_latest_valid_checkpoint(args.checkpoint)
    raw_ckpt = CheckpointManager.load_raw_checkpoint(ckpt_path)

    model_config = ModelConfig.from_dict(raw_ckpt["model_config"])
    training_config = TrainingConfig.from_dict(raw_ckpt.get("training_config", {}))
    tokenizer = CharacterTokenizer.from_dict(raw_ckpt["tokenizer"])

    model = MiniGPT(model_config)
    model.load_state_dict(raw_ckpt["model_state_dict"])
    model.eval()

    # Verify function-preserving invariant on a deterministic probe batch
    torch.manual_seed(42)
    probe_x = torch.randint(0, model_config.vocab_size, (2, min(32, model_config.context_length)))
    with torch.no_grad():
        logits_before, _ = model(probe_x)

    params_before = model.get_num_params()

    if args.routing_mode is not None:
        model.config.routing_mode = args.routing_mode

    if args.add_experts > 0:
        grow_model_experts(
            model,
            add_experts=args.add_experts,
            freeze_backbone=args.freeze_backbone,
            freeze_old_experts=args.freeze_old_experts,
        )

    if args.add_layers > 0:
        grow_model_depth(model, add_layers=args.add_layers)

    if getattr(args, "enable_mod", False):
        enable_model_mod(model, capacity_factor=getattr(args, "mod_capacity", 0.5))

    model.eval()
    with torch.no_grad():
        logits_after, _ = model(probe_x)

    max_drift = (logits_before - logits_after).abs().max().item()
    if not getattr(args, "enable_mod", False) and max_drift > 1e-5:
        raise RuntimeError(
            f"Function-preserving growth check failed! Max logit drift: {max_drift:.2e}"
        )

    params_after = model.get_num_params()
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    out_path = Path(args.output)
    out_dir = out_path.parent
    ckpt_mgr = CheckpointManager(save_dir=out_dir)

    optimizer = model.configure_optimizers(
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    reset_step = getattr(args, "reset_step", True)
    step_to_save = 0 if reset_step else raw_ckpt.get("step", 0)
    best_loss_to_save = float("inf") if reset_step else raw_ckpt.get("best_val_loss", float("inf"))

    saved_path = ckpt_mgr.save_checkpoint(
        checkpoint_name=out_path.name,
        model=model,
        optimizer=optimizer,
        model_config=model.config,
        training_config=training_config,
        tokenizer=tokenizer,
        dataset_fingerprint=raw_ckpt.get("dataset_fingerprint", ""),
        step=step_to_save,
        tokens_seen=raw_ckpt.get("tokens_seen", 0),
        data_position=0,
        best_val_loss=best_loss_to_save,
        run_id=f"{raw_ckpt.get('run_id', 'run')}_grown",
        is_best=False,
    )

    report = {
        "status": "growth_verified",
        "source_checkpoint": str(ckpt_path),
        "output_checkpoint": str(saved_path),
        "params_before": params_before,
        "params_after": params_after,
        "trainable_params": trainable_after,
        "active_params_per_token": model.get_active_params_per_token(),
        "n_layers": model.config.n_layers,
        "n_experts": model.config.n_experts if model.config.adaptive_enabled else 0,
        "mod_enabled": model.config.mod_enabled,
        "mod_capacity_factor": model.config.mod_capacity_factor,
        "max_logit_drift_at_boundary": max_drift,
    }
    print(json.dumps(report, indent=None if getattr(args, "json_output", False) else 2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    device = _resolve_device(args.device)
    ckpt_path = CheckpointManager.find_latest_valid_checkpoint(args.checkpoint)
    raw_ckpt = CheckpointManager.load_raw_checkpoint(ckpt_path)

    model_config = ModelConfig.from_dict(raw_ckpt["model_config"])
    tokenizer = CharacterTokenizer.from_dict(raw_ckpt["tokenizer"])

    model = MiniGPT(model_config).to(device)
    model.load_state_dict(raw_ckpt["model_state_dict"])

    text, _ = _load_task_corpus(args.data)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    _, val_tokens, _ = split_data(tokens)
    if len(val_tokens) <= model_config.context_length:
        val_tokens = tokens
    val_ds = TokenDataset(val_tokens, context_length=model_config.context_length)

    bypass_moe = getattr(args, "bypass_moe", False)
    metrics = evaluate_dataset(
        model=model,
        dataset=val_ds,
        batch_size=args.batch_size,
        device=device,
        max_batches=args.max_batches,
        task_id=args.task_id,
        bypass_moe=bypass_moe,
    )
    if model.config.adaptive_enabled and model.last_aux_info and not bypass_moe:
        metrics["router_entropy"] = model.last_aux_info.get("router_entropy", 0.0)
        metrics["expert_utilization"] = model.last_aux_info.get("expert_utilization", [])

    if getattr(args, "exact_match", False) or args.data == "synthetic:arithmetic":
        arith_res = evaluate_arithmetic_accuracy(
            model=model,
            tokenizer=tokenizer,
            num_problems=100,
            device=device,
            task_id=args.task_id,
            ood_extrapolation=getattr(args, "ood_extrapolation", False),
            bypass_moe=bypass_moe,
        )
        metrics["exact_match_accuracy"] = arith_res["exact_match_accuracy"]
        metrics["accuracy_by_op"] = arith_res.get("accuracy_by_op", {})
        metrics["exact_match_correct"] = arith_res["correct"]
        metrics["exact_match_total"] = arith_res["total_problems"]
        metrics["ood_extrapolation"] = arith_res["ood_extrapolation"]
        metrics["sample_predictions"] = arith_res["sample_predictions"]

    if getattr(args, "m7_suite", False):
        from adaptive_mini_gpt.evaluation.evaluator import (
            evaluate_arithmetic_detailed,
            evaluate_text_retention_suite,
        )

        metrics["m7_arithmetic_detailed"] = evaluate_arithmetic_detailed(
            model=model,
            tokenizer=tokenizer,
            num_per_category=20,
            device=device,
            task_id=args.task_id,
            bypass_moe=bypass_moe,
        )
        metrics["m7_prose_retention_suite"] = evaluate_text_retention_suite(
            model=model,
            tokenizer=tokenizer,
            device=device,
        )

    print(json.dumps(metrics, indent=None if getattr(args, "json_output", False) else 2))
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    device = _resolve_device(args.device)
    ckpt_path = CheckpointManager.find_latest_valid_checkpoint(args.checkpoint)
    raw_ckpt = CheckpointManager.load_raw_checkpoint(ckpt_path)

    model_config = ModelConfig.from_dict(raw_ckpt["model_config"])
    tokenizer = CharacterTokenizer.from_dict(raw_ckpt["tokenizer"])

    model = MiniGPT(model_config).to(device)
    model.load_state_dict(raw_ckpt["model_state_dict"])

    res = generate_sample(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        stop_on_newline=args.stop_on_newline,
        device=device,
        task_id=args.task_id,
        bypass_moe=getattr(args, "bypass_moe", False),
    )

    print(json.dumps(res, indent=None if getattr(args, "json_output", False) else 2))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    if args.checkpoint is not None:
        ckpt_path = CheckpointManager.find_latest_valid_checkpoint(args.checkpoint)
        raw_ckpt = CheckpointManager.load_raw_checkpoint(ckpt_path)
        model_config = ModelConfig.from_dict(raw_ckpt["model_config"])
        model = MiniGPT(model_config)
        model.load_state_dict(raw_ckpt["model_state_dict"])
    else:
        cfg_data = _load_yaml_config(args.config)
        model_config = ModelConfig.from_dict(cfg_data["model"])
        model = MiniGPT(model_config)

    arch_info = model.inspect_architecture()
    print(json.dumps(arch_info, indent=None if getattr(args, "json_output", False) else 2))
    return 0


def cmd_auto_learn(args: argparse.Namespace) -> int:
    """Execute autonomous learning controller cycle (M8)."""
    controller = LearningController(
        base_checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        budget_steps=args.budget_steps,
        min_improvement=args.min_improvement,
        max_regression_tolerance=args.max_regression,
        target_accuracy_threshold=args.target_threshold,
        growth_action=GrowthAction(args.growth_action),
        force_task=args.force_task,
        device=args.device,
        precision=args.precision,
        seed=args.seed,
    )
    result = controller.run_cycle()
    print(json.dumps(result, indent=None if getattr(args, "json_output", False) else 2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--json-output",
        "--json-log",
        "--json",
        dest="json_output",
        action="store_true",
        help="Emit single-line machine-readable JSON telemetry.",
    )

    parser = argparse.ArgumentParser(
        prog="amg",
        description="Adaptive Mini-GPT CLI for training, modular growth, and evaluation.",
        parents=[common_parser],
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = subparsers.add_parser(
        "train",
        help="Train a dense baseline or modular model.",
        parents=[common_parser],
    )
    p_train.add_argument("--config", type=str, default="configs/tiny.yaml")
    p_train.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume.")
    p_train.add_argument("--data", type=str, default=None, help="Dataset path.")
    p_train.add_argument("--save-dir", type=str, default=None)
    p_train.add_argument("--backup-dir", type=str, default=None)
    p_train.add_argument("--max-steps", type=int, default=None)
    p_train.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    p_train.add_argument("--precision", type=str, default=None, choices=["fp32", "fp16", "bf16"])
    p_train.add_argument(
        "--task-id", type=int, default=None, help="Task ID for task-conditioned routing."
    )
    p_train.add_argument(
        "--freeze-backbone", action="store_true", help="Freeze shared dense backbone."
    )
    p_train.add_argument(
        "--freeze-old-experts",
        action="store_true",
        help="Freeze previously trained experts.",
    )
    p_train.add_argument(
        "--allow-new-dataset",
        action="store_true",
        help="Allow resuming a checkpoint onto a new task dataset for continual learning.",
    )
    p_train.add_argument(
        "--mask-prompt",
        action="store_true",
        help="Mask prompt characters before '=' so only answer tokens are supervised.",
    )
    p_train.set_defaults(func=cmd_train)

    # grow (M4/M5)
    p_grow = subparsers.add_parser(
        "grow",
        help="Function-preserving structural growth (experts or depth) on a checkpoint.",
        parents=[common_parser],
    )
    p_grow.add_argument("--checkpoint", type=str, required=True, help="Input checkpoint path.")
    p_grow.add_argument("--output", type=str, required=True, help="Output grown checkpoint path.")
    p_grow.add_argument(
        "--add-experts", type=int, default=0, help="Number of zero-init experts to add."
    )
    p_grow.add_argument(
        "--add-layers", type=int, default=0, help="Number of identity blocks to add."
    )
    p_grow.add_argument("--routing-mode", type=str, default=None, choices=["learned_topk", "task"])
    p_grow.add_argument(
        "--freeze-backbone", action="store_true", help="Freeze shared dense weights."
    )
    p_grow.add_argument(
        "--freeze-old-experts", action="store_true", help="Freeze existing experts."
    )
    p_grow.add_argument(
        "--enable-mod",
        action="store_true",
        help="Enable M5 Mixture-of-Depths dynamic token routing.",
    )
    p_grow.add_argument(
        "--mod-capacity",
        type=float,
        default=0.5,
        help="Capacity factor (0.0, 1.0] for M5 Mixture-of-Depths (default 0.5).",
    )
    p_grow.add_argument(
        "--reset-step",
        action="store_true",
        default=True,
        help="Reset step counter to 0 for next training stage (default True).",
    )
    p_grow.add_argument(
        "--no-reset-step",
        dest="reset_step",
        action="store_false",
        help="Preserve prior stage step count.",
    )
    p_grow.set_defaults(func=cmd_grow)

    # evaluate
    p_eval = subparsers.add_parser(
        "evaluate",
        help="Evaluate a checkpoint on validation split.",
        parents=[common_parser],
    )
    p_eval.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.")
    p_eval.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional dataset path or 'synthetic:arithmetic'.",
    )
    p_eval.add_argument("--batch-size", type=int, default=16)
    p_eval.add_argument("--max-batches", "--eval-steps", dest="max_batches", type=int, default=20)
    p_eval.add_argument(
        "--task-id", type=int, default=None, help="Task ID for task-conditioned routing."
    )
    p_eval.add_argument(
        "--bypass-moe",
        action="store_true",
        help="Bypass MoE layers (evaluate pure dense backbone).",
    )
    p_eval.add_argument(
        "--exact-match",
        action="store_true",
        help="Run exact-match arithmetic/parity verification.",
    )
    p_eval.add_argument(
        "--ood-extrapolation",
        action="store_true",
        help="Evaluate on out-of-distribution [100, 200) operands.",
    )
    p_eval.add_argument(
        "--suite",
        "--m7-suite",
        dest="m7_suite",
        action="store_true",
        help="Run comprehensive M7 suite (fine-grained arithmetic + multi-domain prose).",
    )
    p_eval.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p_eval.set_defaults(func=cmd_evaluate)

    # generate
    p_gen = subparsers.add_parser(
        "generate",
        help="Generate text from a checkpoint.",
        parents=[common_parser],
    )
    p_gen.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.")
    p_gen.add_argument("--prompt", type=str, default="From fairest ")
    p_gen.add_argument(
        "--max-new-tokens", "--max-tokens", dest="max_new_tokens", type=int, default=100
    )
    p_gen.add_argument("--temperature", type=float, default=0.8)
    p_gen.add_argument("--top-k", type=int, default=40)
    p_gen.add_argument(
        "--top-p", type=float, default=None, help="Top-p (nucleus) filtering threshold."
    )
    p_gen.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty applied to context tokens.",
    )
    p_gen.add_argument(
        "--stop-on-newline", action="store_true", help="Stop generation on newline character."
    )
    p_gen.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p_gen.add_argument(
        "--task-id", type=int, default=None, help="Task ID for task-conditioned routing."
    )
    p_gen.add_argument(
        "--bypass-moe",
        action="store_true",
        help="Bypass MoE layers (generate using pure dense backbone).",
    )
    p_gen.set_defaults(func=cmd_generate)

    # inspect
    p_insp = subparsers.add_parser(
        "inspect",
        help="Inspect model architecture and parameter counts.",
        parents=[common_parser],
    )
    p_insp.add_argument("--config", type=str, default="configs/tiny.yaml")
    p_insp.add_argument("--checkpoint", type=str, default=None)
    p_insp.set_defaults(func=cmd_inspect)

    # auto-learn (M8)
    p_auto = subparsers.add_parser(
        "auto-learn",
        help="Run autonomous learning controller cycle (M8).",
        parents=[common_parser],
    )
    p_auto.add_argument(
        "--checkpoint-dir",
        "--checkpoint",
        dest="checkpoint_dir",
        type=str,
        required=True,
        help="Path to active model checkpoint or directory.",
    )
    p_auto.add_argument(
        "--output-dir",
        type=str,
        default="runs/auto_learn",
        help="Directory to store candidate runs, journal, and promoted models.",
    )
    p_auto.add_argument(
        "--budget-steps",
        type=int,
        default=200,
        help="Compute budget: max training steps for candidate.",
    )
    p_auto.add_argument(
        "--min-improvement",
        type=float,
        default=0.05,
        help="Minimum target accuracy improvement required to promote.",
    )
    p_auto.add_argument(
        "--max-regression",
        type=float,
        default=0.05,
        help="Maximum tolerable retention loss regression on prior domains.",
    )
    p_auto.add_argument(
        "--target-threshold",
        type=float,
        default=0.95,
        help="Target skill accuracy threshold for considering a skill mastered.",
    )
    p_auto.add_argument(
        "--growth-action",
        type=str,
        default="auto",
        choices=["auto", "continue", "grow_expert"],
        help="Structural adaptation strategy.",
    )
    p_auto.add_argument(
        "--force-task",
        type=str,
        default=None,
        help="Force learning of a specific task (e.g. 'addition', 'subtraction', 'parity').",
    )
    p_auto.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p_auto.add_argument("--precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    p_auto.add_argument("--seed", type=int, default=42)
    p_auto.set_defaults(func=cmd_auto_learn)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
