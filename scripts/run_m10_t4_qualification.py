"""M10 Unified Autonomous Learning Qualification Script for NVIDIA Tesla T4.

Executes:
1. Baseline pretraining (Character or BPE language backbone).
2. Cycle 1: Autonomous task selection, bounded FP16 training on T4, verification, and promotion.
3. Cycle 2: Sequential learning from promoted checkpoint, verification on next skill.
4. Rollback test: Non-improving candidate demonstrating rollback & preservation.
5. Saves unified machine-readable JSON report.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from adaptive_mini_gpt.autonomous import GrowthAction, LearningController
from adaptive_mini_gpt.data.dataset import TokenDataset, split_data
from adaptive_mini_gpt.data.tokenizer import CharacterTokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.training.config import TrainingConfig
from adaptive_mini_gpt.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m10_qualification")


def pretrain_baseline_model(
    save_dir: Path,
    device: str = "cuda",
    precision: str = "fp16",
    steps: int = 150,
) -> Path:
    """Pretrain a modest baseline model on lawful text corpus + basic arithmetic tokens."""
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Pretraining baseline model for %d steps on device=%s (%s)...", steps, device, precision
    )

    corpus_text = (
        "From fairest creatures we desire increase, That thereby beauty's rose might never die.\n"
        "When forty winters shall besiege thy brow, And dig deep trenches in thy beauty's field.\n"
        "Look in thy glass, and tell the face thou viewest Now is the time that face another.\n"
        "In the year 1594, Shakespeare joined the company of actors in London. ADD 12+34=46\n"
        "Act 2, Scene 4: Enter King Henry with 10 soldiers and 3 messengers. SUB 50-12=38\n"
        "The company performed 154 sonnets across 3 theatres before 2000 patrons. parity 42=even\n"
    ) * 40

    tokenizer = CharacterTokenizer(use_base_ascii=True).fit(corpus_text)
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_layers=3,
        n_heads=2,
        d_ff=128,
        context_length=64,
        dropout=0.0,
    )
    model = MiniGPT(cfg)

    tokens = torch.tensor(tokenizer.encode(corpus_text), dtype=torch.long)
    train_tok, val_tok, _ = split_data(tokens)
    ds_train = TokenDataset(train_tok, context_length=cfg.context_length)
    ds_val = TokenDataset(val_tok, context_length=cfg.context_length)

    train_cfg = TrainingConfig(
        seed=42,
        batch_size=16,
        gradient_accumulation=1,
        learning_rate=2e-3,
        max_steps=steps,
        eval_every_steps=steps // 2,
        device=device,
        precision=precision,
        save_dir=str(save_dir),
    )

    trainer = Trainer(
        model=model,
        model_config=cfg,
        training_config=train_cfg,
        train_dataset=ds_train,
        val_dataset=ds_val,
        tokenizer=tokenizer,
        dataset_fingerprint="pretrain_corpus_v1",
        run_id="m10_pretrain_base",
    )
    t0 = time.time()
    _ = trainer.train()
    elapsed = time.time() - t0
    logger.info("Pretraining complete in %.2f seconds.", elapsed)
    return save_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="M10 Unified Autonomous Learning T4 Qualification")
    default_dev = "cuda" if torch.cuda.is_available() else "cpu"
    default_prec = "fp16" if torch.cuda.is_available() else "fp32"
    parser.add_argument("--device", type=str, default=default_dev)
    parser.add_argument("--precision", type=str, default=default_prec)
    parser.add_argument("--output-dir", type=str, default="runs/m10_qualification")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info("Running M10 Qualification on GPU: %s | Precision: %s", gpu_name, args.precision)

    # 1. Pretrain base model
    base_dir = out_dir / "base_pretrain"
    _ = pretrain_baseline_model(base_dir, device=args.device, precision=args.precision, steps=120)

    # 2. Cycle 1: Autonomous learning of first weak task (Parity classification)
    logger.info("=== Starting Cycle 1: Autonomous Learning ===")
    cycle1_dir = out_dir / "cycle_runs"
    controller = LearningController(
        base_checkpoint_dir=base_dir,
        output_dir=cycle1_dir,
        budget_steps=100,
        max_budget_steps=1000,
        min_improvement=0.05,
        max_regression_tolerance=0.08,
        force_task="parity",
        growth_action=GrowthAction.CONTINUE,  # Dense fine-tuning control
        device=args.device,
        precision=args.precision,
        seed=42,
    )

    c1_res = controller.run_cycle()
    logger.info(
        "Cycle 1 Complete: Decision=%s, Task=%s",
        c1_res.get("decision"),
        c1_res.get("selected_task"),
    )

    # 3. Cycle 2: Sequential learning continuing from promoted checkpoint
    logger.info("=== Starting Cycle 2: Sequential Learning from Promoted Baseline ===")
    controller.prepare_next_cycle()
    c2_res = controller.run_cycle()
    logger.info(
        "Cycle 2 Complete: Decision=%s, Task=%s",
        c2_res.get("decision"),
        c2_res.get("selected_task"),
    )

    # 4. Rollback Verification Test: Force a candidate that regresses or fails threshold
    logger.info("=== Starting Rollback Test: Verifying Safe Rejection & Baseline Preservation ===")
    rb_dir = out_dir / "rollback_test"
    rb_base = (
        cycle1_dir / "active_model"
        if (cycle1_dir / "active_model" / "latest.pt").is_file()
        else base_dir
    )
    rb_controller = LearningController(
        base_checkpoint_dir=rb_base,
        output_dir=rb_dir,
        budget_steps=2,
        min_improvement=0.99,  # Unattainable in 2 steps -> triggers rollback
        max_regression_tolerance=0.00000001,
        force_task="addition",
        growth_action=GrowthAction.CONTINUE,
        device=args.device,
        precision=args.precision,
        seed=999,
    )
    rb_res = rb_controller.run_cycle()
    logger.info(
        "Rollback Test Complete: Decision=%s, Reason=%s",
        rb_res.get("decision"),
        rb_res.get("decision_reason"),
    )

    # 5. Compile Final Qualification Report
    report = {
        "timestamp": time.time(),
        "gpu_name": gpu_name,
        "device": args.device,
        "precision": args.precision,
        "cycle_1": {
            "experiment_id": c1_res.get("experiment_id"),
            "selected_task": c1_res.get("selected_task"),
            "decision": c1_res.get("decision"),
            "decision_reason": c1_res.get("decision_reason"),
            "plan": c1_res.get("plan"),
            "baseline_accuracy": (
                c1_res.get("baseline_metrics", {})
                .get("tasks", {})
                .get(str(c1_res.get("selected_task")), {})
                .get("accuracy", 0.0)
            ),
            "candidate_accuracy": (
                c1_res.get("candidate_metrics", {})
                .get("task", {})
                .get("accuracy", 0.0)
            ),
            "retention": c1_res.get("candidate_metrics", {}).get("retention", {}),
        },
        "cycle_2": {
            "experiment_id": c2_res.get("experiment_id"),
            "selected_task": c2_res.get("selected_task"),
            "decision": c2_res.get("decision"),
            "decision_reason": c2_res.get("decision_reason"),
            "plan": c2_res.get("plan"),
            "baseline_accuracy": (
                c2_res.get("baseline_metrics", {})
                .get("tasks", {})
                .get(str(c2_res.get("selected_task")), {})
                .get("accuracy", 0.0)
            ),
            "candidate_accuracy": (
                c2_res.get("candidate_metrics", {})
                .get("task", {})
                .get("accuracy", 0.0)
            ),
            "retention": c2_res.get("candidate_metrics", {}).get("retention", {}),
        },
        "rollback_test": {
            "experiment_id": rb_res.get("experiment_id"),
            "decision": rb_res.get("decision"),
            "decision_reason": rb_res.get("decision_reason"),
            "base_checkpoint_preserved": rb_res.get("decision") == "ROLLBACK",
        },
        "cumulative_telemetry": {
            "cumulative_steps": controller.journal.cumulative_steps,
            "cumulative_tokens": controller.journal.cumulative_tokens,
            "cumulative_gpu_seconds": round(controller.journal.cumulative_gpu_seconds, 2),
        },
    }

    report_path = out_dir / "m10_t4_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved M10 Qualification Report to: %s", report_path)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
