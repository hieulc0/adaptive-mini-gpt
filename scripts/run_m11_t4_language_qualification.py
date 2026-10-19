"""M11 Autonomous Language Learning Qualification Script for NVIDIA Tesla T4.

Executes:
1. Pretraining of a genuine BPE subword language model backbone on T4.
2. Cycle 1: Autonomous learning of Skill 1 (instruction_symbolic) -> Verification & Promotion.
3. Cycle 2: Sequential learning of Skill 2 (instruction_qa) from active pointer -> Promotion.
4. Cycle 3: Rollback verification test on impossible threshold -> Safe Rejection & Preservation.
5. Qualitative evaluation: Actual generated responses before and after learning.
6. Cumulative telemetry: Steps, tokens, GPU training seconds vs total wall-clock seconds.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from adaptive_mini_gpt.autonomous import GrowthAction, LearningController
from adaptive_mini_gpt.autonomous.tasks import TASK_REGISTRY, evaluate_task_holdout
from adaptive_mini_gpt.data.instruction_data import (
    SEED_INSTRUCTION_EXAMPLES,
    InstructionDataset,
    format_instruction_pair,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.training.checkpoint import CheckpointManager
from adaptive_mini_gpt.training.config import TrainingConfig
from adaptive_mini_gpt.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m11_qualification")


def pretrain_bpe_language_model(
    save_dir: Path,
    device: str = "cuda",
    precision: str = "fp16",
    steps: int = 150,
) -> tuple[Path, BPETokenizer, ModelConfig]:
    """Pretrain a modest BPE dense Transformer on curated natural language corpus."""
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Pretraining BPE language backbone for %d steps on device=%s (%s)...",
        steps,
        device,
        precision,
    )

    corpus_prose = (
        "From fairest creatures we desire increase, That thereby beauty's rose might never die.\n"
        "When forty winters shall besiege thy brow, And dig deep trenches in thy beauty's field.\n"
        "William Shakespeare wrote Hamlet, Macbeth, Othello, and King Lear in London.\n"
        "In quantum physics, photons, bosons, and fermions govern fundamental interactions.\n"
        "Paris is the capital of France, Tokyo is the capital of Japan, and Rome is in Italy.\n"
        "The speed of light in vacuum is approximately 300,000 km/s.\n"
    ) * 30

    instruction_pairs = []
    for ex in SEED_INSTRUCTION_EXAMPLES:
        instruction_pairs.append(format_instruction_pair(ex["instruction"], ex["response"]))
    full_corpus = corpus_prose + "".join(instruction_pairs) * 10

    # Train BPE Tokenizer
    tokenizer = BPETokenizer()
    tokenizer.train(full_corpus, target_vocab_size=350)
    logger.info("Trained BPE Tokenizer with vocabulary size: %d", tokenizer.vocab_size)

    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        context_length=128,
        dropout=0.0,
    )
    model = MiniGPT(cfg)

    # Encode with BPE
    tokens = tokenizer.encode(full_corpus)
    split_idx = int(len(tokens) * 0.9)
    train_tokens = tokens[:split_idx]
    val_tokens = tokens[split_idx:]

    ds_train = InstructionDataset(
        token_ids=train_tokens,
        context_length=cfg.context_length,
        assistant_token_id=tokenizer.assistant_id,
        eos_token_id=tokenizer.eos_id,
        user_token_id=tokenizer.user_id,
    )
    ds_val = InstructionDataset(
        token_ids=val_tokens,
        context_length=cfg.context_length,
        assistant_token_id=tokenizer.assistant_id,
        eos_token_id=tokenizer.eos_id,
        user_token_id=tokenizer.user_id,
    )

    train_cfg = TrainingConfig(
        seed=42,
        batch_size=16,
        gradient_accumulation=1,
        learning_rate=1.5e-3,
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
        dataset_fingerprint="bpe_pretrain_v1",
        run_id="m11_bpe_pretrain",
    )
    t0 = time.time()
    _ = trainer.train()
    elapsed = time.time() - t0
    logger.info("BPE Pretraining complete in %.2f seconds.", elapsed)
    return save_dir, tokenizer, cfg


def generate_qualitative_sample(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    prompt: str,
    device: torch.device,
) -> str:
    """Autoregressively generate assistant response for qualitative inspection."""
    model.eval()
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(48):
            if input_tensor.size(1) >= model.config.context_length:
                break
            logits, _ = model(input_tensor)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            if next_token.item() == tokenizer.eos_id:
                break

    full_text = tokenizer.decode(input_tensor[0].tolist())
    completion = full_text[len(prompt) :].replace("<|eos|>", "").strip()
    return completion


def main() -> int:
    parser = argparse.ArgumentParser(description="M11 Autonomous Language Learning Qualification")
    default_dev = "cuda" if torch.cuda.is_available() else "cpu"
    default_prec = "fp16" if torch.cuda.is_available() else "fp32"
    parser.add_argument("--device", type=str, default=default_dev)
    parser.add_argument("--precision", type=str, default=default_prec)
    parser.add_argument("--output-dir", type=str, default="runs/m11_qualification")
    args = parser.parse_args()

    overall_start_time = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info("Running M11 Qualification on GPU: %s | Precision: %s", gpu_name, args.precision)

    # 1. Pretrain base BPE language model
    base_dir = out_dir / "base_pretrain"
    base_dir, tokenizer, cfg = pretrain_bpe_language_model(
        base_dir, device=args.device, precision=args.precision, steps=150
    )

    # Sample generation before learning
    device_obj = torch.device(args.device)
    base_ckpt = CheckpointManager.find_latest_valid_checkpoint(base_dir)
    raw_base = CheckpointManager.load_raw_checkpoint(base_ckpt)
    test_model = MiniGPT(cfg).to(device_obj)
    test_model.load_state_dict(raw_base["model_state_dict"])

    test_prompt_sym = (
        "<|user|>\nReverse the sequence of characters in the word 'photon'.\n<|assistant|>\n"
    )
    test_prompt_qa = "<|user|>\nWhat is the capital of France?\n<|assistant|>\n"

    sample_before_sym = generate_qualitative_sample(
        test_model, tokenizer, test_prompt_sym, device_obj
    )
    sample_before_qa = generate_qualitative_sample(
        test_model, tokenizer, test_prompt_qa, device_obj
    )

    # 2. Cycle 1: Autonomous learning of Skill 1 (instruction_symbolic)
    logger.info("=== Starting Cycle 1: Autonomous Learning of Symbolic Manipulation ===")
    cycle_dir = out_dir / "cycle_runs"
    controller = LearningController(
        base_checkpoint_dir=base_dir,
        output_dir=cycle_dir,
        budget_steps=120,
        max_budget_steps=2000,
        min_improvement=0.05,
        max_regression_tolerance=0.08,
        force_task="instruction_symbolic",
        growth_action=GrowthAction.CONTINUE,  # Dense control
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

    # 3. Cycle 2: Sequential learning of Skill 2 (instruction_qa) from promoted active pointer
    logger.info("=== Starting Cycle 2: Sequential Learning of Factual QA from Active Pointer ===")
    controller.force_task = "instruction_qa"
    controller.prepare_next_cycle()
    c2_res = controller.run_cycle()
    logger.info(
        "Cycle 2 Complete: Decision=%s, Task=%s",
        c2_res.get("decision"),
        c2_res.get("selected_task"),
    )

    # 4. Cycle 3: Rollback verification test on impossible threshold
    logger.info("=== Starting Cycle 3: Verifying Rollback and Base Preservation ===")
    controller.force_task = "instruction_symbolic"
    controller.min_improvement = 0.99  # Impossible gain threshold in 2 steps
    controller.prepare_next_cycle()
    c3_res = controller.run_cycle()
    logger.info(
        "Cycle 3 Complete: Decision=%s, Reason=%s",
        c3_res.get("decision"),
        c3_res.get("decision_reason"),
    )

    # 5. Evaluate final promoted model on untouched holdouts & qualitative samples
    promoted_ckpt = CheckpointManager.find_latest_valid_checkpoint(cycle_dir / "active_model")
    raw_promoted = CheckpointManager.load_raw_checkpoint(promoted_ckpt)
    promoted_model = MiniGPT(cfg).to(device_obj)
    promoted_model.load_state_dict(raw_promoted["model_state_dict"])

    sample_after_sym = generate_qualitative_sample(
        promoted_model, tokenizer, test_prompt_sym, device_obj
    )
    sample_after_qa = generate_qualitative_sample(
        promoted_model, tokenizer, test_prompt_qa, device_obj
    )

    # Measure untouched holdouts on 500 samples
    sym_eval_500 = evaluate_task_holdout(
        promoted_model,
        tokenizer,
        TASK_REGISTRY["instruction_symbolic"],
        device_obj,
        num_samples=500,
        seed=12345,
    )
    qa_eval_500 = evaluate_task_holdout(
        promoted_model,
        tokenizer,
        TASK_REGISTRY["instruction_qa"],
        device_obj,
        num_samples=500,
        seed=12345,
    )

    total_wall_time = time.time() - overall_start_time

    # 6. Compile Report
    report = {
        "timestamp": time.time(),
        "gpu_name": gpu_name,
        "device": args.device,
        "precision": args.precision,
        "model_architecture": {
            "vocab_size": cfg.vocab_size,
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "d_ff": cfg.d_ff,
            "context_length": cfg.context_length,
        },
        "cycle_1_symbolic": {
            "experiment_id": c1_res.get("experiment_id"),
            "selected_task": c1_res.get("selected_task"),
            "decision": c1_res.get("decision"),
            "decision_reason": c1_res.get("decision_reason"),
            "baseline_accuracy": (
                c1_res.get("baseline_metrics", {})
                .get("tasks", {})
                .get("instruction_symbolic", {})
                .get("accuracy", 0.0)
            ),
            "candidate_accuracy": (
                c1_res.get("candidate_metrics", {}).get("task", {}).get("accuracy", 0.0)
            ),
            "retention": c1_res.get("candidate_metrics", {}).get("retention", {}),
        },
        "cycle_2_factual_qa": {
            "experiment_id": c2_res.get("experiment_id"),
            "selected_task": c2_res.get("selected_task"),
            "decision": c2_res.get("decision"),
            "decision_reason": c2_res.get("decision_reason"),
            "baseline_accuracy": (
                c2_res.get("baseline_metrics", {})
                .get("tasks", {})
                .get("instruction_qa", {})
                .get("accuracy", 0.0)
            ),
            "candidate_accuracy": (
                c2_res.get("candidate_metrics", {}).get("task", {}).get("accuracy", 0.0)
            ),
            "retention": c2_res.get("candidate_metrics", {}).get("retention", {}),
        },
        "cycle_3_rollback": {
            "experiment_id": c3_res.get("experiment_id"),
            "decision": c3_res.get("decision"),
            "decision_reason": c3_res.get("decision_reason"),
            "base_checkpoint_preserved": c3_res.get("decision") == "ROLLBACK",
        },
        "final_untouched_500_sample_holdouts": {
            "instruction_symbolic_accuracy": sym_eval_500.get("accuracy", 0.0),
            "instruction_qa_accuracy": qa_eval_500.get("accuracy", 0.0),
        },
        "qualitative_generation_comparison": {
            "symbolic_reversal": {
                "prompt": test_prompt_sym.strip(),
                "before_learning": sample_before_sym,
                "after_learning": sample_after_sym,
                "expected": "notohp",
            },
            "factual_qa": {
                "prompt": test_prompt_qa.strip(),
                "before_learning": sample_before_qa,
                "after_learning": sample_after_qa,
                "expected_contains": "Paris",
            },
        },
        "timing_and_compute": {
            "cumulative_training_steps": controller.journal.cumulative_steps,
            "cumulative_training_tokens": controller.journal.cumulative_tokens,
            "pure_gpu_training_seconds": round(controller.journal.cumulative_gpu_seconds, 2),
            "total_wall_clock_seconds": round(total_wall_time, 2),
        },
    }

    report_path = out_dir / "m11_t4_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved M11 Qualification Report to: %s", report_path)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
