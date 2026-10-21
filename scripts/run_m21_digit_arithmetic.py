"""M21: Digit-Level Neural Arithmetic and Procedural Generalization Benchmark.

Evaluates whether changing representation and procedural supervision enables the ~1M-parameter
Mini-GPT to learn a reusable addition algorithm without external tools.

Four experimental arms:
- Arm A (Standard Text): "Add: 127 + 386 -> Answer: 513"
- Arm B (Digit-Level MSD): "Add: 1 2 7 + 3 8 6 -> Answer: 5 1 3"
- Arm C (Procedural Scratchpad): Column-by-column carry calculations -> Answer: 5 1 3
- Arm D (Digit-Level LSD): "Add (LSD): 7 2 1 + 6 8 3 -> Answer: 3 1 5"

Protocol:
- Evaluates across 3 independent lineages (seeds 42, 43, 44)
- Fixed ~1M parameter architecture (928,512 params) starting from M18 base checkpoints
- Strict train (1200) / val (200) / in-dist test (200) / extrapolation test (100) isolation
- Neural-only exact-match accuracy without external execution tools
- Per-digit accuracy, carry prediction accuracy, length breakdown, and carry breakdown
- Multi-corpus language retention monitoring
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.rules.arithmetic_dataset import (
    ArithmeticDataset,
    ArithmeticProblem,
    compute_dataset_fingerprint,
    generate_controlled_addition_dataset,
)
from adaptive_mini_gpt.rules.arithmetic_evaluator import (
    evaluate_arithmetic_unaided,
    train_arithmetic_model,
)
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m21_arithmetic")

EXPANDED_RETENTION_CORPUS = {
    "prose_original": (
        "Paris is the capital of France. William Shakespeare wrote Hamlet, Macbeth, and Othello in London. "
        "The speed of light in vacuum is approximately 300,000 km/s. "
        "In quantum physics, photons, bosons, and fermions govern fundamental interactions. "
        "Tokyo is the capital of Japan, and Rome is in Italy."
    ),
    "technical_science": (
        "Thermodynamics defines the relationships between heat, work, temperature, and energy. "
        "The first law states that energy cannot be created or destroyed, only transformed. "
        "Entropy measures molecular disorder and increases in all irreversible processes. "
        "DNA transcription and translation synthesize polypeptides from nucleotide sequences."
    ),
    "geography_history": (
        "The Nile River flows north through northeastern Africa into the Mediterranean Sea. "
        "The Roman Empire expanded across Europe, North Africa, and the Middle East before dividing. "
        "The Industrial Revolution transformed manufacturing from manual labor to machine-driven factories."
    ),
    "mathematics_prose": (
        "Prime numbers are integers greater than one divisible only by one and themselves. "
        "Euclidean geometry explores planes, angles, triangles, and parallel postulates. "
        "Differential calculus computes instantaneous rates of change along smooth curves."
    ),
}


def evaluate_language_loss(model: MiniGPT, tokenizer: BPETokenizer, device: torch.device) -> dict[str, float]:
    """Evaluate language model cross-entropy loss on multi-corpus retention suite."""
    model.eval()
    losses: dict[str, float] = {}
    with torch.no_grad():
        for name, text in EXPANDED_RETENTION_CORPUS.items():
            toks = tokenizer.encode(text)
            if len(toks) > model.config.context_length:
                toks = toks[: model.config.context_length]
            inp = torch.tensor([toks], dtype=torch.long, device=device)
            logits, _ = model(inp)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inp[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1))
            losses[name] = round(loss.item(), 4)
    return losses


def run_m21_benchmark(
    base_checkpoints_dir: Path,
    output_dir: Path,
    device_str: str = "cuda",
    seeds: list[int] = [42, 43, 44],
    epochs: int = 10,
    lr: float = 5e-4,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Execute complete M21 digit-level arithmetic and scratchpad benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    logger.info("==================================================================")
    logger.info("M21: Digit-Level Neural Arithmetic & Procedural Generalization")
    logger.info("Lineages: %s | Device: %s | Epochs: %d | LR: %e", seeds, device, epochs, lr)
    logger.info("==================================================================")

    # 1. Controlled Datasets Generation (Rigorous Partition Isolation)
    logger.info("Generating controlled datasets...")
    train_problems = generate_controlled_addition_dataset(seed=1021, num_samples=1200, lengths=(2, 3), split="train")
    val_problems = generate_controlled_addition_dataset(seed=2021, num_samples=200, lengths=(2, 3), split="val")
    test_in_dist = generate_controlled_addition_dataset(seed=3021, num_samples=200, lengths=(2, 3), split="test")
    test_extrap = generate_controlled_addition_dataset(seed=4021, num_samples=100, lengths=(4, 5), split="test_extrapolation")

    # Verify zero overlap between splits
    train_pairs = set((p.a, p.b) for p in train_problems)
    val_pairs = set((p.a, p.b) for p in val_problems)
    test_id_pairs = set((p.a, p.b) for p in test_in_dist)
    test_ood_pairs = set((p.a, p.b) for p in test_extrap)

    assert len(train_pairs & val_pairs) == 0, "Train-val overlap detected!"
    assert len(train_pairs & test_id_pairs) == 0, "Train-test ID overlap detected!"
    assert len(train_pairs & test_ood_pairs) == 0, "Train-test OOD overlap detected!"

    dataset_metadata = {
        "train": {
            "samples": len(train_problems),
            "fingerprint": compute_dataset_fingerprint(train_problems),
            "carry_distribution": {
                c: sum(1 for p in train_problems if min(p.num_carries, 3) == c) for c in range(4)
            },
        },
        "val": {
            "samples": len(val_problems),
            "fingerprint": compute_dataset_fingerprint(val_problems),
            "carry_distribution": {
                c: sum(1 for p in val_problems if min(p.num_carries, 3) == c) for c in range(4)
            },
        },
        "test_in_distribution": {
            "samples": len(test_in_dist),
            "fingerprint": compute_dataset_fingerprint(test_in_dist),
            "carry_distribution": {
                c: sum(1 for p in test_in_dist if min(p.num_carries, 3) == c) for c in range(4)
            },
        },
        "test_extrapolation": {
            "samples": len(test_extrap),
            "fingerprint": compute_dataset_fingerprint(test_extrap),
            "carry_distribution": {
                c: sum(1 for p in test_extrap if min(p.num_carries, 3) == c) for c in range(4)
            },
        },
    }
    logger.info("Dataset partitions generated successfully: %s", json.dumps(dataset_metadata, indent=2))

    arms = ["standard", "digit_msd", "scratchpad", "digit_lsd"]
    lineage_results: dict[str, Any] = {}

    for seed in seeds:
        seed_key = f"seed_{seed}"
        logger.info("==================================================================")
        logger.info("Processing Lineage %s", seed_key)
        logger.info("==================================================================")

        ckpt_path = base_checkpoints_dir / f"seed_{seed}" / "base_pretrain" / "best.pt"
        raw_ckpt = CheckpointManager.load_raw_checkpoint(ckpt_path)
        tokenizer = CheckpointManager.load_tokenizer(raw_ckpt)
        cfg_dict = raw_ckpt.get("model_config", {})
        cfg = ModelConfig(**cfg_dict) if isinstance(cfg_dict, dict) else cfg_dict

        lineage_eval: dict[str, Any] = {}

        for arm in arms:
            logger.info("--- Starting Arm: %s [%s] ---", arm, seed_key)
            torch.manual_seed(seed)

            # Fresh backbone from base pretrain
            model = MiniGPT(cfg)
            model.load_state_dict(raw_ckpt["model_state_dict"])
            model.to(device)

            retention_pre = evaluate_language_loss(model, tokenizer, device)

            # Datasets for this arm
            ds_train = ArithmeticDataset(train_problems, tokenizer, arm=arm, context_length=cfg.context_length)
            ds_val = ArithmeticDataset(val_problems, tokenizer, arm=arm, context_length=cfg.context_length)

            t0_train = time.perf_counter()
            train_metrics = train_arithmetic_model(
                model=model,
                train_dataset=ds_train,
                val_dataset=ds_val,
                device=device,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
            )
            train_duration_s = time.perf_counter() - t0_train

            # 1. In-Distribution Evaluation (2-3 digits)
            eval_in_dist = evaluate_arithmetic_unaided(
                model=model,
                tokenizer=tokenizer,
                problems=test_in_dist,
                arm=arm,
                device=device,
            )

            # 2. Extrapolation Evaluation (4-5 digits)
            eval_extrap = evaluate_arithmetic_unaided(
                model=model,
                tokenizer=tokenizer,
                problems=test_extrap,
                arm=arm,
                device=device,
            )

            # 3. Retention Evaluation
            retention_post = evaluate_language_loss(model, tokenizer, device)
            drift = {k: round(retention_post[k] - retention_pre[k], 4) for k in retention_pre}

            lineage_eval[arm] = {
                "training": {
                    "train_duration_s": round(train_duration_s, 2),
                    "tokens_trained": train_metrics["total_tokens_trained"],
                    "final_train_loss": train_metrics["final_train_loss"],
                    "final_val_loss": train_metrics["final_val_loss"],
                },
                "in_distribution": {
                    "exact_answer_accuracy": eval_in_dist.exact_answer_accuracy,
                    "digit_level_accuracy": eval_in_dist.digit_level_accuracy,
                    "syntax_validity_rate": eval_in_dist.syntax_validity_rate,
                    "carry_prediction_accuracy": eval_in_dist.carry_prediction_accuracy,
                    "accuracy_by_length": eval_in_dist.accuracy_by_length,
                    "accuracy_by_carries": eval_in_dist.accuracy_by_carries,
                    "mean_latency_ms": eval_in_dist.mean_latency_ms,
                    "loss": eval_in_dist.loss,
                },
                "extrapolation": {
                    "exact_answer_accuracy": eval_extrap.exact_answer_accuracy,
                    "digit_level_accuracy": eval_extrap.digit_level_accuracy,
                    "syntax_validity_rate": eval_extrap.syntax_validity_rate,
                    "accuracy_by_length": eval_extrap.accuracy_by_length,
                    "accuracy_by_carries": eval_extrap.accuracy_by_carries,
                    "loss": eval_extrap.loss,
                },
                "retention_drift": drift,
            }

            logger.info("Arm %s finished [%s]: In-Dist Exact=%.2f%%, Digit=%.2f%% | Extrap Exact=%.2f%% | Val Loss=%.4f",
                        arm, seed_key,
                        eval_in_dist.exact_answer_accuracy, eval_in_dist.digit_level_accuracy,
                        eval_extrap.exact_answer_accuracy, train_metrics["final_val_loss"])

            del model
            torch.cuda.empty_cache()

        lineage_results[seed_key] = lineage_eval

    # 3. Aggregate Statistical Summary
    def _mean_std(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"mean": 0.0, "std": 0.0}
        m = sum(vals) / len(vals)
        s = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
        return {"mean": round(m, 2), "std": round(s, 2)}

    aggregate_summary: dict[str, Any] = {}
    for arm in arms:
        exact_in = [lineage_results[f"seed_{s}"][arm]["in_distribution"]["exact_answer_accuracy"] for s in seeds]
        digit_in = [lineage_results[f"seed_{s}"][arm]["in_distribution"]["digit_level_accuracy"] for s in seeds]
        valid_in = [lineage_results[f"seed_{s}"][arm]["in_distribution"]["syntax_validity_rate"] for s in seeds]
        carry_in = [lineage_results[f"seed_{s}"][arm]["in_distribution"]["carry_prediction_accuracy"] for s in seeds]
        exact_extrap = [lineage_results[f"seed_{s}"][arm]["extrapolation"]["exact_answer_accuracy"] for s in seeds]
        digit_extrap = [lineage_results[f"seed_{s}"][arm]["extrapolation"]["digit_level_accuracy"] for s in seeds]
        val_loss = [lineage_results[f"seed_{s}"][arm]["training"]["final_val_loss"] for s in seeds]
        train_time = [lineage_results[f"seed_{s}"][arm]["training"]["train_duration_s"] for s in seeds]

        aggregate_summary[arm] = {
            "in_distribution": {
                "exact_answer_accuracy": _mean_std(exact_in),
                "digit_level_accuracy": _mean_std(digit_in),
                "syntax_validity_rate": _mean_std(valid_in),
                "carry_prediction_accuracy": _mean_std(carry_in),
            },
            "extrapolation": {
                "exact_answer_accuracy": _mean_std(exact_extrap),
                "digit_level_accuracy": _mean_std(digit_extrap),
            },
            "training": {
                "val_loss": _mean_std(val_loss),
                "train_duration_s": _mean_std(train_time),
            },
        }

    report = {
        "milestone": "M21",
        "description": "Digit-Level Neural Arithmetic and Procedural Generalization Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": seeds,
        "device": device_str,
        "parameters": 928512,
        "dataset_metadata": dataset_metadata,
        "aggregate_summary": aggregate_summary,
        "lineage_results": lineage_results,
    }

    report_path = output_dir / "m21_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("M21 Benchmark report successfully saved to %s", report_path)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoints-dir", type=Path, default=ROOT_DIR / "runs/kaggle/m18-checkpoints")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "runs/m21_digit_arithmetic")
    args = parser.parse_args()

    run_m21_benchmark(
        base_checkpoints_dir=args.checkpoints_dir,
        output_dir=args.output_dir,
        device_str=args.device,
        seeds=args.seeds,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
