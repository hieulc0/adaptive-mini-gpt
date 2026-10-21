"""M22: Compact Arithmetic Scratchpads and Length Generalization Benchmark.

Investigates whether shorter procedural representations and alternative positional encodings
(RoPE, ALiBi) improve Mini-GPT's exact arithmetic accuracy, carry accuracy, and length extrapolation.

Five experimental arms:
- Arm A (Verbose Scratchpad, Absolute): M21 baseline with untruncated generation limit (max_new_tokens=250)
- Arm B (Compact Scratchpad, Absolute): Compact procedural scratchpad with learned absolute embeddings
- Arm C (Compact Scratchpad, RoPE): Compact procedural scratchpad with Rotary Position Embeddings
- Arm D (Compact Scratchpad, ALiBi): Compact procedural scratchpad with Attention with Linear Biases
- Arm E (Compact Scratchpad, Absolute + 10% Replay): Compact scratchpad with language retention replay

Protocol:
- 3 independent pretraining lineages (seeds 42, 43, 44)
- Strict pairwise disjointness across train (1200), val (200), test in-dist (200, 2-3 digits),
  and test extrapolation (150, 4-6 digits).
- Comprehensive error decomposition: Truncation rate, Syntax validity rate, Carry prediction accuracy,
  Digit-level accuracy, and Exact-answer accuracy.
- Multi-corpus language retention monitoring (cross-entropy drift on 4 distinct corpora).
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
logger = logging.getLogger("m22_arithmetic")

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


def run_m22_benchmark(
    base_checkpoints_dir: Path,
    output_dir: Path,
    device_str: str = "cuda",
    seeds: list[int] = [42, 43, 44],
    epochs: int = 10,
    lr: float = 5e-4,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Execute complete M22 compact arithmetic, positional encoding, and length generalization benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    logger.info("==================================================================")
    logger.info("M22: Compact Scratchpads, RoPE, ALiBi & Length Generalization")
    logger.info("Lineages: %s | Device: %s | Epochs: %d | LR: %e", seeds, device, epochs, lr)
    logger.info("==================================================================")

    # 1. Controlled Datasets Generation (Rigorous Partition Isolation)
    logger.info("Generating controlled datasets...")
    train_problems = generate_controlled_addition_dataset(seed=1022, num_samples=1200, lengths=(2, 3), split="train")
    train_pairs = set((min(p.a, p.b), max(p.a, p.b)) for p in train_problems)

    val_problems = generate_controlled_addition_dataset(
        seed=2022, num_samples=200, lengths=(2, 3), split="val", exclude_pairs=train_pairs
    )
    val_pairs = set((min(p.a, p.b), max(p.a, p.b)) for p in val_problems)

    test_in_dist = generate_controlled_addition_dataset(
        seed=3022, num_samples=200, lengths=(2, 3), split="test", exclude_pairs=train_pairs | val_pairs
    )
    test_id_pairs = set((min(p.a, p.b), max(p.a, p.b)) for p in test_in_dist)

    test_extrap = generate_controlled_addition_dataset(
        seed=4022,
        num_samples=150,
        lengths=(4, 5, 6),
        split="test_extrapolation",
        exclude_pairs=train_pairs | val_pairs | test_id_pairs,
    )
    test_ood_pairs = set((min(p.a, p.b), max(p.a, p.b)) for p in test_extrap)

    # Verify zero overlap between splits
    assert len(train_pairs & val_pairs) == 0, "Train-val overlap detected!"
    assert len(train_pairs & test_id_pairs) == 0, "Train-test ID overlap detected!"
    assert len(train_pairs & test_ood_pairs) == 0, "Train-test OOD overlap detected!"
    assert len(val_pairs & test_id_pairs) == 0, "Val-test ID overlap detected!"
    assert len(val_pairs & test_ood_pairs) == 0, "Val-test OOD overlap detected!"
    assert len(test_id_pairs & test_ood_pairs) == 0, "Test ID-OOD overlap detected!"

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
            "length_distribution": {
                l: sum(1 for p in test_extrap if p.max_length == l) for l in (4, 5, 6)
            },
        },
    }
    logger.info("Dataset partitions generated successfully: %s", json.dumps(dataset_metadata, indent=2))

    replay_corpus_list = list(EXPANDED_RETENTION_CORPUS.values())

    arms_config = [
        {"name": "arm_a_verbose_abs", "arm_format": "scratchpad", "pos_encoding": "absolute", "replay_ratio": 0.0},
        {"name": "arm_b_compact_abs", "arm_format": "compact_scratchpad", "pos_encoding": "absolute", "replay_ratio": 0.0},
        {"name": "arm_c_compact_rope", "arm_format": "compact_scratchpad", "pos_encoding": "rope", "replay_ratio": 0.0},
        {"name": "arm_d_compact_alibi", "arm_format": "compact_scratchpad", "pos_encoding": "alibi", "replay_ratio": 0.0},
        {"name": "arm_e_compact_replay", "arm_format": "compact_scratchpad", "pos_encoding": "absolute", "replay_ratio": 0.1},
    ]

    lineage_results: dict[str, Any] = {}

    for seed in seeds:
        seed_key = f"seed_{seed}"
        logger.info("==================================================================")
        logger.info("Processing Lineage %s", seed_key)
        logger.info("==================================================================")

        ckpt_path = base_checkpoints_dir / f"seed_{seed}" / "base_pretrain" / "best.pt"
        raw_ckpt = CheckpointManager.load_raw_checkpoint(ckpt_path)
        tokenizer = CheckpointManager.load_tokenizer(raw_ckpt)
        base_cfg_dict = raw_ckpt.get("model_config", {})
        if not isinstance(base_cfg_dict, dict):
            base_cfg_dict = base_cfg_dict.to_dict()

        lineage_eval: dict[str, Any] = {}

        for arm_spec in arms_config:
            arm_name = arm_spec["name"]
            arm_format = arm_spec["arm_format"]
            pos_enc = arm_spec["pos_encoding"]
            rep_ratio = arm_spec["replay_ratio"]

            logger.info("--- Starting Arm: %s (pos=%s, format=%s, replay=%.1f) [%s] ---",
                        arm_name, pos_enc, arm_format, rep_ratio, seed_key)
            torch.manual_seed(seed)

            # Build config with designated pos_encoding
            cfg_dict = dict(base_cfg_dict)
            cfg_dict["pos_encoding"] = pos_enc
            cfg = ModelConfig(**cfg_dict)

            # Initialize model and load pretrained weights (wpe ignored gracefully if RoPE/ALiBi)
            model = MiniGPT(cfg)
            model.load_state_dict(raw_ckpt["model_state_dict"], strict=False)
            model.to(device)

            param_count = sum(p.numel() for p in model.parameters())

            retention_pre = evaluate_language_loss(model, tokenizer, device)

            # Build datasets
            ds_train = ArithmeticDataset(
                train_problems,
                tokenizer,
                arm=arm_format,
                context_length=cfg.context_length,
                replay_texts=replay_corpus_list if rep_ratio > 0.0 else None,
                replay_ratio=rep_ratio,
                seed=seed,
            )
            ds_val = ArithmeticDataset(val_problems, tokenizer, arm=arm_format, context_length=cfg.context_length)

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

            # In-Distribution Evaluation (max_new_tokens=250 allows untruncated verbose generation)
            eval_in_dist = evaluate_arithmetic_unaided(
                model=model,
                tokenizer=tokenizer,
                problems=test_in_dist,
                arm=arm_format,
                device=device,
                max_new_tokens=250,
            )

            # Extrapolation Evaluation (4-6 digits)
            eval_extrap = evaluate_arithmetic_unaided(
                model=model,
                tokenizer=tokenizer,
                problems=test_extrap,
                arm=arm_format,
                device=device,
                max_new_tokens=250,
            )

            # Retention Evaluation
            retention_post = evaluate_language_loss(model, tokenizer, device)
            drift = {k: round(retention_post[k] - retention_pre[k], 4) for k in retention_pre}

            lineage_eval[arm_name] = {
                "parameters": param_count,
                "pos_encoding": pos_enc,
                "format": arm_format,
                "replay_ratio": rep_ratio,
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
                    "truncation_rate": eval_in_dist.truncation_rate,
                    "accuracy_by_length": eval_in_dist.accuracy_by_length,
                    "accuracy_by_carries": eval_in_dist.accuracy_by_carries,
                    "mean_latency_ms": eval_in_dist.mean_latency_ms,
                    "loss": eval_in_dist.loss,
                },
                "extrapolation": {
                    "exact_answer_accuracy": eval_extrap.exact_answer_accuracy,
                    "digit_level_accuracy": eval_extrap.digit_level_accuracy,
                    "syntax_validity_rate": eval_extrap.syntax_validity_rate,
                    "carry_prediction_accuracy": eval_extrap.carry_prediction_accuracy,
                    "truncation_rate": eval_extrap.truncation_rate,
                    "accuracy_by_length": eval_extrap.accuracy_by_length,
                    "accuracy_by_carries": eval_extrap.accuracy_by_carries,
                    "loss": eval_extrap.loss,
                },
                "retention_drift": drift,
            }

            logger.info("Arm %s finished [%s]: In Exact=%.2f%% (Trunc=%.1f%%) | Extrap Exact=%.2f%% (Trunc=%.1f%%) | Val Loss=%.4f",
                        arm_name, seed_key,
                        eval_in_dist.exact_answer_accuracy, eval_in_dist.truncation_rate,
                        eval_extrap.exact_answer_accuracy, eval_extrap.truncation_rate,
                        train_metrics["final_val_loss"])

            del model
            torch.cuda.empty_cache()

        lineage_results[seed_key] = lineage_eval

    # Statistical Summary across lineages
    def _mean_std(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"mean": 0.0, "std": 0.0}
        m = sum(vals) / len(vals)
        s = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
        return {"mean": round(m, 2), "std": round(s, 2)}

    arm_names = [a["name"] for a in arms_config]
    aggregate_summary: dict[str, Any] = {}
    for an in arm_names:
        exact_in = [lineage_results[f"seed_{s}"][an]["in_distribution"]["exact_answer_accuracy"] for s in seeds]
        digit_in = [lineage_results[f"seed_{s}"][an]["in_distribution"]["digit_level_accuracy"] for s in seeds]
        valid_in = [lineage_results[f"seed_{s}"][an]["in_distribution"]["syntax_validity_rate"] for s in seeds]
        carry_in = [lineage_results[f"seed_{s}"][an]["in_distribution"]["carry_prediction_accuracy"] for s in seeds]
        trunc_in = [lineage_results[f"seed_{s}"][an]["in_distribution"]["truncation_rate"] for s in seeds]

        exact_extrap = [lineage_results[f"seed_{s}"][an]["extrapolation"]["exact_answer_accuracy"] for s in seeds]
        digit_extrap = [lineage_results[f"seed_{s}"][an]["extrapolation"]["digit_level_accuracy"] for s in seeds]
        valid_extrap = [lineage_results[f"seed_{s}"][an]["extrapolation"]["syntax_validity_rate"] for s in seeds]
        trunc_extrap = [lineage_results[f"seed_{s}"][an]["extrapolation"]["truncation_rate"] for s in seeds]

        val_loss = [lineage_results[f"seed_{s}"][an]["training"]["final_val_loss"] for s in seeds]
        train_time = [lineage_results[f"seed_{s}"][an]["training"]["train_duration_s"] for s in seeds]
        mean_drift = [
            sum(lineage_results[f"seed_{s}"][an]["retention_drift"].values()) / 4.0 for s in seeds
        ]

        aggregate_summary[an] = {
            "parameters": lineage_results[f"seed_{seeds[0]}"][an]["parameters"],
            "pos_encoding": lineage_results[f"seed_{seeds[0]}"][an]["pos_encoding"],
            "in_distribution": {
                "exact_answer_accuracy": _mean_std(exact_in),
                "digit_level_accuracy": _mean_std(digit_in),
                "syntax_validity_rate": _mean_std(valid_in),
                "carry_prediction_accuracy": _mean_std(carry_in),
                "truncation_rate": _mean_std(trunc_in),
            },
            "extrapolation": {
                "exact_answer_accuracy": _mean_std(exact_extrap),
                "digit_level_accuracy": _mean_std(digit_extrap),
                "syntax_validity_rate": _mean_std(valid_extrap),
                "truncation_rate": _mean_std(trunc_extrap),
            },
            "training": {
                "val_loss": _mean_std(val_loss),
                "train_duration_s": _mean_std(train_time),
                "mean_retention_drift": _mean_std(mean_drift),
            },
        }

    report = {
        "milestone": "M22",
        "description": "Compact Arithmetic Scratchpads, RoPE, ALiBi & Length Generalization Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": seeds,
        "device": device_str,
        "dataset_metadata": dataset_metadata,
        "aggregate_summary": aggregate_summary,
        "lineage_results": lineage_results,
    }

    report_path = output_dir / "m22_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("M22 Benchmark report successfully saved to %s", report_path)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoints-dir", type=Path, default=ROOT_DIR / "runs/kaggle/m18-checkpoints")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "runs/m22_compact_arithmetic")
    args = parser.parse_args()

    run_m22_benchmark(
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
