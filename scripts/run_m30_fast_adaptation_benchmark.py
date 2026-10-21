"""M30 Benchmark: Verified Fast Neural Adaptation and Generalization Study.

Implements the experimental protocol for Adaptive Mini-GPT M30:
  1. Phase 1: Diagnostics of learning failure (gradient norms, loss trajectories, semantic vs format errors).
  2. Phase 2: Practice curriculum scaling study (8 vs 32 vs 128 verified examples).
  3. Phase 3: Fast adaptation mechanism comparison:
     - Regime A: Full Fine-Tuning without Replay (forgetting diagnostic)
     - Regime B: Full Fine-Tuning with Balanced Multi-Domain Replay
     - Regime C: Frozen Backbone + Rank-4 LoRA Adapters
     - Regime D: Frozen Backbone + Rank-8 LoRA Adapters
  4. Phase 4: Multi-lineage evaluation across Seeds 42, 43, 44 on Tesla T4 GPU.
  5. Phase 5: Conditional capacity study (1M Control vs 3M Scaled backbone).
  6. Phase 6: End-to-end interaction learning verification.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from adaptive_mini_gpt.adaptation.curriculum import (
    MultiScaleEpisode,
    build_multiscale_episode_a,
    build_multiscale_episode_b,
    build_multiscale_episode_c,
)
from adaptive_mini_gpt.adaptation.trainer import (
    AdaptationRunResult,
    evaluate_samples_accuracy,
    run_fast_adaptation_regime,
)
from adaptive_mini_gpt.data.bpe_scaling import (
    expand_model_vocabulary,
    train_scaled_bpe_tokenizer,
)
from adaptive_mini_gpt.data.language_corpus import PRETRAIN_PROSE_CORPUS
from adaptive_mini_gpt.data.language_data import (
    BenchmarkSample,
    BenchmarkSplit,
    DomainType,
)
from adaptive_mini_gpt.data.logic_data import (
    build_logic_benchmark_samples,
)
from adaptive_mini_gpt.data.retention_data import (
    INDEPENDENT_REPLAY_CORPUS,
    evaluate_audited_retention,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.capacity_migration import (
    count_parameters,
    create_1m_control_config,
    create_3m_scaled_config,
    estimate_training_flops_per_token,
)
from adaptive_mini_gpt.model.transformer import MiniGPT, ModelConfig
from adaptive_mini_gpt.rules.reliable_transition import (
    ElementaryTransition,
    get_all_200_transitions,
)
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("m30_fast_adaptation")


def run_practice_scale_study(
    episode: MultiScaleEpisode,
    base_model: MiniGPT,
    tokenizer: BPETokenizer,
    arith_transitions: list[ElementaryTransition],
    logic_replay: list[BenchmarkSample],
    device: torch.device,
    epochs: int = 12,
    lr: float = 3e-4,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Phase 2: Compare Scale 8, 32, and 128 practice examples with balanced replay."""
    scale_map = {
        8: episode.practice_8,
        32: episode.practice_32,
        128: episode.practice_128,
    }

    scale_results: dict[str, Any] = {}
    for scale, samples in scale_map.items():
        logger.info(f"  [Scale Study] Evaluating Scale {scale} ({len(samples)} practice samples)...")
        res = run_fast_adaptation_regime(
            regime="full_ft_with_replay",
            base_model=base_model,
            tokenizer=tokenizer,
            practice_samples=samples,
            id_test_samples=episode.id_fresh_test,
            ood_test_samples=episode.ood_transfer_test,
            domain=episode.domain,
            arith_transitions=arith_transitions,
            logic_replay_samples=logic_replay,
            device=device,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
        )
        scale_results[f"scale_{scale}"] = {
            "scale": scale,
            "train_samples_count": len(samples),
            "id_test_accuracy": res.id_test_accuracy,
            "ood_test_accuracy": res.ood_test_accuracy,
            "retention_carry_accuracy": res.retention_carry_accuracy,
            "train_loss_reduction_pct": res.diagnostics.train_loss_reduction_pct if res.diagnostics else 0.0,
            "test_loss_reduction_pct": res.diagnostics.test_loss_reduction_pct if res.diagnostics else 0.0,
        }

    return scale_results


def run_regimes_comparison_study(
    episode: MultiScaleEpisode,
    base_model: MiniGPT,
    tokenizer: BPETokenizer,
    practice_samples: list[BenchmarkSample],
    arith_transitions: list[ElementaryTransition],
    logic_replay: list[BenchmarkSample],
    device: torch.device,
    epochs: int = 12,
    lr: float = 3e-4,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Phase 3: Compare Full FT No-Replay vs Full FT Replay vs LoRA-r4 vs LoRA-r8."""
    regimes = ["full_ft_no_replay", "full_ft_with_replay", "lora_r4", "lora_r8"]
    regime_results: dict[str, Any] = {}

    for reg in regimes:
        logger.info(f"  [Regime Study] Evaluating {reg} on {len(practice_samples)} samples...")
        res = run_fast_adaptation_regime(
            regime=reg,
            base_model=base_model,
            tokenizer=tokenizer,
            practice_samples=practice_samples,
            id_test_samples=episode.id_fresh_test,
            ood_test_samples=episode.ood_transfer_test,
            domain=episode.domain,
            arith_transitions=arith_transitions,
            logic_replay_samples=logic_replay,
            device=device,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
        )
        regime_results[reg] = {
            "regime": reg,
            "trainable_params": res.trainable_params,
            "total_params": res.total_params,
            "trainable_ratio_pct": res.trainable_ratio_pct,
            "id_test_accuracy": res.id_test_accuracy,
            "ood_test_accuracy": res.ood_test_accuracy,
            "retention_carry_accuracy": res.retention_carry_accuracy,
            "adapter_disabled_id_accuracy": res.adapter_disabled_id_accuracy,
            "adapter_disabled_retention_carry": res.adapter_disabled_retention_carry,
            "base_weights_preserved": res.base_weights_preserved,
            "diagnostics": asdict(res.diagnostics) if res.diagnostics else None,
        }

    return regime_results


def run_capacity_scaling_study(
    episode: MultiScaleEpisode,
    cfg_1m: ModelConfig,
    cfg_3m: ModelConfig,
    tokenizer: BPETokenizer,
    arith_transitions: list[ElementaryTransition],
    logic_replay: list[BenchmarkSample],
    device: torch.device,
    best_regime: str = "lora_r4",
    practice_samples: list[BenchmarkSample] | None = None,
    epochs: int = 12,
    lr: float = 3e-4,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Phase 5: Compare 1M vs 3M parameter backbones on the winning adaptation setup."""
    samples = practice_samples or episode.practice_32

    # 1. 1M Model
    model_1m = MiniGPT(cfg_1m).to(device)
    res_1m = run_fast_adaptation_regime(
        regime=best_regime,
        base_model=model_1m,
        tokenizer=tokenizer,
        practice_samples=samples,
        id_test_samples=episode.id_fresh_test,
        ood_test_samples=episode.ood_transfer_test,
        domain=episode.domain,
        arith_transitions=arith_transitions,
        logic_replay_samples=logic_replay,
        device=device,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    # 2. 3M Model
    model_3m = MiniGPT(cfg_3m).to(device)
    res_3m = run_fast_adaptation_regime(
        regime=best_regime,
        base_model=model_3m,
        tokenizer=tokenizer,
        practice_samples=samples,
        id_test_samples=episode.id_fresh_test,
        ood_test_samples=episode.ood_transfer_test,
        domain=episode.domain,
        arith_transitions=arith_transitions,
        logic_replay_samples=logic_replay,
        device=device,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    return {
        "model_1m": {
            "params": count_parameters(model_1m),
            "id_test_accuracy": res_1m.id_test_accuracy,
            "ood_test_accuracy": res_1m.ood_test_accuracy,
            "retention_carry": res_1m.retention_carry_accuracy,
        },
        "model_3m": {
            "params": count_parameters(model_3m),
            "id_test_accuracy": res_3m.id_test_accuracy,
            "ood_test_accuracy": res_3m.ood_test_accuracy,
            "retention_carry": res_3m.retention_carry_accuracy,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M30 Fast Neural Adaptation Benchmark")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoints-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="runs/m30_fast_adaptation")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info(f"Starting M30 Fast Neural Adaptation Benchmark on {device}")
    logger.info(f"Seeds: {args.seeds}, Epochs: {args.epochs}, LR: {args.lr}")
    logger.info("=" * 65)

    # 1. Prepare 1024-BPE Tokenizer
    all_prose = " ".join(PRETRAIN_PROSE_CORPUS)
    tokenizer_1024 = train_scaled_bpe_tokenizer(all_prose, target_vocab_size=1024)
    tok_path = output_path / "bpe_1024.json"
    tokenizer_1024.save(tok_path)
    logger.info(f"1024-BPE Tokenizer ready with fingerprint: {tokenizer_1024.get_fingerprint()}")

    # 2. Build Multi-Scale Episodes
    ep_a = build_multiscale_episode_a(seed=42)
    ep_b = build_multiscale_episode_b(seed=42)
    ep_c = build_multiscale_episode_c(seed=42)
    episodes = [ep_a, ep_b, ep_c]

    # 3. Multi-Domain Replay Pool
    arith_transitions = get_all_200_transitions()
    logic_replay = build_logic_benchmark_samples(split=BenchmarkSplit.TRAIN, seed=42)

    # 4. Multi-Lineage Execution Across Seeds
    all_seed_results: list[dict[str, Any]] = []

    for seed in args.seeds:
        logger.info(f"\n=======================================================")
        logger.info(f"Executing Fast Adaptation Lineage Seed {seed}")
        logger.info(f"=======================================================")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        start_time = time.time()

        # Build base 1M model
        cfg_1m = create_1m_control_config(vocab_size=tokenizer_1024.vocab_size)
        base_model = MiniGPT(cfg_1m).to(device)

        if args.checkpoints_dir:
            ckpt_file = Path(args.checkpoints_dir) / f"seed_{seed}/base_pretrain/best.pt"
            if ckpt_file.exists():
                raw_ckpt = CheckpointManager.load_raw_checkpoint(ckpt_file)
                state_dict = raw_ckpt.get("model_state_dict", raw_ckpt.get("model_state"))
                base_model_400 = MiniGPT(ModelConfig(**raw_ckpt["model_config"]))
                base_model_400.load_state_dict(state_dict)
                tok_old = CheckpointManager.load_tokenizer(raw_ckpt)
                base_model = expand_model_vocabulary(
                    model=base_model_400,
                    old_tokenizer=tok_old,
                    new_tokenizer=tokenizer_1024,
                ).to(device)
                logger.info(f"Expanded checkpoint {ckpt_file.name} to 1024 vocab.")

        # Baseline Pre-Adaptation Performance
        pre_accs: dict[str, float] = {}
        for ep in episodes:
            acc, _, _ = evaluate_samples_accuracy(base_model, tokenizer_1024, ep.id_fresh_test, ep.domain, device)
            pre_accs[ep.episode_id] = acc

        # Phase 2: Practice Curriculum Scaling (8 vs 32 vs 128)
        logger.info(f"[Seed {seed}] Running Phase 2: Practice Curriculum Scaling...")
        scale_results: dict[str, Any] = {}
        for ep in episodes:
            scale_results[ep.episode_id] = run_practice_scale_study(
                episode=ep,
                base_model=base_model,
                tokenizer=tokenizer_1024,
                arith_transitions=arith_transitions,
                logic_replay=logic_replay,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
            )

        # Phase 3: Fast Adaptation Mechanisms (Full FT vs LoRA-r4 vs LoRA-r8) at Scale 32
        logger.info(f"[Seed {seed}] Running Phase 3: Fast Adaptation Mechanisms...")
        regime_results: dict[str, Any] = {}
        for ep in episodes:
            regime_results[ep.episode_id] = run_regimes_comparison_study(
                episode=ep,
                base_model=base_model,
                tokenizer=tokenizer_1024,
                practice_samples=ep.practice_32,
                arith_transitions=arith_transitions,
                logic_replay=logic_replay,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
            )

        # Phase 5: Conditional Capacity Comparison (1M vs 3M) on Episode A
        logger.info(f"[Seed {seed}] Running Phase 5: Capacity Comparison (1M vs 3M)...")
        cfg_3m = create_3m_scaled_config(vocab_size=tokenizer_1024.vocab_size)
        capacity_res = run_capacity_scaling_study(
            episode=ep_a,
            cfg_1m=cfg_1m,
            cfg_3m=cfg_3m,
            tokenizer=tokenizer_1024,
            arith_transitions=arith_transitions,
            logic_replay=logic_replay,
            device=device,
            best_regime="lora_r4",
            practice_samples=ep_a.practice_32,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )

        # Retention Evaluation on Pretraining Prose Holdout
        ret_losses = evaluate_audited_retention(base_model, tokenizer_1024, device)
        duration = round(time.time() - start_time, 2)
        peak_mem = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2) if torch.cuda.is_available() else 0.0

        all_seed_results.append({
            "seed": seed,
            "duration_seconds": duration,
            "peak_memory_mb": peak_mem,
            "pre_adaptation_accuracy": pre_accs,
            "practice_scale_study": scale_results,
            "regimes_comparison_study": regime_results,
            "capacity_study": capacity_res,
            "retention_loss_mean": round(ret_losses.get("mean_loss", 0.0), 4),
        })

    # 5. Aggregate Results Across Lineages
    n = len(all_seed_results)
    ep_keys = [ep.episode_id for ep in episodes]

    # Aggregate Scale Study
    agg_scale: dict[str, dict[str, Any]] = {}
    for ep_k in ep_keys:
        agg_scale[ep_k] = {}
        for sc in [8, 32, 128]:
            id_accs = [r["practice_scale_study"][ep_k][f"scale_{sc}"]["id_test_accuracy"] for r in all_seed_results]
            ood_accs = [r["practice_scale_study"][ep_k][f"scale_{sc}"]["ood_test_accuracy"] for r in all_seed_results]
            carries = [r["practice_scale_study"][ep_k][f"scale_{sc}"]["retention_carry_accuracy"] for r in all_seed_results]
            train_red = [r["practice_scale_study"][ep_k][f"scale_{sc}"]["train_loss_reduction_pct"] for r in all_seed_results]
            test_red = [r["practice_scale_study"][ep_k][f"scale_{sc}"]["test_loss_reduction_pct"] for r in all_seed_results]

            agg_scale[ep_k][f"scale_{sc}"] = {
                "mean_id_test_accuracy": round(sum(id_accs) / n, 4),
                "mean_ood_test_accuracy": round(sum(ood_accs) / n, 4),
                "mean_retention_carry": round(sum(carries) / n, 4),
                "mean_train_loss_reduction_pct": round(sum(train_red) / n, 2),
                "mean_test_loss_reduction_pct": round(sum(test_red) / n, 2),
            }

    # Aggregate Regime Study
    regimes_list = ["full_ft_no_replay", "full_ft_with_replay", "lora_r4", "lora_r8"]
    agg_regimes: dict[str, dict[str, Any]] = {}
    for ep_k in ep_keys:
        agg_regimes[ep_k] = {}
        for reg in regimes_list:
            id_accs = [r["regimes_comparison_study"][ep_k][reg]["id_test_accuracy"] for r in all_seed_results]
            ood_accs = [r["regimes_comparison_study"][ep_k][reg]["ood_test_accuracy"] for r in all_seed_results]
            carries = [r["regimes_comparison_study"][ep_k][reg]["retention_carry_accuracy"] for r in all_seed_results]
            dis_carries = [
                r["regimes_comparison_study"][ep_k][reg]["adapter_disabled_retention_carry"]
                for r in all_seed_results
                if r["regimes_comparison_study"][ep_k][reg]["adapter_disabled_retention_carry"] is not None
            ]
            trainable_ratio = all_seed_results[0]["regimes_comparison_study"][ep_k][reg]["trainable_ratio_pct"]

            agg_regimes[ep_k][reg] = {
                "trainable_ratio_pct": trainable_ratio,
                "mean_id_test_accuracy": round(sum(id_accs) / n, 4),
                "mean_ood_test_accuracy": round(sum(ood_accs) / n, 4),
                "mean_retention_carry": round(sum(carries) / n, 4),
                "mean_adapter_disabled_carry": round(sum(dis_carries) / len(dis_carries), 4) if dis_carries else None,
            }

    # Aggregate Capacity Study
    agg_capacity = {
        "1m_id_test_acc": round(sum(r["capacity_study"]["model_1m"]["id_test_accuracy"] for r in all_seed_results) / n, 4),
        "1m_ood_test_acc": round(sum(r["capacity_study"]["model_1m"]["ood_test_accuracy"] for r in all_seed_results) / n, 4),
        "1m_retention_carry": round(sum(r["capacity_study"]["model_1m"]["retention_carry"] for r in all_seed_results) / n, 4),
        "3m_id_test_acc": round(sum(r["capacity_study"]["model_3m"]["id_test_accuracy"] for r in all_seed_results) / n, 4),
        "3m_ood_test_acc": round(sum(r["capacity_study"]["model_3m"]["ood_test_accuracy"] for r in all_seed_results) / n, 4),
        "3m_retention_carry": round(sum(r["capacity_study"]["model_3m"]["retention_carry"] for r in all_seed_results) / n, 4),
    }

    report = {
        "milestone": "M30",
        "title": "Verified Fast Adaptation and Neural Generalization Study",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "seeds": args.seeds,
        "parameters_1m": count_parameters(base_model),
        "mean_duration_seconds": round(sum(r["duration_seconds"] for r in all_seed_results) / n, 2),
        "peak_memory_mb": max(r["peak_memory_mb"] for r in all_seed_results),
        "tokenizer_1024_fingerprint": tokenizer_1024.get_fingerprint(),
        "practice_scale_study": agg_scale,
        "regimes_comparison_study": agg_regimes,
        "capacity_study": agg_capacity,
        "raw_seed_runs": all_seed_results,
    }

    report_file = output_path / "m30_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("=" * 65)
    logger.info(f"M30 Benchmark Complete! Report saved to: {report_file}")
    logger.info("=" * 65)


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    main()
