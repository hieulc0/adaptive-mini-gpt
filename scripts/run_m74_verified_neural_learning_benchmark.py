"""Milestone M74: Verified Neural Weight Learning and High-Information Generalization Benchmark.

Executes:
1. Phase 1: M73 Forensic Scientific Audit & Exact Tensor-Level Parameter Lineage Verification (928,512 -> 974,658).
2. Phase 2: Forward & Backward Neural Pathway Instrumentation (hooks, gradient norms, weight updates ||Delta W||_2, logits).
3. Phase 3: 8-Arm Causal Weight Ablation Benchmark (TRAINED, FRESH_RANDOM, RESET, SHUFFLED, UNRELATED, DISABLED, SHUFFLED_TARGETS, FROZEN_BASE).
4. Phase 4 & 5: High-Information Procedural Benchmark across 5 families, evaluating Exact Match, sequence loss, token accuracy on held-out sealed test sets.
5. Phase 6 & 7: Lifelong Procedural Stream across 5 policies (SELECTIVE_NO_EXTRA_PROBES, SELECTIVE_RETAIN, RESET, FRESH_ADAPT, ALWAYS_REUSE) across Seeds 42, 43, 44, with recurring tasks up to 5 intervening episodes apart.
6. Phase 8: Capacity diagnostic (Rank-16 vs Rank-32) and empirical retention safeguards (double SHA-256 hashes, 100% carry arithmetic, 100% logic probes, bit-exact rollback).
7. Aggregates multi-seed report with Student's t paired confidence intervals and saves m74_report.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.neural_weight_learning import (
    CausalWeightAblationBenchmark,
    LifelongProceduralStreamEngine,
    M73ScientificAuditor,
    M74SafeguardsAuditor,
    NeuralPathwayInstrumenter,
    ProceduralRuleFamily,
    ProceduralStreamPolicy,
    ProceduralTaskGenerator,
    Rank16SequenceAdapter,
)
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("m74_benchmark")


def setup_base_model(
    checkpoint_dir: Path,
    seed: int,
    device: torch.device,
) -> tuple[MiniGPT, str, str]:
    """Load or initialize canonical M18 base model with parameter hashes."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    relative_path = Path(f"seed_{seed}/base_pretrain/best.pt")
    full_path = checkpoint_dir / relative_path

    if full_path.exists():
        logger.info("Loading pre-trained base model from %s", full_path)
        raw_ckpt = CheckpointManager.load_raw_checkpoint(full_path)
        cfg_dict = raw_ckpt["model_config"]
        cfg = ModelConfig(**cfg_dict) if isinstance(cfg_dict, dict) else cfg_dict
        model = MiniGPT(cfg)
        state_dict = raw_ckpt.get("model_state_dict", raw_ckpt.get("model_state"))
        model.load_state_dict(state_dict)
    else:
        logger.warning(
            "Checkpoint %s not found. Initializing standard M18 architecture model.", full_path
        )
        cfg = ModelConfig(
            vocab_size=400,
            d_model=128,
            n_layers=4,
            n_heads=4,
            d_ff=512,
            context_length=256,
            bias=True,
        )
        model = MiniGPT(cfg)

    model.to(device)
    model.eval()

    unnamed_hash = M74SafeguardsAuditor.compute_unnamed_param_hash(model)
    named_hash = M74SafeguardsAuditor.compute_named_param_hash(model)
    return model, unnamed_hash, named_hash


def run_m74_benchmark_for_seed(
    seed: int,
    checkpoint_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Execute complete M74 benchmark workflow for a single random seed."""
    logger.info("=== Starting Milestone M74 Benchmark for Seed %d ===", seed)
    model, initial_unnamed_hash, initial_named_hash = setup_base_model(checkpoint_dir, seed, device)

    # 1. Phase 1: M73 Forensic Scientific Audit & Parameter Lineage
    logger.info("--- Phase 1: M73 Forensic Audit & Parameter Lineage ---")
    lineage_audit = M73ScientificAuditor.audit_parameter_lineage()
    results_audit = M73ScientificAuditor.audit_m73_results()

    # 2. Phase 2: Active Neural Pathway Instrumentation
    logger.info("--- Phase 2: Neural Pathway Instrumentation ---")
    gen = ProceduralTaskGenerator(vocab_size=16)
    target_task = gen.generate_task(
        task_id=f"seed_{seed}_caesar_shift",
        rule_family=ProceduralRuleFamily.CAESAR_SHIFT_MOD3,
        seed=seed,
        seq_len=3,
        n_train=48,
        n_dev=16,
        n_qual=16,
        n_test=32,
    )
    unrelated_task = gen.generate_task(
        task_id=f"seed_{seed}_affine_unrelated",
        rule_family=ProceduralRuleFamily.AFFINE_TRANSFORM,
        seed=seed + 999,
        seq_len=3,
        n_train=48,
        n_dev=16,
        n_qual=16,
        n_test=32,
    )

    probe_ad = Rank16SequenceAdapter().to(device)
    probe_ad, pathway_record = NeuralPathwayInstrumenter.train_and_instrument(
        model=model,
        adapter=probe_ad,
        task=target_task,
        adapter_id="instrumented_probe_adapter",
        device=device,
        steps=60,
        lr=3e-3,
    )

    # 3. Phase 3: 8-Arm Causal Weight Ablation Benchmark
    logger.info("--- Phase 3: 8-Arm Causal Weight Ablation Benchmark ---")
    ablation_res = CausalWeightAblationBenchmark.run_benchmark(
        model=model,
        task=target_task,
        unrelated_task=unrelated_task,
        device=device,
        seed=seed,
    )

    # 4. Phase 4 & 5: High-Information Generalization across 5 procedural families
    logger.info("--- Phase 4 & 5: High-Information Procedural Rule Generalization ---")
    families = [
        ProceduralRuleFamily.CAESAR_SHIFT_MOD3,
        ProceduralRuleFamily.CAESAR_SHIFT_MOD5,
        ProceduralRuleFamily.AFFINE_TRANSFORM,
        ProceduralRuleFamily.COMPLEMENT_REFLECT,
        ProceduralRuleFamily.NONLINEAR_SBOX,
    ]
    family_evaluations: dict[str, dict[str, float]] = {}

    for fam in families:
        fam_task = gen.generate_task(
            task_id=f"fam_{fam.value}_seed_{seed}",
            rule_family=fam,
            seed=seed,
            n_train=48,
            n_dev=16,
            n_qual=16,
            n_test=32,
        )
        fam_ad = Rank16SequenceAdapter().to(device)
        fam_ad, _ = NeuralPathwayInstrumenter.train_and_instrument(
            model=model,
            adapter=fam_ad,
            task=fam_task,
            adapter_id=f"ad_{fam.value}",
            device=device,
            steps=60,
            lr=3e-3,
        )

        test_h, test_y = NeuralPathwayInstrumenter.extract_symbol_hidden_states(
            model, fam_task.sealed_test_examples, device
        )
        fam_ad.eval()
        with torch.no_grad():
            preds = torch.argmax(fam_ad(test_h), dim=-1)
            em = float((preds == test_y).all(dim=-1).float().mean().item())
            tok_acc = float((preds == test_y).float().mean().item())

        shuffled_ad = fam_ad.shuffle_weights(seed=seed)
        shuffled_ad.eval()
        with torch.no_grad():
            s_preds = torch.argmax(shuffled_ad(test_h), dim=-1)
            s_em = float((s_preds == test_y).all(dim=-1).float().mean().item())

        family_evaluations[fam.value] = {
            "trained_exact_match": round(em, 4),
            "shuffled_exact_match": round(s_em, 4),
            "exact_match_delta": round(em - s_em, 4),
            "token_accuracy": round(tok_acc, 4),
        }

    # 5. Phase 6 & 7: Lifelong Procedural Stream across 5 Policies
    logger.info("--- Phase 6 & 7: Lifelong Continual Stream across 5 Policies ---")
    stream_policies = [
        ProceduralStreamPolicy.SELECTIVE_NO_EXTRA_PROBES,
        ProceduralStreamPolicy.SELECTIVE_RETAIN,
        ProceduralStreamPolicy.RESET,
        ProceduralStreamPolicy.FRESH_ADAPT,
        ProceduralStreamPolicy.ALWAYS_REUSE,
    ]
    stream_results: dict[str, Any] = {}

    for pol in stream_policies:
        logger.info("Running Lifelong Stream Policy: %s (Seed %d)", pol.value, seed)
        summary = LifelongProceduralStreamEngine.run_stream(
            model=model,
            policy=pol,
            seed=seed,
            device=device,
            num_episodes=10,
        )
        stream_results[pol.value] = asdict(summary)

    # 6. Phase 8: Capacity Diagnostic & Retention Safeguards
    logger.info("--- Phase 8: Retention Safeguards & Capacity Diagnostics ---")
    base_invariant = M74SafeguardsAuditor.verify_base_invariance(
        model, initial_unnamed_hash, initial_named_hash
    )
    carry_acc = M74SafeguardsAuditor.verify_arithmetic_carry_retention()
    logic_acc = M74SafeguardsAuditor.verify_chained_operational_logic()
    rollback_ok = M74SafeguardsAuditor.verify_bit_exact_rollback(probe_ad)
    capacity_diagnostic = M74SafeguardsAuditor.compare_capacity_variants(
        d_model=128, num_symbols=16
    )

    return {
        "seed": seed,
        "phase1_lineage": lineage_audit,
        "phase1_audit": results_audit,
        "phase2_pathway": asdict(pathway_record),
        "phase3_ablation": ablation_res,
        "phase4_5_generalization": family_evaluations,
        "phase6_7_stream_policies": stream_results,
        "phase8_safeguards": {
            "base_invariance_verified": base_invariant,
            "arithmetic_carry_retention": carry_acc,
            "chained_operational_logic": logic_acc,
            "bit_exact_rollback_verified": rollback_ok,
            "capacity_diagnostic": capacity_diagnostic,
        },
    }


def aggregate_m74_report(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate multi-seed findings into final M74 report with paired Student's t uncertainty."""
    n_seeds = len(seed_results)
    # Student's t critical value for df = n_seeds - 1
    t_crit = 4.303 if n_seeds == 3 else (2.776 if n_seeds == 5 else 1.96)

    policies = [
        "SELECTIVE_NO_EXTRA_PROBES",
        "SELECTIVE_RETAIN",
        "RESET",
        "FRESH_ADAPT",
        "ALWAYS_REUSE",
    ]

    policy_aggregates: dict[str, Any] = {}
    for pol in policies:
        macro_ems = [
            res["phase6_7_stream_policies"][pol]["macro_exact_match"] for res in seed_results
        ]
        repeat_ems = [
            res["phase6_7_stream_policies"][pol]["repeated_zero_shot_exact_match"]
            for res in seed_results
        ]
        utils = [
            res["phase6_7_stream_policies"][pol]["physical_compute_utility"] for res in seed_results
        ]
        steps = [
            res["phase6_7_stream_policies"][pol]["total_training_steps"] for res in seed_results
        ]
        queries = [
            res["phase6_7_stream_policies"][pol]["total_probe_queries"] for res in seed_results
        ]
        params = [
            res["phase6_7_stream_policies"][pol]["final_stored_parameters"] for res in seed_results
        ]

        mean_em = float(np.mean(macro_ems))
        std_em = float(np.std(macro_ems, ddof=1)) if n_seeds > 1 else 0.0
        sem_em = std_em / np.sqrt(n_seeds) if n_seeds > 1 else 0.0

        mean_util = float(np.mean(utils))
        std_util = float(np.std(utils, ddof=1)) if n_seeds > 1 else 0.0

        policy_aggregates[pol] = {
            "mean_macro_exact_match": round(mean_em, 4),
            "std_macro_exact_match": round(std_em, 4),
            "ci95_macro_exact_match": [
                round(mean_em - t_crit * sem_em, 4),
                round(mean_em + t_crit * sem_em, 4),
            ],
            "mean_repeated_zero_shot_exact_match": round(float(np.mean(repeat_ems)), 4),
            "mean_physical_compute_utility": round(mean_util, 4),
            "std_physical_compute_utility": round(std_util, 4),
            "mean_training_steps": round(float(np.mean(steps)), 1),
            "mean_probe_queries": round(float(np.mean(queries)), 1),
            "mean_stored_parameters": round(float(np.mean(params)), 1),
        }

    # Weight ablation summary
    ablation_arms = [r["arm_name"] for r in seed_results[0]["phase3_ablation"]["results"]]
    ablation_summary: dict[str, Any] = {}
    for arm in ablation_arms:
        arm_ems = []
        arm_losses = []
        for res in seed_results:
            for r in res["phase3_ablation"]["results"]:
                if r["arm_name"] == arm:
                    arm_ems.append(r["exact_match"])
                    arm_losses.append(r["sequence_loss"])
        ablation_summary[arm] = {
            "mean_exact_match": round(float(np.mean(arm_ems)), 4),
            "std_exact_match": round(float(np.std(arm_ems, ddof=1)), 4) if n_seeds > 1 else 0.0,
            "mean_sequence_loss": round(float(np.mean(arm_losses)), 4),
        }

    # Generalization summary across families
    families = list(seed_results[0]["phase4_5_generalization"].keys())
    gen_summary: dict[str, Any] = {}
    for fam in families:
        t_ems = [res["phase4_5_generalization"][fam]["trained_exact_match"] for res in seed_results]
        s_ems = [
            res["phase4_5_generalization"][fam]["shuffled_exact_match"] for res in seed_results
        ]
        gen_summary[fam] = {
            "mean_trained_exact_match": round(float(np.mean(t_ems)), 4),
            "mean_shuffled_exact_match": round(float(np.mean(s_ems)), 4),
            "mean_delta": round(float(np.mean(t_ems)) - float(np.mean(s_ems)), 4),
        }

    # Check key hypotheses:
    # 1. Weight Attribution Verified: TRAINED beats SHUFFLED by >= 0.80 EM
    trained_em = ablation_summary["A_TRAINED_ADAPTER"]["mean_exact_match"]
    shuffled_em = ablation_summary["D_SHUFFLED_TRAINED_WEIGHTS"]["mean_exact_match"]
    weight_attribution_pass = trained_em - shuffled_em >= 0.80

    # 2. SELECTIVE_NO_EXTRA_PROBES vs RESET & FRESH_ADAPT in Utility
    no_probe_util = policy_aggregates["SELECTIVE_NO_EXTRA_PROBES"]["mean_physical_compute_utility"]
    reset_util = policy_aggregates["RESET"]["mean_physical_compute_utility"]
    fresh_util = policy_aggregates["FRESH_ADAPT"]["mean_physical_compute_utility"]
    utility_advantage_vs_reset = no_probe_util - reset_util
    utility_advantage_vs_fresh = no_probe_util - fresh_util

    return {
        "status": "COMPLETED",
        "milestone": "M74",
        "num_seeds": n_seeds,
        "t_critical_95": t_crit,
        "phase1_parameter_lineage": seed_results[0]["phase1_lineage"],
        "phase2_pathway_verified": seed_results[0]["phase2_pathway"]["evaluation_verified"],
        "phase3_causal_ablation_summary": ablation_summary,
        "phase4_5_family_generalization_summary": gen_summary,
        "phase6_7_lifelong_stream_summary": policy_aggregates,
        "scientific_hypotheses": {
            "weight_attribution_pass": weight_attribution_pass,
            "trained_vs_shuffled_delta_em": round(trained_em - shuffled_em, 4),
            "utility_advantage_vs_reset": round(utility_advantage_vs_reset, 4),
            "utility_advantage_vs_fresh": round(utility_advantage_vs_fresh, 4),
        },
        "phase8_safeguards_summary": {
            "base_invariance_all_seeds": all(
                r["phase8_safeguards"]["base_invariance_verified"] for r in seed_results
            ),
            "arithmetic_carry_retention_all_seeds": all(
                r["phase8_safeguards"]["arithmetic_carry_retention"] == 1.0 for r in seed_results
            ),
            "chained_logic_retention_all_seeds": all(
                r["phase8_safeguards"]["chained_operational_logic"] == 1.0 for r in seed_results
            ),
            "bit_exact_rollback_all_seeds": all(
                r["phase8_safeguards"]["bit_exact_rollback_verified"] for r in seed_results
            ),
        },
        "seed_level_results": seed_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Milestone M74 Verified Neural Weight Learning Benchmark"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds for benchmark"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("runs/kaggle/m18-checkpoints"),
        help="Path to base M18 checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/kaggle-m74/output/amg/m74_neural_learning"),
        help="Output directory for reports",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Executing M74 benchmark on device: %s (CUDA available: %s)",
        device,
        torch.cuda.is_available(),
    )

    seed_reports: list[dict[str, Any]] = []
    for seed in args.seeds:
        report = run_m74_benchmark_for_seed(seed, args.checkpoint_dir, device)
        seed_reports.append(report)

    final_report = aggregate_m74_report(seed_reports)
    report_path = args.output_dir / "m74_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    logger.info("=== Milestone M74 Benchmark Complete ===")
    logger.info("Report written to: %s", report_path)
    logger.info(
        "Weight Attribution Pass: %s (Delta EM vs Shuffled: +%.4f)",
        final_report["scientific_hypotheses"]["weight_attribution_pass"],
        final_report["scientific_hypotheses"]["trained_vs_shuffled_delta_em"],
    )
    logger.info(
        "Physical Utility: SELECTIVE_NO_EXTRA_PROBES (%.4f) vs RESET (%.4f) [Delta: %+.4f]",
        final_report["phase6_7_lifelong_stream_summary"]["SELECTIVE_NO_EXTRA_PROBES"][
            "mean_physical_compute_utility"
        ],
        final_report["phase6_7_lifelong_stream_summary"]["RESET"]["mean_physical_compute_utility"],
        final_report["scientific_hypotheses"]["utility_advantage_vs_reset"],
    )
    logger.info(
        "Physical Utility: SELECTIVE_NO_EXTRA_PROBES (%.4f) vs FRESH_ADAPT (%.4f) [Delta: %+.4f]",
        final_report["phase6_7_lifelong_stream_summary"]["SELECTIVE_NO_EXTRA_PROBES"][
            "mean_physical_compute_utility"
        ],
        final_report["phase6_7_lifelong_stream_summary"]["FRESH_ADAPT"][
            "mean_physical_compute_utility"
        ],
        final_report["scientific_hypotheses"]["utility_advantage_vs_fresh"],
    )


if __name__ == "__main__":
    main()
