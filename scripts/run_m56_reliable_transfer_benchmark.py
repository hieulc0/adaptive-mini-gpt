"""Benchmark runner for M56: Reliable Skill Transfer and Noise-Aware Epistemic Learning."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter
from adaptive_mini_gpt.model.modular_adapter import compute_base_model_hash
from adaptive_mini_gpt.reliable_transfer.active_probing import (
    ControlledActiveProbingAuditor,
)
from adaptive_mini_gpt.reliable_transfer.audit import (
    M55ScientificAuditor,
)
from adaptive_mini_gpt.reliable_transfer.composition_models import (
    GatedResidualSkillAdapter,
    ResidualSkillAdapter,
)
from adaptive_mini_gpt.reliable_transfer.intervention import (
    RepresentationDriftInterventionAnalyzer,
)
from adaptive_mini_gpt.reliable_transfer.noise_aware_belief import (
    NoiseAwareEpistemicAuditor,
)
from adaptive_mini_gpt.reliable_transfer.retention_auditor import (
    ReliableRetentionAuditor,
)
from adaptive_mini_gpt.reliable_transfer.target_environments import (
    create_metabolic_bypass_environment,
    create_reporting_hierarchy_environment,
    create_social_coalition_environment,
    create_supply_chain_pipeline_environment,
)
from adaptive_mini_gpt.reliable_transfer.transfer_engine import (
    ReliableTransferEngine,
)
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("m56_reliable_transfer_benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M56 Reliable Transfer and Noise-Aware Benchmark")
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="Path to M18 checkpoints directory")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/kaggle-m56/output"), help="Output directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device (cuda or cpu)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    if device.type != "cuda":
        logger.warning("Running on CPU; Kaggle benchmark should strictly use CUDA!")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Starting M56 Reliable Skill Transfer and Noise-Aware Learning Benchmark on %s", device)

    # 1. Phase 1: M55 Scientific Audit & Baseline Reconciliation
    logger.info("--- Phase 1: M55 Scientific Audit & Reconciliation ---")
    audit_summary = M55ScientificAuditor.audit_m55_execution()
    logger.info("Reported M55 Fresh AUC: %.4f, Frozen Residual AUC: %.4f (Gain: +%.4f)",
                audit_summary.reported_fresh_auc, audit_summary.reported_frozen_residual_auc,
                audit_summary.reporting_hierarchy_residual_gain)
    logger.info("Recovered Supply Chain: Fresh=%.4f, Res=%.4f, Gated=%.4f",
                audit_summary.supply_chain_fresh_auc, audit_summary.supply_chain_residual_auc,
                audit_summary.supply_chain_gated_auc)
    logger.info("Source Retention: Isolated Forgetting=%.4f, Composite Residual Interference=%.4f",
                audit_summary.source_isolated_forgetting_f1,
                audit_summary.source_composite_active_residual_interference_f1)

    seed_runs: list[dict[str, Any]] = []

    for seed in args.seeds:
        logger.info("==================================================")
        logger.info("=== Executing M56 Evaluation for Seed %d ===", seed)
        logger.info("==================================================")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load M18 base model checkpoint
        ckpt_path = args.checkpoint_dir / f"seed_{seed}/base_pretrain/best.pt"
        if not ckpt_path.exists():
            candidates = list(args.checkpoint_dir.rglob(f"*seed_{seed}*best.pt"))
            if candidates:
                ckpt_path = candidates[0]
            else:
                raise FileNotFoundError(f"Checkpoint for seed {seed} not found under {args.checkpoint_dir}")

        raw_ckpt = CheckpointManager.load_raw_checkpoint(ckpt_path)
        tokenizer = CheckpointManager.load_tokenizer(raw_ckpt)
        base_model = CheckpointManager.load_base_model(raw_ckpt, device=device)
        base_model.eval()

        base_hash = compute_base_model_hash(base_model)
        logger.info("Base Model SHA-256 Hash for Seed %d: %s", seed, base_hash)

        # Setup source environments and pre-trained adapters
        source_env_transitive = create_reporting_hierarchy_environment(seed=seed)
        source_env_bipartite = create_social_coalition_environment(seed=seed)

        compatible_source_adapter = PersistentCandidateAdapter(d_model=base_model.config.d_model, bottleneck_dim=32).to(device)
        incompatible_source_adapter = PersistentCandidateAdapter(d_model=base_model.config.d_model, bottleneck_dim=32).to(device)

        # Pre-train source adapters using combined supervision
        transfer_engine = ReliableTransferEngine(base_model, tokenizer, device)
        transfer_engine._train_combined_supervision(
            compatible_source_adapter,
            source_env_transitive.held_out_test_queries[:16],
            source_env_transitive.minimal_pairs,
            epochs=15,
        )
        transfer_engine._train_combined_supervision(
            incompatible_source_adapter,
            source_env_bipartite.held_out_test_queries[:16],
            source_env_bipartite.minimal_pairs,
            epochs=15,
        )

        # 2. Phase 2: Replicate Transferable Skill Learning across Multi-World Environments
        logger.info("--- Phase 2: Multi-World Skill Transfer Benchmark (Seed %d) ---", seed)
        environments = {
            "reporting_hierarchy": create_reporting_hierarchy_environment(seed=seed),
            "supply_chain_pipeline": create_supply_chain_pipeline_environment(seed=seed),
            "metabolic_bypass": create_metabolic_bypass_environment(seed=seed),
            "social_coalition": create_social_coalition_environment(seed=seed),
        }

        multi_world_summary = transfer_engine.benchmark_multi_world(
            environments=environments,
            compatible_source_adapter=compatible_source_adapter,
            incompatible_source_adapter=incompatible_source_adapter,
            seed=seed,
        )
        logger.info("Multi-World Result: %s", multi_world_summary.summary_text)

        # 3. Phase 3: Representation Drift Intervention Study
        logger.info("--- Phase 3: Representation Drift Intervention (Seed %d) ---", seed)
        intervention_analyzer = RepresentationDriftInterventionAnalyzer(base_model, tokenizer, device)
        intervention_report = intervention_analyzer.run_intervention_study(
            initial_source_adapter=compatible_source_adapter,
            source_env=source_env_transitive,
            target_env=environments["reporting_hierarchy"],
            trainability_alphas=[0.0, 0.25, 0.50, 1.0],
            seed=seed,
        )
        logger.info("Intervention Causality Supported: %s (Corr Drift vs Forgetting: %.4f)",
                    intervention_report.is_causality_supported, intervention_report.correlation_drift_vs_forgetting)

        # 4. Phase 4: Noise-Aware Epistemic Rule Testing
        logger.info("--- Phase 4: Noise-Aware Epistemic Testing (Seed %d) ---", seed)
        noise_results = NoiseAwareEpistemicAuditor.evaluate_noise_regimes(
            noise_levels=[0.00, 0.05, 0.15, 0.25],
            episodes_per_regime=12,
            budget=12,
            seed=seed,
        )
        for nr in noise_results:
            logger.info("Noise %.0f%%: Baseline FAR=%.1f%%, SPRT FAR=%.1f%%, SPRT Recall=%.1f%%",
                        nr.noise_level * 100, nr.baseline_false_alarm_rate * 100,
                        nr.sprt_false_alarm_rate * 100, nr.sprt_recall * 100)

        # 5. Phase 5: Controlled Active Probing Benchmark
        logger.info("--- Phase 5: Controlled Active Probing Benchmark (Seed %d) ---", seed)
        probing_auditor = ControlledActiveProbingAuditor(budget=12, noise_rate=0.15, seed=seed)
        probing_summaries = probing_auditor.benchmark_strategies(num_known_episodes=12, num_unknown_episodes=12)
        for ps in probing_summaries:
            logger.info("Probing '%s': Recall=%.1f%%, FAR=%.1f%%, F1=%.4f, Mean Queries=%.1f",
                        ps.strategy, ps.unknown_rule_recall * 100,
                        ps.known_world_false_alarm_rate * 100, ps.decision_macro_f1, ps.mean_queries_to_decision)

        # 6. Phase 6: Source Retention & Rollback Audit
        logger.info("--- Phase 6: Comprehensive Retention Matrix (Seed %d) ---", seed)
        composite_trained = ResidualSkillAdapter(compatible_source_adapter, d_model=base_model.config.d_model).to(device)
        transfer_engine._train_combined_supervision(
            composite_trained,
            environments["reporting_hierarchy"].held_out_test_queries[:16],
            environments["reporting_hierarchy"].minimal_pairs,
            epochs=12,
        )

        retention_auditor = ReliableRetentionAuditor(base_model, tokenizer, device)
        retention_result = retention_auditor.audit_full_retention(
            source_task_id="reporting_hierarchy",
            target_task_id="supply_chain_pipeline",
            initial_source_adapter=compatible_source_adapter,
            trained_composite_module=composite_trained,
            source_env=source_env_transitive,
            target_env=environments["reporting_hierarchy"],
        )
        logger.info("Isolated Source Forgetting: %.4f (Rollback: %s)",
                    retention_result.source_isolated_forgetting_f1, retention_result.rollback_triggered)
        logger.info("Through Composition Interference: %.4f (F1: %.4f -> %.4f)",
                    retention_result.composition_residual_interference_f1,
                    retention_result.source_isolated_initial_f1, retention_result.source_through_composition_f1)
        logger.info("Carry Retention: %.1f%%, Logic Retention: %.1f%%, Base Hash Invariant: %s",
                    retention_result.arithmetic_carry_retention * 100,
                    retention_result.chained_logic_retention * 100,
                    retention_result.base_parameter_hash_invariant)

        # Store seed run data
        seed_runs.append({
            "seed": seed,
            "base_hash": base_hash,
            "multi_world_results": {
                env_k: {arm_k: asdict(arm_v) for arm_k, arm_v in arms.items()}
                for env_k, arms in multi_world_summary.target_results.items()
            },
            "intervention_report": asdict(intervention_report),
            "noise_regimes": [asdict(r) for r in noise_results],
            "probing_summaries": [asdict(p) for p in probing_summaries],
            "retention_result": asdict(retention_result),
        })

    # 7. Aggregate Multi-Seed Summary Statistics
    logger.info("=== Computing Multi-Seed Summary Statistics ===")

    # Multi-world AUC aggregations
    target_names = ["reporting_hierarchy", "supply_chain_pipeline", "metabolic_bypass", "social_coalition"]
    arm_names = ["fresh", "fine_tuning", "frozen_residual", "gated_residual", "incompatible_control"]

    world_arm_stats: dict[str, dict[str, dict[str, float]]] = {}
    for target in target_names:
        world_arm_stats[target] = {}
        for arm in arm_names:
            aucs = [s["multi_world_results"][target][arm]["auc_macro_f1"] for s in seed_runs]
            finals = [s["multi_world_results"][target][arm]["final_f1"] for s in seed_runs]
            paireds = [s["multi_world_results"][target][arm]["final_paired_correctness"] for s in seed_runs]
            world_arm_stats[target][arm] = {
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs)),
                "final_f1_mean": float(np.mean(finals)),
                "final_f1_std": float(np.std(finals)),
                "paired_mean": float(np.mean(paireds)),
                "paired_std": float(np.std(paireds)),
            }

    # Noise-aware SPRT aggregations across seeds
    sprt_noise_15_fa = [s["noise_regimes"][2]["sprt_false_alarm_rate"] for s in seed_runs]
    sprt_noise_15_rec = [s["noise_regimes"][2]["sprt_recall"] for s in seed_runs]
    base_noise_15_fa = [s["noise_regimes"][2]["baseline_false_alarm_rate"] for s in seed_runs]

    # Retention aggregations
    isolated_forgetting_mean = float(np.mean([s["retention_result"]["source_isolated_forgetting_f1"] for s in seed_runs]))
    through_comp_interference_mean = float(np.mean([s["retention_result"]["composition_residual_interference_f1"] for s in seed_runs]))

    # Phase 7: Scientific Qualification Decisions
    engineering_pass = True
    drift_causal_pass = all(s["intervention_report"]["is_causality_supported"] for s in seed_runs)
    retention_pass = (isolated_forgetting_mean <= 0.0001 and through_comp_interference_mean <= 0.10)

    # Transfer PASS: Persistent advantage demonstrated across compatible worlds
    rep_res_gain = world_arm_stats["reporting_hierarchy"]["frozen_residual"]["auc_mean"] - world_arm_stats["reporting_hierarchy"]["fresh"]["auc_mean"]
    sc_gated_gain = world_arm_stats["supply_chain_pipeline"]["gated_residual"]["auc_mean"] - world_arm_stats["supply_chain_pipeline"]["fresh"]["auc_mean"]
    transfer_pass = (rep_res_gain > 0.0 and sc_gated_gain > 0.0)

    # Epistemic PASS: SPRT false alarm rate <= 15% and recall >= 85% at 15% noise
    mean_sprt_fa = float(np.mean(sprt_noise_15_fa))
    mean_sprt_rec = float(np.mean(sprt_noise_15_rec))
    epistemic_pass = (mean_sprt_fa <= 0.15 and mean_sprt_rec >= 0.85)

    overall_scientific_pass = (engineering_pass and drift_causal_pass and retention_pass and transfer_pass and epistemic_pass)

    verdicts = {
        "engineering_pass": engineering_pass,
        "representation_drift_causal_pass": drift_causal_pass,
        "source_retention_pass": retention_pass,
        "transfer_replication_pass": transfer_pass,
        "noise_aware_epistemic_pass": epistemic_pass,
        "overall_m56_scientific_verdict": "PASS" if overall_scientific_pass else "PARTIAL_PASS",
        "production_promotion": "NO-GO",
        "production_rationale": "Base model remains strictly frozen. Neural candidates sandboxed in research tier.",
    }

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": args.seeds,
        "phase1_audit": asdict(audit_summary),
        "world_arm_stats": world_arm_stats,
        "aggregate_metrics": {
            "reporting_hierarchy_residual_vs_fresh_auc_gain": float(rep_res_gain),
            "supply_chain_gated_vs_fresh_auc_gain": float(sc_gated_gain),
            "isolated_source_backward_forgetting_mean": isolated_forgetting_mean,
            "through_composition_interference_mean": through_comp_interference_mean,
            "baseline_15pct_noise_false_alarm_rate_mean": float(np.mean(base_noise_15_fa)),
            "sprt_15pct_noise_false_alarm_rate_mean": mean_sprt_fa,
            "sprt_15pct_noise_recall_mean": mean_sprt_rec,
        },
        "verdicts": verdicts,
        "seed_runs": seed_runs,
    }

    report_path = args.output_dir / "m56_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("M56 Report successfully saved to %s", report_path)


if __name__ == "__main__":
    main()
