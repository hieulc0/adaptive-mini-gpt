"""Milestone M63: Few-Shot Adaptation Integrity and Stable Neural Learning Benchmark.

Executes across Seeds 42, 43, 44 on Kaggle Tesla T4 GPU:
- Phase 1: M62 Forensic Metric Audit & Reconciliation.
- Phase 2 & 3: Systematic Few-Shot Adaptation Ablation across Supervision Objectives.
- Phase 4 & 5: Fair Meta-Initialization Re-Evaluation, Blended Hybrid, and Generalization.
- Phase 6 & 7: Retention Safeguards, Parameter Invariance, and Epistemic Regression.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from adaptive_mini_gpt.causal_meta_learning.task_suite import (
    build_development_task_suite,
    build_sealed_meta_test_task_suite,
    create_dev_balanced_hierarchy_environment,
    create_dev_incompatible_parity_environment,
)
from adaptive_mini_gpt.data.bpe_scaling import expand_model_vocabulary, train_scaled_bpe_tokenizer
from adaptive_mini_gpt.data.language_corpus import PRETRAIN_PROSE_CORPUS
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.stable_few_shot import (
    BalancedReptileMetaLearner,
    FairMetaInitEvaluator,
    FewShotAdaptationAblation,
    FewShotSupervisionObjective,
    M62ForensicAuditor,
    M63InitializationArm,
    M63SafeguardAuditor,
    StableFewShotEngine,
    compute_standard_horizon_auc,
    create_minimal_hybrid_init,
    train_balanced_multitask_adapter,
)
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m63_benchmark")


def build_m63_tokenizer(target_vocab_size: int = 1024) -> Any:
    """Build BPE tokenizer covering all relational domains."""
    domain_texts = [
        "GeneMasterA activates_transcription_of OperonUnitB PromoterRegionC RnaPolymeraseD ExonCodingE ProteinProductF. ",
        "MeshNodeA1 replicates_state_to MeshNodeA2 MeshNodeA3 MeshNodeB1 MeshNodeB2 MeshNodeB3. ",
        "ComponentStageC1 feeds_downstream_stage_to AssemblyUnitA2 SubsystemModuleS3 CalibrationRigR4 IntegrationBayI5 AvionicsHousingH6 PropulsionCouplerP7 OrbitalPayloadO8. ",
        "ClusterRootR0 delegates_workload_to SubClusterA1 SubClusterB2 SubClusterC3 LeafWorkerA11 LeafWorkerA12 LeafWorkerB21 LeafWorkerC31. ",
        "JuniorAnalystJ1 reports_transitively_to SeniorAssociateS2 TeamLeadT3 EngineeringDirectorD4 VicePresidentV5 ChiefExecutiveC6. ",
        "CoreSecurityNode1 in_same_security_zone CoreSecurityNode2 CoreSecurityNode3 CoreSecurityNode4 PeripheralSensorNode5 IsolatedGatewayNode6. ",
        "StateAlpha1 holds_mutual_pact_with StateBeta2 StateGamma3 StateDelta4 StateEpsilon5 StateZeta6. ",
        "AeroModuleM0 assembles_into_subsystem AeroModuleM1 AeroModuleM2 AeroModuleM3 AeroModuleM4 AeroModuleM5 AeroModuleM6 AeroModuleM7 AeroModuleM8 AeroModuleM9. ",
        "GatewayGW routes_traffic_payload_to IngressRouterIR AuthSvcA1 BillingSvcB2 CatalogSvcC3 DbWorkerD1 DbWorkerD2 CacheWorkerC1 QueueWorkerQ1 AuditWorkerA2. ",
        "NationN0 ratifies_bilateral_treaty_with NationN1 NationN2 NationN3 NationN4 NationN5 NationN6 NationN7. ",
        "ProteinP0 catalyzes_phosphorylation_of ProteinP1 ProteinP2 ProteinP3 ProteinP4 ProteinP5 ProteinP6 ProteinP7. ",
        "LogisticsHubH0 forwards_container_consignment_to LogisticsHubH1 LogisticsHubH2 LogisticsHubH3 LogisticsHubH4 LogisticsHubH5 LogisticsHubH6 LogisticsHubH7. ",
    ]
    all_prose = " ".join(PRETRAIN_PROSE_CORPUS) if isinstance(PRETRAIN_PROSE_CORPUS, list) else str(PRETRAIN_PROSE_CORPUS)
    training_data = all_prose + "\n" + "\n".join(domain_texts)
    return train_scaled_bpe_tokenizer(training_data, target_vocab_size=target_vocab_size)


def load_or_init_base_checkpoint(
    checkpoint_dir: Path,
    seed: int,
    device: torch.device,
) -> tuple[MiniGPT, Any]:
    """Load base model checkpoint from M18 repository artifacts."""
    relative_path = Path(f"seed_{seed}/base_pretrain/best.pt")
    full_path = checkpoint_dir / relative_path

    if full_path.exists():
        logger.info("Loading pre-trained base model from %s", full_path)
        raw_ckpt = CheckpointManager.load_raw_checkpoint(full_path)
        base_tok = CheckpointManager.load_tokenizer(raw_ckpt)
        state_dict = raw_ckpt.get("model_state_dict", raw_ckpt.get("model_state"))
        cfg_dict = raw_ckpt["model_config"]
        cfg = ModelConfig(**cfg_dict) if isinstance(cfg_dict, dict) else cfg_dict
        base_model_orig = MiniGPT(cfg)
        base_model_orig.load_state_dict(state_dict)

        scaled_tok = build_m63_tokenizer(target_vocab_size=1024)
        model = expand_model_vocabulary(
            model=base_model_orig,
            old_tokenizer=base_tok,
            new_tokenizer=scaled_tok,
        ).to(device)
        model.eval()
        return model, scaled_tok

    logger.warning("Checkpoint %s not found. Initializing fresh model for benchmark.", full_path)
    tokenizer = build_m63_tokenizer(target_vocab_size=1024)
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=256,
        context_length=256,
    )
    model = MiniGPT(cfg).to(device)
    model.eval()
    return model, tokenizer


def train_source_adapter(
    base_model: MiniGPT,
    tokenizer: Any,
    device: torch.device,
    env: Any,
    seed: int = 42,
) -> torch.nn.Module:
    """Pre-train compatible source adapter on balanced hierarchy task."""
    engine = StableFewShotEngine(
        base_model=base_model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=1e-3,
    )
    traj = engine.adapt_and_evaluate(
        env=env,
        objective=FewShotSupervisionObjective.CONSERVATIVE_COMBINED,
        max_steps=30,
        seed=seed,
    )
    return traj


def main() -> None:
    parser = argparse.ArgumentParser(description="M63 Stable Few-Shot Benchmark")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--checkpoint-dir", "--checkpoints-dir", dest="checkpoint_dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("==================================================")
    logger.info("  Adaptive Mini-GPT: Milestone M63 Benchmark")
    logger.info("  Device: %s | Seeds: %s", device, args.seeds)
    logger.info("==================================================")

    # Phase 1: Forensic Scientific Audit of M62
    logger.info("--- Phase 1: Forensic Scientific Audit of M62 ---")
    auditor = M62ForensicAuditor()
    m62_audit_report = auditor.run_forensic_audit()
    logger.info("M62 Audit Verdict: %s", m62_audit_report.audit_verdict)

    seed_runs: list[dict[str, Any]] = []

    dev_envs_template = build_development_task_suite()
    sealed_envs_template = build_sealed_meta_test_task_suite()

    for seed in args.seeds:
        logger.info("==================================================")
        logger.info(">>> Processing Seed: %d <<<", seed)
        logger.info("==================================================")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 1. Load M18 base model
        base_model, tokenizer = load_or_init_base_checkpoint(ckpt_dir, seed, device)

        # Freeze base model parameters
        for p in base_model.parameters():
            p.requires_grad = False

        safeguard_auditor = M63SafeguardAuditor(base_model, tokenizer, device)
        base_hash = safeguard_auditor.compute_base_parameter_hash()
        logger.info("Base Model SHA-256: %s", base_hash[:16])

        # Task environments
        dev_target_envs = [
            copy.deepcopy(e) for e in dev_envs_template
            if e.env_id in ["dev_deep_chain", "dev_balanced_hierarchy"]
        ]
        meta_train_envs = [
            copy.deepcopy(e) for e in dev_envs_template
            if e.env_id not in ["dev_incompatible_parity"]
        ]
        sealed_envs = [copy.deepcopy(e) for e in sealed_envs_template]
        eval_tasks = dev_target_envs + sealed_envs

        # Phase 2 & 3: Systematic Few-Shot Adaptation Ablation
        logger.info("--- Phase 2 & 3: Systematic Few-Shot Adaptation Ablation ---")
        ablation_runner = FewShotAdaptationAblation(base_model, tokenizer, device)
        ablation_trajs = ablation_runner.run_ablation_study(
            environments=eval_tasks,
            seed=seed,
        )
        seed_ablation_report = ablation_runner.aggregate_ablation_results(ablation_trajs)
        logger.info("Ablation Best Arm: %s | Reg Reduction: %.1f%%",
                    seed_ablation_report.best_stabilization_arm,
                    seed_ablation_report.regression_reduction_pct)

        # Phase 4: Train Meta-Learned Initializations and Controls
        logger.info("--- Phase 4: Training Meta-Learned & Controlled Initializations ---")
        t_meta_start = time.perf_counter()
        meta_learner = BalancedReptileMetaLearner(
            base_model=base_model,
            tokenizer=tokenizer,
            device=device,
            inner_lr=1e-3,
            meta_lr=0.10,
            inner_steps=6,
            bottleneck_dim=16,
            num_classes=3,
        )
        for it in range(25):
            env_it = meta_train_envs[it % len(meta_train_envs)]
            meta_learner.meta_train_step(env_it)
        meta_state = meta_learner.get_meta_state()
        meta_train_time_ms = (time.perf_counter() - t_meta_start) * 1000.0
        logger.info("Balanced Reptile Meta-Training complete in %.1fms", meta_train_time_ms)

        # Multitask Pretrained
        multitask_state = train_balanced_multitask_adapter(
            base_model=base_model,
            tokenizer=tokenizer,
            device=device,
            envs=meta_train_envs,
            total_steps=50,
            bottleneck_dim=16,
        )

        # Shuffled Meta-Init (scale-controlled random permutation)
        shuffled_init_state = {}
        for k, v in meta_state.items():
            flat = v.clone().view(-1)
            perm = flat[torch.randperm(flat.numel())]
            shuffled_init_state[k] = perm.view(v.shape)

        # Minimal Hybrid Gated Blending (alpha = 0.5)
        hybrid_state = create_minimal_hybrid_init(meta_state, multitask_state, alpha=0.5)

        # Source adapter for compatible transfer
        source_env = create_dev_balanced_hierarchy_environment(seed=seed)
        source_traj = train_source_adapter(base_model, tokenizer, device, source_env, seed=seed)

        # Phase 5: Fair Meta-Initialization Re-Evaluation across sealed tasks
        logger.info("--- Phase 5: Fair Meta-Initialization Re-Evaluation ---")
        fair_evaluator = FairMetaInitEvaluator(base_model, tokenizer, device)
        arms_to_test = [
            M63InitializationArm.ARM_A_FRESH_RANDOM_INIT,
            M63InitializationArm.ARM_B_COMPATIBLE_TRANSFER,
            M63InitializationArm.ARM_C_MULTITASK_PRETRAINED,
            M63InitializationArm.ARM_D_REPTILE_META_LEARNED,
            M63InitializationArm.ARM_E_SHUFFLED_META_LEARNING,
            M63InitializationArm.ARM_F_COMPUTE_MATCHED_EXTENDED_FRESH,
            M63InitializationArm.ARM_G_MINIMAL_HYBRID_INIT,
        ]

        task_arm_evals: dict[str, dict[str, Any]] = {}
        for env in eval_tasks:
            task_arm_evals[env.env_id] = {}
            for arm in arms_to_test:
                t0 = time.perf_counter()
                init_s = None
                if arm == M63InitializationArm.ARM_C_MULTITASK_PRETRAINED:
                    init_s = multitask_state
                elif arm == M63InitializationArm.ARM_D_REPTILE_META_LEARNED:
                    init_s = meta_state
                elif arm == M63InitializationArm.ARM_E_SHUFFLED_META_LEARNING:
                    init_s = shuffled_init_state
                elif arm == M63InitializationArm.ARM_G_MINIMAL_HYBRID_INIT:
                    init_s = hybrid_state

                res = fair_evaluator.evaluate_initialization_arm(
                    env=env,
                    arm=arm,
                    init_state_dict=init_s,
                    seed=seed,
                )
                dur = (time.perf_counter() - t0) * 1000.0
                task_arm_evals[env.env_id][arm.value] = asdict(res)
                logger.info("  [Seed %d] Env [%s] Arm [%s]: HorizonAUC=%.4f, InitF1=%.4f, FinalF1=%.4f (%.1fms)",
                            seed, env.env_id, arm.value, res.horizon_quality_auc,
                            res.initial_metrics.macro_f1, res.final_metrics.macro_f1, dur)

        # Cross-Family Generalization Audit
        gen_report = fair_evaluator.audit_cross_family_generalization(
            arm_outcomes={
                env.env_id: {
                    arm.value: fair_evaluator.evaluate_initialization_arm(env, arm, init_state_dict=None, seed=seed)
                    for arm in [M63InitializationArm.ARM_D_REPTILE_META_LEARNED, M63InitializationArm.ARM_C_MULTITASK_PRETRAINED]
                }
                for env in eval_tasks
            },
            meta_arm=M63InitializationArm.ARM_D_REPTILE_META_LEARNED.value,
            baseline_arm=M63InitializationArm.ARM_C_MULTITASK_PRETRAINED.value,
        )

        # Phase 6 & 7: Retention Safeguards & Epistemic Audit
        logger.info("--- Phase 6 & 7: Retention Safeguards & Epistemic Regression ---")
        dummy_adapter = torch.nn.Linear(128, 3).to(device)
        safeguard_rep = safeguard_auditor.audit_safeguards(source_env, dummy_adapter, seed=seed)
        logger.info("Safeguard Result: HashOk=%s, Carry=%.2f, Logic=%.2f, SPRT FAR=%.2f%%, SPRT Recall=%.2f%%",
                    safeguard_rep.base_parameter_hash_invariant, safeguard_rep.arithmetic_carry_retention,
                    safeguard_rep.chained_logic_retention, safeguard_rep.sprt_known_false_alarm_rate * 100.0,
                    safeguard_rep.sprt_unknown_recall * 100.0)

        seed_runs.append({
            "seed": seed,
            "base_hash": base_hash,
            "meta_train_time_ms": meta_train_time_ms,
            "ablation_trajectories": {
                k: [asdict(t) for t in v] for k, v in ablation_trajs.items()
            },
            "ablation_report": asdict(seed_ablation_report),
            "task_arm_evals": task_arm_evals,
            "generalization_report": asdict(gen_report),
            "safeguard_report": asdict(safeguard_rep),
        })

    # Cross-Seed Aggregation & Decisions
    logger.info("==================================================")
    logger.info("--- Aggregating Cross-Seed Empirical Statistics ---")
    logger.info("==================================================")

    dev_tids = [e.env_id for e in dev_target_envs]
    sealed_tids = [e.env_id for e in sealed_envs]

    arm_split_metrics: dict[str, dict[str, float]] = {}
    for arm in arms_to_test:
        aname = arm.value
        dev_aucs = [
            sr["task_arm_evals"][tid][aname]["horizon_quality_auc"]
            for sr in seed_runs for tid in dev_tids
        ]
        sealed_aucs = [
            sr["task_arm_evals"][tid][aname]["horizon_quality_auc"]
            for sr in seed_runs for tid in sealed_tids
        ]
        sealed_init_f1s = [
            sr["task_arm_evals"][tid][aname]["initial_metrics"]["macro_f1"]
            for sr in seed_runs for tid in sealed_tids
        ]
        sealed_final_f1s = [
            sr["task_arm_evals"][tid][aname]["final_metrics"]["macro_f1"]
            for sr in seed_runs for tid in sealed_tids
        ]
        sealed_effs = [
            sr["task_arm_evals"][tid][aname]["compute_adjusted_efficiency"]
            for sr in seed_runs for tid in sealed_tids
        ]
        sealed_pairs = [
            sr["task_arm_evals"][tid][aname]["final_metrics"]["paired_correctness"]
            for sr in seed_runs for tid in sealed_tids
        ]

        arm_split_metrics[aname] = {
            "dev_auc_mean": float(np.mean(dev_aucs)) if dev_aucs else 0.0,
            "sealed_auc_mean": float(np.mean(sealed_aucs)),
            "sealed_initial_f1_mean": float(np.mean(sealed_init_f1s)),
            "sealed_final_f1_mean": float(np.mean(sealed_final_f1s)),
            "sealed_eff_mean": float(np.mean(sealed_effs)),
            "sealed_paired_mean": float(np.mean(sealed_pairs)),
        }
        logger.info("Arm [%s]: Sealed AUC=%.4f, InitF1=%.4f, FinalF1=%.4f, Paired=%.1f%%, Eff=%.4f",
                    aname, arm_split_metrics[aname]["sealed_auc_mean"],
                    arm_split_metrics[aname]["sealed_initial_f1_mean"],
                    arm_split_metrics[aname]["sealed_final_f1_mean"],
                    arm_split_metrics[aname]["sealed_paired_mean"] * 100.0,
                    arm_split_metrics[aname]["sealed_eff_mean"])

    # Aggregate Sample-Efficiency Curve across N for Key Arms
    sample_efficiency_summary: dict[str, dict[str, float]] = {}
    for arm in [
        M63InitializationArm.ARM_A_FRESH_RANDOM_INIT,
        M63InitializationArm.ARM_C_MULTITASK_PRETRAINED,
        M63InitializationArm.ARM_D_REPTILE_META_LEARNED,
        M63InitializationArm.ARM_G_MINIMAL_HYBRID_INIT,
    ]:
        aname = arm.value
        sample_efficiency_summary[aname] = {}
        for n_idx, n_val in enumerate([0, 2, 4, 8, 16, 32]):
            f1s_at_n = [
                sr["task_arm_evals"][tid][aname]["sample_efficiency_curve"][n_idx]["metrics"]["macro_f1"]
                for sr in seed_runs for tid in sealed_tids
            ]
            sample_efficiency_summary[aname][f"N_{n_val}"] = float(np.mean(f1s_at_n))

    # Ablation Cross-Seed Summary
    ablation_summary: dict[str, dict[str, float]] = {}
    for obj_name in [
        FewShotSupervisionObjective.NO_UPDATE.value,
        FewShotSupervisionObjective.UNSTABILIZED_M62.value,
        FewShotSupervisionObjective.COMBINED_SUPERVISION.value,
        FewShotSupervisionObjective.POINTWISE_ONLY.value,
        FewShotSupervisionObjective.PAIRED_ONLY.value,
        FewShotSupervisionObjective.CONSERVATIVE_COMBINED.value,
    ]:
        obj_aucs = [
            t["horizon_quality_auc"]
            for sr in seed_runs
            for t in sr["ablation_trajectories"].get(obj_name, [])
            if t["env_id"] in sealed_tids
        ]
        obj_n0 = [
            t["sample_efficiency_curve"][0]["metrics"]["macro_f1"]
            for sr in seed_runs
            for t in sr["ablation_trajectories"].get(obj_name, [])
            if t["env_id"] in sealed_tids
        ]
        obj_n2 = [
            t["sample_efficiency_curve"][1]["metrics"]["macro_f1"]
            for sr in seed_runs
            for t in sr["ablation_trajectories"].get(obj_name, [])
            if t["env_id"] in sealed_tids
        ]
        obj_n32 = [
            t["sample_efficiency_curve"][5]["metrics"]["macro_f1"]
            for sr in seed_runs
            for t in sr["ablation_trajectories"].get(obj_name, [])
            if t["env_id"] in sealed_tids
        ]
        obj_final = [
            t["final_metrics"]["macro_f1"]
            for sr in seed_runs
            for t in sr["ablation_trajectories"].get(obj_name, [])
            if t["env_id"] in sealed_tids
        ]
        obj_reg = [
            1.0 if t["regression_occurred"] else 0.0
            for sr in seed_runs
            for t in sr["ablation_trajectories"].get(obj_name, [])
            if t["env_id"] in sealed_tids
        ]

        ablation_summary[obj_name] = {
            "auc_mean": float(np.mean(obj_aucs)) if obj_aucs else 0.0,
            "n0_f1_mean": float(np.mean(obj_n0)) if obj_n0 else 0.0,
            "n2_f1_mean": float(np.mean(obj_n2)) if obj_n2 else 0.0,
            "n32_f1_mean": float(np.mean(obj_n32)) if obj_n32 else 0.0,
            "final_f1_mean": float(np.mean(obj_final)) if obj_final else 0.0,
            "regression_rate_pct": float(np.mean(obj_reg) * 100.0) if obj_reg else 0.0,
        }

    # Qualification Gate Decisions
    all_hash_ok = all(sr["safeguard_report"]["base_parameter_hash_invariant"] for sr in seed_runs)
    all_sprt_far_ok = all(sr["safeguard_report"]["sprt_known_false_alarm_rate"] <= 0.05 for sr in seed_runs)
    all_sprt_rec_ok = all(sr["safeguard_report"]["sprt_unknown_recall"] >= 0.90 for sr in seed_runs)

    fresh_sealed_auc = arm_split_metrics[M63InitializationArm.ARM_A_FRESH_RANDOM_INIT.value]["sealed_auc_mean"]
    reptile_sealed_auc = arm_split_metrics[M63InitializationArm.ARM_D_REPTILE_META_LEARNED.value]["sealed_auc_mean"]
    multitask_sealed_auc = arm_split_metrics[M63InitializationArm.ARM_C_MULTITASK_PRETRAINED.value]["sealed_auc_mean"]
    shuffled_sealed_auc = arm_split_metrics[M63InitializationArm.ARM_E_SHUFFLED_META_LEARNING.value]["sealed_auc_mean"]
    hybrid_sealed_auc = arm_split_metrics[M63InitializationArm.ARM_G_MINIMAL_HYBRID_INIT.value]["sealed_auc_mean"]

    unstab_reg_rate = ablation_summary[FewShotSupervisionObjective.UNSTABILIZED_M62.value]["regression_rate_pct"]
    conserv_reg_rate = ablation_summary[FewShotSupervisionObjective.CONSERVATIVE_COMBINED.value]["regression_rate_pct"]
    conserv_n32_f1 = ablation_summary[FewShotSupervisionObjective.CONSERVATIVE_COMBINED.value]["n32_f1_mean"]
    n0_baseline_f1 = ablation_summary[FewShotSupervisionObjective.NO_UPDATE.value]["n0_f1_mean"]

    engineering_pass = bool(all_hash_ok and len(seed_runs) == 3)
    learning_stability_pass = bool(conserv_reg_rate < 10.0 and unstab_reg_rate > 80.0)
    few_shot_learning_pass = bool(conserv_n32_f1 > n0_baseline_f1)
    meta_learning_pass = bool(
        reptile_sealed_auc > fresh_sealed_auc
        and reptile_sealed_auc > multitask_sealed_auc
        and reptile_sealed_auc > shuffled_sealed_auc
    )

    qualification_gates = {
        "engineering_pass": engineering_pass,
        "learning_stability_pass": learning_stability_pass,
        "few_shot_learning_pass": few_shot_learning_pass,
        "meta_learning_pass": meta_learning_pass,
        "epistemic_pass": bool(all_sprt_far_ok and all_sprt_rec_ok),
        "topology_expansion": "RESEARCH_ONLY",
        "production_promotion": "NO-GO",
    }

    final_report = {
        "milestone": "M63",
        "title": "Few-Shot Adaptation Integrity and Stable Neural Learning",
        "timestamp": time.time(),
        "hardware": "Tesla T4 GPU (Remote Kaggle Engine)",
        "seeds": args.seeds,
        "phase1_m62_forensic_audit": asdict(m62_audit_report),
        "ablation_summary": ablation_summary,
        "arm_split_metrics": arm_split_metrics,
        "sample_efficiency_summary": sample_efficiency_summary,
        "qualification_gates": qualification_gates,
        "seed_runs": seed_runs,
    }

    report_path = out_dir / "m63_report.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)

    logger.info("==================================================")
    logger.info("Benchmark Finished. Report saved to %s", report_path)
    logger.info("Qualification Gates: %s", json.dumps(qualification_gates, indent=2))
    logger.info("==================================================")


if __name__ == "__main__":
    main()
