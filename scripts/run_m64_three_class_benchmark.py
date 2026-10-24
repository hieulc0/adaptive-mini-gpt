"""Milestone M64: Three-Class Learning and Cross-Family Neural Meta-Generalization Benchmark.

Executes across Seeds 42, 43, 44 on Kaggle Tesla T4 GPU:
- Phase 1 & 2: M63 Forensic Audit & Metric Reconciliation.
- Phase 3: Verified Three-Class Supervision & Budgeted Sampling.
- Phase 4: Narrow vs Broad-Family Reptile Meta-Training Candidates.
- Phase 5 & 6: 8-Arm Controlled Benchmark on Familiar vs Unseen Rule Families.
- Phase 7: Retention Safeguards, Per-Seed Hash Invariance, and Epistemic Regression.
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
    create_dev_dense_cluster_exceptions_environment,
    create_dev_incompatible_parity_environment,
    create_dev_non_transitive_bipartite_environment,
)
from adaptive_mini_gpt.data.bpe_scaling import expand_model_vocabulary, train_scaled_bpe_tokenizer
from adaptive_mini_gpt.data.language_corpus import PRETRAIN_PROSE_CORPUS
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.three_class_meta import (
    CrossFamilyMetaEvaluator,
    M63ForensicAuditor,
    M64InitializationArm,
    M64SafeguardAuditor,
    ThreeClassFewShotSampler,
    ThreeClassReptileLearner,
    ThreeClassSupervisionEngine,
    ThreeClassSupervisionMode,
    compute_standard_horizon_auc,
    create_hybrid_init,
    train_three_class_multitask_adapter,
)
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m64_benchmark")


def build_m64_tokenizer(target_vocab_size: int = 1024) -> Any:
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

        scaled_tok = build_m64_tokenizer(target_vocab_size=1024)
        model = expand_model_vocabulary(
            model=base_model_orig,
            old_tokenizer=base_tok,
            new_tokenizer=scaled_tok,
        ).to(device)
        model.eval()
        return model, scaled_tok

    logger.warning("Checkpoint %s not found. Initializing fresh model for benchmark.", full_path)
    tokenizer = build_m64_tokenizer(target_vocab_size=1024)
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
    engine = ThreeClassSupervisionEngine(
        base_model=base_model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=1e-3,
    )
    traj = engine.adapt_and_evaluate(
        env=env,
        mode=ThreeClassSupervisionMode.M64_THREE_CLASS_TRAINING,
        max_steps=30,
        seed=seed,
    )
    return traj


def main() -> None:
    parser = argparse.ArgumentParser(description="M64 Three-Class & Cross-Family Benchmark")
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
    logger.info("  Adaptive Mini-GPT: Milestone M64 Benchmark")
    logger.info("  Device: %s | Seeds: %s", device, args.seeds)
    logger.info("==================================================")

    # Phase 1 & 2: Forensic Scientific Audit of M63
    logger.info("--- Phase 1 & 2: Forensic Scientific Audit of M63 ---")
    auditor = M63ForensicAuditor()
    m63_audit_report = auditor.run_forensic_audit()
    logger.info("M63 Audit Verdict: %s", m63_audit_report.audit_verdict)

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

        safeguard_auditor = M64SafeguardAuditor(base_model, tokenizer, device, seed=seed)
        base_hash = safeguard_auditor.compute_base_parameter_hash()
        expected_hash = auditor.KNOWN_SEED_HASHES.get(seed, "")
        logger.info("Seed %d Base Model SHA-256: %s (Matches Expected: %s)",
                    seed, base_hash[:16], base_hash == expected_hash)

        # Task distributions
        # Narrow Meta-Training: DAG/hierarchical only
        meta_train_narrow = [
            copy.deepcopy(e) for e in dev_envs_template
            if e.env_id in ["dev_deep_chain", "dev_wide_branching_tree", "dev_balanced_hierarchy"]
        ]
        # Broad Meta-Training: DAGs, exception clusters, bipartite, parity
        meta_train_broad = [
            copy.deepcopy(e) for e in dev_envs_template
        ]

        sealed_envs = [copy.deepcopy(e) for e in sealed_envs_template]
        eval_tasks = sealed_envs

        # Phase 4: Candidate Meta-Training (Narrow vs Broad)
        logger.info("--- Phase 4: Training Narrow vs Broad-Family Reptile Candidates ---")
        t_meta_start = time.perf_counter()

        # Candidate A: Narrow Reptile
        narrow_learner = ThreeClassReptileLearner(
            base_model=base_model,
            tokenizer=tokenizer,
            device=device,
            inner_lr=1e-3,
            meta_lr=0.10,
            inner_steps=6,
            bottleneck_dim=16,
        )
        for it in range(25):
            env_it = meta_train_narrow[it % len(meta_train_narrow)]
            narrow_learner.meta_train_iteration(env_it)
        narrow_meta_state = narrow_learner.get_meta_state()

        # Candidate B: Broad-Family Reptile
        broad_learner = ThreeClassReptileLearner(
            base_model=base_model,
            tokenizer=tokenizer,
            device=device,
            inner_lr=1e-3,
            meta_lr=0.10,
            inner_steps=6,
            bottleneck_dim=16,
        )
        for it in range(25):
            env_it = meta_train_broad[it % len(meta_train_broad)]
            broad_learner.meta_train_iteration(env_it)
        broad_meta_state = broad_learner.get_meta_state()
        meta_train_time_ms = (time.perf_counter() - t_meta_start) * 1000.0

        # Multitask Pretrained 3-Class Adapter
        multitask_state = train_three_class_multitask_adapter(
            base_model=base_model,
            tokenizer=tokenizer,
            device=device,
            envs=meta_train_broad,
            total_steps=50,
            bottleneck_dim=16,
        )

        # Shuffled Broad Meta-Init (scale-controlled parameter permutation)
        shuffled_init_state = {}
        for k, v in broad_meta_state.items():
            flat = v.clone().view(-1)
            perm = flat[torch.randperm(flat.numel())]
            shuffled_init_state[k] = perm.view(v.shape)

        # Fixed 50/50 Hybrid (Broad Meta + Multitask)
        hybrid_state = create_hybrid_init(broad_meta_state, multitask_state, alpha=0.5)

        # Source adapter for compatible transfer
        source_env = create_dev_balanced_hierarchy_environment(seed=seed)
        source_traj = train_source_adapter(base_model, tokenizer, device, source_env, seed=seed)

        # Phase 5 & 6: 8-Arm Controlled Benchmark on Sealed Tasks
        logger.info("--- Phase 5 & 6: 8-Arm Controlled Benchmark on Sealed Tasks ---")
        evaluator = CrossFamilyMetaEvaluator(base_model, tokenizer, device)
        arms_to_test = [
            M64InitializationArm.ARM_A_FRESH_RANDOM,
            M64InitializationArm.ARM_B_MULTITASK_PRETRAINED,
            M64InitializationArm.ARM_C_NARROW_REPTILE,
            M64InitializationArm.ARM_D_BROAD_REPTILE,
            M64InitializationArm.ARM_E_SHUFFLED_CONTROL,
            M64InitializationArm.ARM_F_COMPATIBLE_TRANSFER,
            M64InitializationArm.ARM_G_COMPUTE_MATCHED_EXTENDED_FRESH,
            M64InitializationArm.ARM_H_FIXED_HYBRID,
        ]

        task_arm_evals: dict[str, dict[str, Any]] = {}
        for env in eval_tasks:
            task_arm_evals[env.env_id] = {}
            for arm in arms_to_test:
                t0 = time.perf_counter()
                init_s = None
                if arm == M64InitializationArm.ARM_B_MULTITASK_PRETRAINED:
                    init_s = multitask_state
                elif arm == M64InitializationArm.ARM_C_NARROW_REPTILE:
                    init_s = narrow_meta_state
                elif arm == M64InitializationArm.ARM_D_BROAD_REPTILE:
                    init_s = broad_meta_state
                elif arm == M64InitializationArm.ARM_E_SHUFFLED_CONTROL:
                    init_s = shuffled_init_state
                elif arm == M64InitializationArm.ARM_H_FIXED_HYBRID:
                    init_s = hybrid_state

                res = evaluator.evaluate_initialization_arm(
                    env=env,
                    arm=arm,
                    init_state_dict=init_s,
                    seed=seed,
                )
                dur = (time.perf_counter() - t0) * 1000.0
                task_arm_evals[env.env_id][arm.value] = asdict(res)
                logger.info("  [Seed %d] Env [%s] Arm [%s]: AUC=%.4f, InitF1=%.4f, FinalF1=%.4f, UNK_Rec=%.2f, Paired=%.1f%% (%.1fms)",
                            seed, env.env_id, arm.value, res.horizon_quality_auc,
                            res.initial_metrics.macro_f1, res.final_metrics.macro_f1,
                            res.final_metrics.recall_per_class.get("UNKNOWN", 0.0),
                            res.final_metrics.paired_correctness * 100.0, dur)

        # Cross-Family Generalization Audit (Broad Reptile vs Baselines)
        gen_report = CrossFamilyMetaEvaluator.audit_family_generalization(
            task_arm_evals={
                env.env_id: {
                    arm.value: evaluator.evaluate_initialization_arm(
                        env, arm,
                        init_state_dict=(broad_meta_state if arm == M64InitializationArm.ARM_D_BROAD_REPTILE
                                         else narrow_meta_state if arm == M64InitializationArm.ARM_C_NARROW_REPTILE
                                         else multitask_state if arm == M64InitializationArm.ARM_B_MULTITASK_PRETRAINED
                                         else None),
                        seed=seed,
                    )
                    for arm in [
                        M64InitializationArm.ARM_D_BROAD_REPTILE,
                        M64InitializationArm.ARM_C_NARROW_REPTILE,
                        M64InitializationArm.ARM_B_MULTITASK_PRETRAINED,
                        M64InitializationArm.ARM_A_FRESH_RANDOM,
                    ]
                }
                for env in eval_tasks
            },
            target_meta_arm=M64InitializationArm.ARM_D_BROAD_REPTILE.value,
            baseline_arm=M64InitializationArm.ARM_A_FRESH_RANDOM.value,
        )

        # Phase 7: Retention Safeguards & Epistemic Audit
        logger.info("--- Phase 7: Retention Safeguards & Epistemic Regression ---")
        dummy_adapter = torch.nn.Linear(128, 3).to(device)
        safeguard_rep = safeguard_auditor.audit_safeguards(source_env, dummy_adapter)
        logger.info("Safeguard Result: HashOk=%s, Carry=%.2f, Logic=%.2f, SPRT FAR=%.2f%%, SPRT Recall=%.2f%%",
                    safeguard_rep.base_parameter_hash_invariant, safeguard_rep.arithmetic_carry_retention,
                    safeguard_rep.chained_logic_retention, safeguard_rep.sprt_known_false_alarm_rate * 100.0,
                    safeguard_rep.sprt_unknown_recall * 100.0)

        seed_runs.append({
            "seed": seed,
            "base_hash": base_hash,
            "meta_train_time_ms": meta_train_time_ms,
            "task_arm_evals": task_arm_evals,
            "generalization_report": asdict(gen_report),
            "safeguard_report": asdict(safeguard_rep),
        })

    # Cross-Seed Aggregation & Decisions
    logger.info("==================================================")
    logger.info("--- Aggregating Cross-Seed Empirical Statistics ---")
    logger.info("==================================================")

    sealed_tids = [e.env_id for e in sealed_envs]
    familiar_tids = [tid for tid in sealed_tids if any(k in tid for k in ["chain", "hierarchy", "branching", "tree"])]
    unseen_tids = [tid for tid in sealed_tids if tid not in familiar_tids]

    arm_split_metrics: dict[str, dict[str, float]] = {}
    for arm in arms_to_test:
        aname = arm.value
        all_aucs = [
            sr["task_arm_evals"][tid][aname]["horizon_quality_auc"]
            for sr in seed_runs for tid in sealed_tids
        ]
        familiar_aucs = [
            sr["task_arm_evals"][tid][aname]["horizon_quality_auc"]
            for sr in seed_runs for tid in familiar_tids
        ]
        unseen_aucs = [
            sr["task_arm_evals"][tid][aname]["horizon_quality_auc"]
            for sr in seed_runs for tid in unseen_tids
        ]
        init_f1s = [
            sr["task_arm_evals"][tid][aname]["initial_metrics"]["macro_f1"]
            for sr in seed_runs for tid in sealed_tids
        ]
        final_f1s = [
            sr["task_arm_evals"][tid][aname]["final_metrics"]["macro_f1"]
            for sr in seed_runs for tid in sealed_tids
        ]
        effs = [
            sr["task_arm_evals"][tid][aname]["compute_adjusted_efficiency"]
            for sr in seed_runs for tid in sealed_tids
        ]
        paired = [
            sr["task_arm_evals"][tid][aname]["final_metrics"]["paired_correctness"]
            for sr in seed_runs for tid in sealed_tids
        ]
        unk_recalls = [
            sr["task_arm_evals"][tid][aname]["final_metrics"]["recall_per_class"].get("UNKNOWN", 0.0)
            for sr in seed_runs for tid in sealed_tids
        ]
        reg_flags = [
            1.0 if sr["task_arm_evals"][tid][aname]["regression_occurred"] else 0.0
            for sr in seed_runs for tid in sealed_tids
        ]

        arm_split_metrics[aname] = {
            "sealed_auc_mean": float(np.mean(all_aucs)),
            "familiar_auc_mean": float(np.mean(familiar_aucs)) if familiar_aucs else 0.0,
            "unseen_auc_mean": float(np.mean(unseen_aucs)) if unseen_aucs else 0.0,
            "sealed_initial_f1_mean": float(np.mean(init_f1s)),
            "sealed_final_f1_mean": float(np.mean(final_f1s)),
            "sealed_eff_mean": float(np.mean(effs)),
            "sealed_paired_mean": float(np.mean(paired)),
            "sealed_unknown_recall_mean": float(np.mean(unk_recalls)),
            "regression_rate_pct": float(np.mean(reg_flags) * 100.0),
        }
        logger.info("Arm [%s]: Sealed AUC=%.4f (Fam=%.4f, Unseen=%.4f) | FinalF1=%.4f | UNK_Rec=%.2f | Paired=%.1f%% | Reg=%.1f%%",
                    aname, arm_split_metrics[aname]["sealed_auc_mean"],
                    arm_split_metrics[aname]["familiar_auc_mean"],
                    arm_split_metrics[aname]["unseen_auc_mean"],
                    arm_split_metrics[aname]["sealed_final_f1_mean"],
                    arm_split_metrics[aname]["sealed_unknown_recall_mean"],
                    arm_split_metrics[aname]["sealed_paired_mean"] * 100.0,
                    arm_split_metrics[aname]["regression_rate_pct"])

    # Aggregate Sample-Efficiency Curve across N
    sample_efficiency_summary: dict[str, dict[str, float]] = {}
    for arm in arms_to_test:
        aname = arm.value
        sample_efficiency_summary[aname] = {}
        for n_idx, n_val in enumerate([0, 2, 4, 8, 16, 32]):
            f1s_at_n = [
                sr["task_arm_evals"][tid][aname]["sample_efficiency_curve"][n_idx]["metrics"]["macro_f1"]
                for sr in seed_runs for tid in sealed_tids
            ]
            sample_efficiency_summary[aname][f"N_{n_val}"] = float(np.mean(f1s_at_n))

    # Qualification Gate Decisions
    all_hash_ok = all(sr["safeguard_report"]["base_parameter_hash_invariant"] for sr in seed_runs)
    all_sprt_far_ok = all(sr["safeguard_report"]["sprt_known_false_alarm_rate"] <= 0.05 for sr in seed_runs)
    all_sprt_rec_ok = all(sr["safeguard_report"]["sprt_unknown_recall"] >= 0.90 for sr in seed_runs)

    broad_reptile_auc = arm_split_metrics[M64InitializationArm.ARM_D_BROAD_REPTILE.value]["sealed_auc_mean"]
    narrow_reptile_auc = arm_split_metrics[M64InitializationArm.ARM_C_NARROW_REPTILE.value]["sealed_auc_mean"]
    fresh_auc = arm_split_metrics[M64InitializationArm.ARM_A_FRESH_RANDOM.value]["sealed_auc_mean"]
    multitask_auc = arm_split_metrics[M64InitializationArm.ARM_B_MULTITASK_PRETRAINED.value]["sealed_auc_mean"]
    shuffled_auc = arm_split_metrics[M64InitializationArm.ARM_E_SHUFFLED_CONTROL.value]["sealed_auc_mean"]

    broad_unseen_auc = arm_split_metrics[M64InitializationArm.ARM_D_BROAD_REPTILE.value]["unseen_auc_mean"]
    fresh_unseen_auc = arm_split_metrics[M64InitializationArm.ARM_A_FRESH_RANDOM.value]["unseen_auc_mean"]
    narrow_unseen_auc = arm_split_metrics[M64InitializationArm.ARM_C_NARROW_REPTILE.value]["unseen_auc_mean"]

    broad_unk_recall = arm_split_metrics[M64InitializationArm.ARM_D_BROAD_REPTILE.value]["sealed_unknown_recall_mean"]
    broad_reg_rate = arm_split_metrics[M64InitializationArm.ARM_D_BROAD_REPTILE.value]["regression_rate_pct"]

    engineering_pass = bool(all_hash_ok and len(seed_runs) == 3)
    three_class_learning_pass = bool(broad_unk_recall >= 0.30)
    meta_learning_pass = bool(
        broad_reptile_auc > fresh_auc
        and broad_reptile_auc > multitask_auc
        and broad_reptile_auc > shuffled_auc
    )
    cross_family_pass = bool(broad_unseen_auc > fresh_unseen_auc and broad_unseen_auc > narrow_unseen_auc)
    learning_stability_pass = bool(broad_reg_rate <= 10.0)

    qualification_gates = {
        "engineering_pass": engineering_pass,
        "three_class_learning_pass": three_class_learning_pass,
        "meta_learning_pass": meta_learning_pass,
        "cross_family_pass": cross_family_pass,
        "learning_stability_pass": learning_stability_pass,
        "epistemic_pass": bool(all_sprt_far_ok and all_sprt_rec_ok),
        "topology_expansion": "RESEARCH_ONLY",
        "production_promotion": "NO-GO",
    }

    final_report = {
        "milestone": "M64",
        "title": "Three-Class Learning and Cross-Family Neural Meta-Generalization",
        "timestamp": time.time(),
        "hardware": "Tesla T4 GPU (Remote Kaggle Engine)",
        "seeds": args.seeds,
        "phase1_m63_forensic_audit": asdict(m63_audit_report),
        "arm_split_metrics": arm_split_metrics,
        "sample_efficiency_summary": sample_efficiency_summary,
        "qualification_gates": qualification_gates,
        "seed_runs": seed_runs,
    }

    report_path = out_dir / "m64_report.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)

    logger.info("==================================================")
    logger.info("Benchmark Finished. Report saved to %s", report_path)
    logger.info("Qualification Gates: %s", json.dumps(qualification_gates, indent=2))
    logger.info("==================================================")


if __name__ == "__main__":
    main()
