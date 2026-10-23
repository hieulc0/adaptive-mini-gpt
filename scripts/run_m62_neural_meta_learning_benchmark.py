"""Comprehensive 3-Seed Benchmark Runner for Milestone M62 on Tesla T4 GPU.

Executes all 8 phases of M62:
1. Forensic Audit of M61 Measurement Protocol (Horizon Quality AUC, RESET reconciliation).
2. Reptile First-Order Neural Meta-Learning Training on Meta-Train task distribution.
3. Ordinary Multitask Pretraining baseline & Incompatible Meta-Init control.
4. 7-Arm Controlled Initialization Comparison (Fresh, Compatible Transfer, Multitask,
   Reptile Meta-Init, Shuffled Meta-Init, Incompatible Meta-Init, Extended Fresh).
5. Sample-Efficiency Evaluation across N in [0, 2, 4, 8, 16, 32] practice examples.
6. Cross-Family Generalization Audit across Similar, Different, and Incompatible tasks.
7. Retention Safeguards, Parameter Hash Invariance, and Rollback Protocol.
8. Final Scientific Qualification Decisions across Seeds 42, 43, 44.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter
from adaptive_mini_gpt.causal_meta_learning.task_suite import (
    build_development_task_suite,
    build_sealed_meta_test_task_suite,
    create_dev_balanced_hierarchy_environment,
    create_dev_incompatible_parity_environment,
    create_meta_long_chain_aerospace,
)
from adaptive_mini_gpt.data.bpe_scaling import expand_model_vocabulary, train_scaled_bpe_tokenizer
from adaptive_mini_gpt.data.language_corpus import PRETRAIN_PROSE_CORPUS
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.neural_meta_learning import (
    CrossFamilyGeneralizationAuditor,
    M61MeasurementAuditor,
    M62AdaptationEvaluator,
    M62InitializationArm,
    M62SafeguardAuditor,
    ReptileMetaLearner,
    train_multitask_pretrained_adapter,
)
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("m62_benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M62 Neural Meta-Initialization Benchmark")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds")
    parser.add_argument("--checkpoints-dir", "--checkpoint-dir", dest="checkpoints_dir", type=str, default="/kaggle/input/amg-m18-checkpoints", help="M18 Checkpoints directory")
    parser.add_argument("--output-dir", type=str, default="runs/m62_neural_meta_learning", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    return parser.parse_args()


def build_m62_tokenizer(target_vocab_size: int = 1024) -> Any:
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
    training_data = all_prose + chr(10) + chr(10).join(domain_texts)
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

        scaled_tok = build_m62_tokenizer(target_vocab_size=1024)
        model = expand_model_vocabulary(
            model=base_model_orig,
            old_tokenizer=base_tok,
            new_tokenizer=scaled_tok,
        ).to(device)
        model.eval()
        return model, scaled_tok

    logger.warning("Checkpoint %s not found. Initializing fresh model for benchmark.", full_path)
    tokenizer = build_m62_tokenizer(target_vocab_size=1024)
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
    base_model: MiniGPT, tokenizer: Any, device: torch.device, source_env: Any, seed: int
) -> PersistentCandidateAdapter:
    """Pre-train source adapter on corporate governance / balanced hierarchy task."""
    torch.manual_seed(seed)
    d_model = getattr(base_model.config, "d_model", getattr(base_model.config, "n_embd", 128))
    adapter = PersistentCandidateAdapter(
        d_model=d_model, bottleneck_dim=16, num_classes=2
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=2e-3, weight_decay=1e-4)

    def extract_hidden(text: str) -> torch.Tensor:
        toks = tokenizer.encode(text)
        inp = torch.tensor([toks], dtype=torch.long, device=device)
        b, t = inp.size()
        tok_emb = base_model.wte(inp)
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        x = base_model.drop(tok_emb + base_model.wpe(pos))
        for block in base_model.blocks:
            x, _ = block(x)
        return base_model.ln_f(x)

    queries = [mp.base_query for mp in source_env.minimal_pairs] + [mp.perturbed_query for mp in source_env.minimal_pairs]
    adapter.train()
    for step in range(30):
        optimizer.zero_grad()
        q = queries[step % len(queries)]
        h = extract_hidden(q.question_text)
        logits = adapter(h)
        tgt_idx = 1 if q.expected_answer == "YES" else 0
        tgt = torch.tensor([tgt_idx], dtype=torch.long, device=device)
        loss = torch.nn.functional.cross_entropy(logits, tgt)
        loss.backward()
        optimizer.step()

    adapter.eval()
    return adapter


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info("==================================================")
    logger.info("Starting M62 Neural Meta-Initialization Benchmark")
    logger.info("Hardware: %s | Device: %s | Seeds: %s", gpu_name, device, args.seeds)
    logger.info("==================================================")

    # Phase 1: M61 Measurement Protocol Audit
    logger.info("--- Phase 1: Executing Forensic Audit of M61 Protocol ---")
    auditor = M61MeasurementAuditor()
    m61_audit = auditor.run_measurement_audit()
    logger.info("Audit Result: Reset Rate=%.1f%%, Experience Memory Incr Gain=%.4f",
                m61_audit.reset_rate_pct, m61_audit.experience_memory_incremental_gain)

    # Establish Task Suites
    # Meta-Train Suite: 3 training environments
    dev_envs = build_development_task_suite()
    sealed_envs = build_sealed_meta_test_task_suite()
    meta_train_envs = dev_envs[:3]
    dev_target_envs = dev_envs[3:]
    target_tasks = dev_target_envs + sealed_envs  # 2 dev + 5 sealed = 7 target tasks
    logger.info("Task Suites Established: Meta-Train=%d, Dev Targets=%d, Sealed Meta-Test=%d",
                len(meta_train_envs), len(dev_target_envs), len(sealed_envs))

    arms_to_test = [
        M62InitializationArm.ARM_A_FRESH_RANDOM_INIT,
        M62InitializationArm.ARM_B_COMPATIBLE_TRANSFER_ADAPTER,
        M62InitializationArm.ARM_C_MULTITASK_PRETRAINED_INIT,
        M62InitializationArm.ARM_D_REPTILE_META_LEARNED_INIT,
        M62InitializationArm.ARM_E_SHUFFLED_META_LEARNING,
        M62InitializationArm.ARM_F_INCOMPATIBLE_META_INIT,
        M62InitializationArm.ARM_G_COMPUTE_MATCHED_EXTENDED_FRESH,
    ]

    seed_runs: list[dict[str, Any]] = []

    for seed in args.seeds:
        logger.info("--------------------------------------------------")
        logger.info(">>> Processing Seed: %d <<<", seed)
        logger.info("--------------------------------------------------")

        base_model, tokenizer = load_or_init_base_checkpoint(Path(args.checkpoints_dir), seed, device)
        safeguard_auditor = M62SafeguardAuditor(base_model, tokenizer, device)
        base_hash = safeguard_auditor.compute_base_parameter_hash()
        logger.info("Seed %d Base Parameter SHA-256 Hash: %s", seed, base_hash[:16])

        # Pre-train frozen source adapter
        source_env = create_dev_balanced_hierarchy_environment(seed=seed)
        source_adapter = train_source_adapter(base_model, tokenizer, device, source_env, seed=seed)

        # Phase 2: Reptile Neural Meta-Learning Training
        logger.info("--- Phase 2: Training Reptile Meta-Initialization ---")
        t_meta_start = time.perf_counter()
        meta_learner = ReptileMetaLearner(
            base_model=base_model,
            tokenizer=tokenizer,
            device=device,
            inner_lr=3e-3,
            meta_lr=0.15,
            inner_steps=5,
            bottleneck_dim=16,
            num_classes=2,
        )
        meta_records = meta_learner.run_meta_training_loop(meta_train_envs, num_iterations=25)
        meta_init_state = meta_learner.get_meta_initialization_state()
        meta_train_time_ms = (time.perf_counter() - t_meta_start) * 1000.0
        logger.info("Reptile Meta-Training Complete in %.1fms (Iterations=%d)", meta_train_time_ms, len(meta_records))

        # Phase 3: Ordinary Multitask Pretraining & Shuffled/Incompatible Controls
        logger.info("--- Phase 3: Training Controlled Baselines ---")
        multitask_adapter = train_multitask_pretrained_adapter(
            base_model=base_model,
            tokenizer=tokenizer,
            device=device,
            meta_training_envs=meta_train_envs,
            total_steps=50,
        )
        multitask_init_state = {k: v.detach().clone() for k, v in multitask_adapter.state_dict().items()}

        # Shuffled Meta-Init (permuted weights)
        shuffled_init_state = {}
        for k, v in meta_init_state.items():
            flat = v.clone().view(-1)
            perm = flat[torch.randperm(flat.numel())]
            shuffled_init_state[k] = perm.view(v.shape)

        # Incompatible Meta-Init (trained only on alternating parity clash)
        incomp_env = create_dev_incompatible_parity_environment(seed=seed)
        incomp_learner = ReptileMetaLearner(
            base_model=base_model,
            tokenizer=tokenizer,
            device=device,
            inner_lr=3e-3,
            meta_lr=0.15,
            inner_steps=5,
        )
        incomp_learner.run_meta_training_loop([incomp_env], num_iterations=15)
        incompatible_init_state = incomp_learner.get_meta_initialization_state()

        # Phase 4 & 5: Adaptation & Sample-Efficiency Benchmark
        evaluator = M62AdaptationEvaluator(base_model, tokenizer, device)
        task_arm_evals: dict[str, dict[str, Any]] = {}

        for env in target_tasks:
            task_arm_evals[env.env_id] = {}
            for arm in arms_to_test:
                t0 = time.perf_counter()

                # Select initial state based on arm
                init_state = None
                src_ad = None

                if arm == M62InitializationArm.ARM_B_COMPATIBLE_TRANSFER_ADAPTER:
                    src_ad = source_adapter
                elif arm == M62InitializationArm.ARM_C_MULTITASK_PRETRAINED_INIT:
                    init_state = multitask_init_state
                elif arm == M62InitializationArm.ARM_D_REPTILE_META_LEARNED_INIT:
                    init_state = meta_init_state
                elif arm == M62InitializationArm.ARM_E_SHUFFLED_META_LEARNING:
                    init_state = shuffled_init_state
                elif arm == M62InitializationArm.ARM_F_INCOMPATIBLE_META_INIT:
                    init_state = incompatible_init_state

                res = evaluator.evaluate_adaptation_trajectory(
                    env=env,
                    arm=arm,
                    initial_state_dict=init_state,
                    source_adapter=src_ad,
                    seed=seed,
                )
                dur = (time.perf_counter() - t0) * 1000.0
                task_arm_evals[env.env_id][arm.value] = asdict(res)
                logger.info("  [Seed %d] Env [%s] Arm [%s]: HorizonAUC=%.4f, InitF1=%.4f, FinalF1=%.4f, Paired=%.1f%% (%.1fms)",
                            seed, env.env_id, arm.value, res.horizon_quality_auc, res.initial_f1,
                            res.final_f1, res.paired_correctness * 100.0, dur)

        # Phase 6: Cross-Family Generalization Audit
        logger.info("--- Phase 6: Cross-Family Generalization Audit ---")
        gen_auditor = CrossFamilyGeneralizationAuditor()
        gen_report = gen_auditor.audit_family_generalization(
            task_arm_evals=task_arm_evals,
            meta_arm_name=M62InitializationArm.ARM_D_REPTILE_META_LEARNED_INIT.value,
            baseline_arm_name=M62InitializationArm.ARM_C_MULTITASK_PRETRAINED_INIT.value,
        )
        logger.info("Generalization Result: InFamilyAdv=%+.4f, CrossFamilyAdv=%+.4f | %s",
                    gen_report.in_family_transfer_advantage, gen_report.cross_family_transfer_advantage,
                    gen_report.generalization_verdict)

        # Phase 7: Retention Safeguards & Epistemic Regression
        logger.info("--- Phase 7: Retention Safeguards & Epistemic Audit ---")
        safeguard_rep = safeguard_auditor.audit_safeguards(
            source_env=source_env,
            initial_source_adapter=source_adapter,
            seed=seed,
        )
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

    # Cross-Seed Aggregation & Qualification Decisions
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
        dev_effs = [
            sr["task_arm_evals"][tid][aname]["compute_adjusted_efficiency"]
            for sr in seed_runs for tid in dev_tids
        ]
        sealed_effs = [
            sr["task_arm_evals"][tid][aname]["compute_adjusted_efficiency"]
            for sr in seed_runs for tid in sealed_tids
        ]
        sealed_pairs = [
            sr["task_arm_evals"][tid][aname]["paired_correctness"]
            for sr in seed_runs for tid in sealed_tids
        ]
        sealed_init_f1s = [
            sr["task_arm_evals"][tid][aname]["initial_f1"]
            for sr in seed_runs for tid in sealed_tids
        ]
        sealed_final_f1s = [
            sr["task_arm_evals"][tid][aname]["final_f1"]
            for sr in seed_runs for tid in sealed_tids
        ]

        arm_split_metrics[aname] = {
            "dev_auc_mean": float(np.mean(dev_aucs)) if dev_aucs else 0.0,
            "sealed_auc_mean": float(np.mean(sealed_aucs)),
            "dev_eff_mean": float(np.mean(dev_effs)) if dev_effs else 0.0,
            "sealed_eff_mean": float(np.mean(sealed_effs)),
            "sealed_paired_mean": float(np.mean(sealed_pairs)),
            "sealed_initial_f1_mean": float(np.mean(sealed_init_f1s)),
            "sealed_final_f1_mean": float(np.mean(sealed_final_f1s)),
        }
        logger.info("Arm [%s]: Sealed AUC=%.4f, InitF1=%.4f, FinalF1=%.4f, Paired=%.1f%%, Sealed Eff=%.4f",
                    aname, arm_split_metrics[aname]["sealed_auc_mean"],
                    arm_split_metrics[aname]["sealed_initial_f1_mean"],
                    arm_split_metrics[aname]["sealed_final_f1_mean"],
                    arm_split_metrics[aname]["sealed_paired_mean"] * 100.0,
                    arm_split_metrics[aname]["sealed_eff_mean"])

    # Aggregate Sample-Efficiency Curve across N for Meta-Init vs Multitask vs Fresh
    sample_efficiency_summary: dict[str, dict[str, float]] = {}
    for arm in [
        M62InitializationArm.ARM_A_FRESH_RANDOM_INIT,
        M62InitializationArm.ARM_C_MULTITASK_PRETRAINED_INIT,
        M62InitializationArm.ARM_D_REPTILE_META_LEARNED_INIT,
    ]:
        aname = arm.value
        sample_efficiency_summary[aname] = {}
        for n_idx, n_val in enumerate([0, 2, 4, 8, 16, 32]):
            f1s_at_n = [
                sr["task_arm_evals"][tid][aname]["sample_efficiency_curve"][n_idx]["val_f1"]
                for sr in seed_runs for tid in sealed_tids
            ]
            sample_efficiency_summary[aname][f"N_{n_val}"] = float(np.mean(f1s_at_n))

    # Qualification Decision Assessment
    reptile_sealed_auc = arm_split_metrics[M62InitializationArm.ARM_D_REPTILE_META_LEARNED_INIT.value]["sealed_auc_mean"]
    multitask_sealed_auc = arm_split_metrics[M62InitializationArm.ARM_C_MULTITASK_PRETRAINED_INIT.value]["sealed_auc_mean"]
    fresh_sealed_auc = arm_split_metrics[M62InitializationArm.ARM_A_FRESH_RANDOM_INIT.value]["sealed_auc_mean"]
    shuffled_sealed_auc = arm_split_metrics[M62InitializationArm.ARM_E_SHUFFLED_META_LEARNING.value]["sealed_auc_mean"]

    # Meta-Learning PASS: Reptile Meta-Init beats both Fresh Random AND Multitask Pretrained
    meta_learning_pass = bool(
        reptile_sealed_auc > fresh_sealed_auc
        and reptile_sealed_auc > multitask_sealed_auc
        and reptile_sealed_auc > shuffled_sealed_auc
    )

    # Transfer PASS: Improvements persist across multiple independently generated task families
    in_fam_adv = float(np.mean([sr["generalization_report"]["in_family_transfer_advantage"] for sr in seed_runs]))
    cross_fam_adv = float(np.mean([sr["generalization_report"]["cross_family_transfer_advantage"] for sr in seed_runs]))
    transfer_pass = bool(in_fam_adv > 0.0 and cross_fam_adv >= -0.02)

    # Epistemic PASS
    epistemic_pass = all(sr["safeguard_report"]["sprt_known_false_alarm_rate"] <= 0.05 for sr in seed_runs)

    # Engineering PASS
    engineering_pass = all(sr["safeguard_report"]["all_safeguards_passed"] for sr in seed_runs)

    qualification_gates = {
        "engineering_pass": engineering_pass,
        "meta_learning_pass": meta_learning_pass,
        "transfer_pass": transfer_pass,
        "epistemic_pass": epistemic_pass,
        "topology_expansion": "RESEARCH_ONLY",
        "production_promotion": "NO-GO",
    }

    final_report = {
        "milestone": "M62",
        "description": "Neural Meta-Initialization and Verified Learning-to-Learn",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": gpu_name,
        "seeds": args.seeds,
        "phase1_m61_measurement_audit": asdict(m61_audit),
        "arm_split_metrics": arm_split_metrics,
        "sample_efficiency_summary": sample_efficiency_summary,
        "qualification_gates": qualification_gates,
        "seed_runs": seed_runs,
    }

    report_file = output_path / "m62_report.json"
    with open(report_file, "w") as f:
        json.dump(final_report, f, indent=2)

    logger.info("==================================================")
    logger.info("M62 Benchmark Complete! Report written to %s", report_file)
    logger.info("Qualification Gates: %s", json.dumps(qualification_gates, indent=2))
    logger.info("==================================================")


if __name__ == "__main__":
    main()
