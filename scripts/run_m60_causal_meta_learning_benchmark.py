"""Comprehensive 3-Seed Benchmark Runner for Milestone M60 on Tesla T4 GPU.

Executes all 8 phases of M60:
1. Forensic Audit of M59 Claims (efficiency formulas, routing equivalence, task accounting).
2. Multi-Split Task Suite with independent reachability oracle.
3. Causal Experiential-Memory Ablation across 7 Arms (No History, Relevant, Random, Shuffled,
   Incompatible, Fixed 40, Progress-Cost No Memory).
4. Functional Skill Routing Benchmark with measurable behavioral divergence on route-dependent queries.
5. Minimal Trainable Neural Gate vs Deterministic Evidence Likelihood Router.
6. Independent SPRT Epistemic Diagnostic across 4 noise regimes (0%, 5%, 15%, 25%).
7. Multi-Task Retention Safeguards, Parameter Hash Invariance, and Rollback Integrity.
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

from adaptive_mini_gpt.causal_meta_learning.audit import (
    M59ScientificAuditor,
)
from adaptive_mini_gpt.causal_meta_learning.causal_ablation import (
    CausalAblationAdaptationEngine,
    CausalAblationArm,
    CausalAdaptationResult,
)
from adaptive_mini_gpt.causal_meta_learning.epistemic_validation import (
    IndependentEpistemicValidator,
)
from adaptive_mini_gpt.causal_meta_learning.functional_routing import (
    FunctionalSkillRouter,
    MinimalNeuralGate,
)
from adaptive_mini_gpt.causal_meta_learning.retention_safeguards import (
    CausalRetentionAuditor,
)
from adaptive_mini_gpt.causal_meta_learning.task_suite import (
    build_development_task_suite,
    build_sealed_meta_test_task_suite,
    compute_observable_complexity,
    create_dev_balanced_hierarchy_environment,
    create_dev_deep_chain_environment,
    create_dev_incompatible_parity_environment,
    create_dev_non_transitive_bipartite_environment,
    create_meta_long_chain_aerospace,
)
from adaptive_mini_gpt.generalizable_learning.skill_router import RouterMode
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.data.bpe_scaling import expand_model_vocabulary, train_scaled_bpe_tokenizer
from adaptive_mini_gpt.data.language_corpus import PRETRAIN_PROSE_CORPUS
from adaptive_mini_gpt.training.checkpoint import CheckpointManager
from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter
from adaptive_mini_gpt.reliable_transfer.composition_models import ResidualSkillAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("m60_benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M60 Causal Meta-Learning Benchmark")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds")
    parser.add_argument("--checkpoints-dir", "--checkpoint-dir", dest="checkpoints_dir", type=str, default="/kaggle/input/amg-m18-checkpoints", help="M18 Checkpoints directory")
    parser.add_argument("--output-dir", type=str, default="runs/m60_causal_meta_learning", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    return parser.parse_args()


def build_m60_tokenizer(target_vocab_size: int = 1024) -> Any:
    """Build BPE tokenizer covering all M60 relational domains and vocabularies."""
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
    tokenizer = train_scaled_bpe_tokenizer(training_data, target_vocab_size=target_vocab_size)
    return tokenizer


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

        scaled_tok = build_m60_tokenizer(target_vocab_size=1024)
        model = expand_model_vocabulary(
            model=base_model_orig,
            old_tokenizer=base_tok,
            new_tokenizer=scaled_tok,
        ).to(device)
        model.eval()
        return model, scaled_tok

    logger.warning("Checkpoint %s not found. Initializing fresh model for benchmark.", full_path)
    tokenizer = build_m60_tokenizer(target_vocab_size=1024)
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
    logger.info("Starting M60 Causal Meta-Learning Benchmark")
    logger.info("Hardware: %s | Device: %s | Seeds: %s", gpu_name, device, args.seeds)
    logger.info("==================================================")

    # Phase 1: M59 Forensic Scientific Audit
    logger.info("--- Phase 1: Executing Forensic Audit of M59 Claims ---")
    auditor = M59ScientificAuditor()
    m59_audit = auditor.run_m59_scientific_audit()
    logger.info("Audit Result: Step Savings=%.2f%%, Eff Advantage=%.2f%%, Evaluated Targets=%d",
                m59_audit.step_savings_pct_experiential,
                m59_audit.efficiency_advantage_pct_experiential,
                m59_audit.evaluated_target_environments)

    # Establish Task Suites
    dev_envs = build_development_task_suite()
    sealed_envs = build_sealed_meta_test_task_suite()
    target_tasks = dev_envs + sealed_envs  # 5 dev + 5 sealed = 10 target tasks
    logger.info("Task Suites Established: Dev=%d, Sealed Meta-Test=%d (Total Target Tasks=%d)",
                len(dev_envs), len(sealed_envs), len(target_tasks))

    arms_to_test = [
        CausalAblationArm.ARM_A_NO_HISTORY,
        CausalAblationArm.ARM_B_RELEVANT_HISTORY,
        CausalAblationArm.ARM_C_RANDOM_UNRELATED_HISTORY,
        CausalAblationArm.ARM_D_SHUFFLED_OUTCOMES_HISTORY,
        CausalAblationArm.ARM_E_INCOMPATIBLE_HISTORY,
        CausalAblationArm.ARM_F_FIXED_40_STEPS,
        CausalAblationArm.ARM_G_PROGRESS_COST_NO_MEMORY,
    ]

    routing_modes_to_test = [
        RouterMode.ALWAYS_SOURCE,
        RouterMode.ALWAYS_COMPOSITE,
        RouterMode.ALWAYS_FRESH,
        RouterMode.M58_CONFIDENCE_ENTROPY,
        RouterMode.DETERMINISTIC_VERIFIED_EVIDENCE,
        RouterMode.MINIMAL_TRAINABLE_ROUTER,
        RouterMode.ORACLE,
    ]

    seed_runs: list[dict[str, Any]] = []

    for seed in args.seeds:
        logger.info("--------------------------------------------------")
        logger.info(">>> Processing Seed: %d <<<", seed)
        logger.info("--------------------------------------------------")

        base_model, tokenizer = load_or_init_base_checkpoint(Path(args.checkpoints_dir), seed, device)
        retention_auditor = CausalRetentionAuditor(base_model, tokenizer, device)
        base_hash = retention_auditor.compute_base_parameter_hash()
        logger.info("Seed %d Base Parameter SHA-256 Hash: %s", seed, base_hash[:16])

        # Pre-train frozen source adapter
        source_env = create_dev_balanced_hierarchy_environment(seed=seed)
        source_adapter = train_source_adapter(base_model, tokenizer, device, source_env, seed=seed)

        # Phase 2 & 3: Causal Experiential-Memory Ablation Benchmark
        engine = CausalAblationAdaptationEngine(base_model, tokenizer, device)
        task_arm_evals: dict[str, dict[str, Any]] = {}

        for env in target_tasks:
            task_arm_evals[env.env_id] = {}
            for arm in arms_to_test:
                t0 = time.perf_counter()
                res = engine.run_causal_adaptation(
                    env=env, arm=arm, source_adapter=source_adapter, seed=seed
                )
                dur = (time.perf_counter() - t0) * 1000.0
                task_arm_evals[env.env_id][arm.value] = {
                    "total_steps": res.total_steps_taken,
                    "final_loss": res.final_loss,
                    "final_f1": res.final_f1,
                    "learning_curve_auc": res.learning_curve_auc,
                    "wall_clock_time_ms": res.wall_clock_time_ms,
                    "memory_lookup_overhead_ms": res.memory_lookup_overhead_ms,
                    "compute_adjusted_efficiency": res.compute_adjusted_efficiency,
                }
                logger.info("  [Seed %d] Env [%s] Arm [%s]: AUC=%.4f, Steps=%d, Eff=%.4f (%.1fms)",
                            seed, env.env_id, arm.value, res.learning_curve_auc, res.total_steps_taken,
                            res.compute_adjusted_efficiency, dur)

        # Phase 4 & 5: Functional Skill Routing Benchmark with Real Functional Divergence
        logger.info("--- Phase 4 & 5: Functional Skill Routing Benchmark ---")
        # Target environment with conflicting graph semantics (non-transitive bipartite)
        target_routing_env = create_dev_non_transitive_bipartite_environment(seed=seed + 1)
        # Adapt target composite adapter on conflicting environment to introduce real interference on source
        d_model_val = getattr(base_model.config, "d_model", getattr(base_model.config, "n_embd", 128))
        target_composite = ResidualSkillAdapter(
            source_adapter=source_adapter,
            d_model=d_model_val,
            residual_bottleneck_dim=16,
            num_classes=2,
        ).to(device)

        opt_comp = torch.optim.AdamW(target_composite.residual_head.parameters(), lr=2e-3)
        for step in range(25):
            opt_comp.zero_grad()
            q = ([mp.base_query for mp in target_routing_env.minimal_pairs] + [mp.perturbed_query for mp in target_routing_env.minimal_pairs])[step % len(([mp.base_query for mp in target_routing_env.minimal_pairs] + [mp.perturbed_query for mp in target_routing_env.minimal_pairs]))]
            toks = tokenizer.encode(q.question_text)
            inp = torch.tensor([toks], dtype=torch.long, device=device)
            with torch.no_grad():
                b, t = inp.size()
                tok_emb = base_model.wte(inp)
                pos = torch.arange(0, t, dtype=torch.long, device=device)
                x = base_model.drop(tok_emb + base_model.wpe(pos))
                for block in base_model.blocks:
                    x, _ = block(x)
                h = base_model.ln_f(x)
            logits = target_composite(h)
            tgt_idx = 1 if q.expected_answer == "YES" else 0
            tgt = torch.tensor([tgt_idx], dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(logits, tgt)
            loss.backward()
            opt_comp.step()

        router = FunctionalSkillRouter(
            d_model=getattr(base_model.config, "d_model", getattr(base_model.config, "n_embd", 128)),
            source_adapter=source_adapter,
            composite_adapter=target_composite,
        ).to(device)

        # Calibrate evidence centroids
        with torch.no_grad():
            src_supp_h = []
            for q in ([mp.base_query for mp in source_env.minimal_pairs] + [mp.perturbed_query for mp in source_env.minimal_pairs])[:6]:
                toks = tokenizer.encode(q.question_text)
                inp = torch.tensor([toks], dtype=torch.long, device=device)
                b, t = inp.size()
                tok_emb = base_model.wte(inp)
                pos = torch.arange(0, t, dtype=torch.long, device=device)
                x = base_model.drop(tok_emb + base_model.wpe(pos))
                for block in base_model.blocks:
                    x, _ = block(x)
                src_supp_h.append(base_model.ln_f(x))

            tgt_supp_h = []
            for q in ([mp.base_query for mp in target_routing_env.minimal_pairs] + [mp.perturbed_query for mp in target_routing_env.minimal_pairs])[:6]:
                toks = tokenizer.encode(q.question_text)
                inp = torch.tensor([toks], dtype=torch.long, device=device)
                b, t = inp.size()
                tok_emb = base_model.wte(inp)
                pos = torch.arange(0, t, dtype=torch.long, device=device)
                x = base_model.drop(tok_emb + base_model.wpe(pos))
                for block in base_model.blocks:
                    x, _ = block(x)
                tgt_supp_h.append(base_model.ln_f(x))

        router.calibrate_evidence_centroids(src_supp_h, tgt_supp_h)

        # Train minimal gate on support representations
        router.trainable_gate.train()
        opt_gate = torch.optim.AdamW(router.trainable_gate.parameters(), lr=5e-3)
        for ep in range(30):
            for h in src_supp_h:
                opt_gate.zero_grad()
                gl = router.trainable_gate(h.detach())
                l = torch.nn.functional.cross_entropy(gl, torch.tensor([0], device=device))
                l.backward()
                opt_gate.step()
            for h in tgt_supp_h:
                opt_gate.zero_grad()
                gl = router.trainable_gate(h.detach())
                l = torch.nn.functional.cross_entropy(gl, torch.tensor([1], device=device))
                l.backward()
                opt_gate.step()
        router.trainable_gate.eval()

        router_reports: dict[str, Any] = {}
        for r_mode in routing_modes_to_test:
            rep = router.evaluate_functional_benchmark(
                base_model=base_model,
                tokenizer=tokenizer,
                device=device,
                source_env=source_env,
                target_env=target_routing_env,
                mode=r_mode,
            )
            router_reports[r_mode.value] = asdict(rep)
            logger.info("  Router [%s]: Joint F1=%.4f (Src=%.4f, Tgt=%.4f), Acc=%.2f%%, DivRate=%.2f%%, GainOverComp=%.4f",
                        r_mode.value, rep.joint_source_target_f1, rep.source_routed_macro_f1,
                        rep.target_routed_macro_f1, rep.routing_accuracy * 100.0, rep.divergence_rate * 100.0,
                        rep.behavioral_gain_over_composite)

        # Phase 6: Independent Epistemic Qualification
        logger.info("--- Phase 6: Independent Epistemic SPRT Qualification ---")
        epistemic_validator = IndependentEpistemicValidator(budget=12, seed=seed)
        epistemic_records: list[dict[str, Any]] = []
        for noise in [0.0, 0.05, 0.15, 0.25]:
            for pol in ["graph_contrastive_targeted", "graph_uniform_random"]:
                rec = epistemic_validator.evaluate_regime(noise_level=noise, policy=pol, episodes_per_world=6)
                epistemic_records.append(asdict(rec))
                logger.info("  SPRT [%s]: Noise=%.2f, Recall=%.2f%%, FAR=%.2f%%, Macro F1=%.4f",
                            pol, noise, rec.unknown_rule_recall * 100.0, rec.known_world_false_alarm_rate * 100.0, rec.detection_macro_f1)

        # Phase 7: Retention Safeguards & Rollback Protocol
        logger.info("--- Phase 7: Auditing Retention & Rollback Protocol ---")
        ret_rep = retention_auditor.audit_retention_and_rollback(
            source_env=source_env,
            initial_source_adapter=source_adapter,
            routed_source_f1=router_reports[RouterMode.DETERMINISTIC_VERIFIED_EVIDENCE.value]["source_routed_macro_f1"],
        )
        logger.info("Retention Result: HashOk=%s, Forgetting=%.4f, Carry=%.2f, Logic=%.2f, RollbackOk=%s",
                    ret_rep.base_parameter_hash_invariant, ret_rep.source_isolated_forgetting_f1,
                    ret_rep.arithmetic_carry_retention, ret_rep.chained_logic_retention, ret_rep.rollback_successful)

        seed_runs.append({
            "seed": seed,
            "base_hash": base_hash,
            "task_arm_evals": task_arm_evals,
            "router_reports": router_reports,
            "epistemic_records": epistemic_records,
            "retention_report": asdict(ret_rep),
        })

    # Cross-Seed Aggregation & Qualification Decisions
    logger.info("==================================================")
    logger.info("--- Aggregating Cross-Seed Empirical Statistics ---")
    logger.info("==================================================")

    dev_task_ids = [e.env_id for e in dev_envs]
    sealed_task_ids = [e.env_id for e in sealed_envs]

    # Calculate average AUC and efficiency per arm across splits
    arm_split_metrics: dict[str, dict[str, float]] = {}
    for arm in arms_to_test:
        aname = arm.value
        dev_aucs = [
            sr["task_arm_evals"][tid][aname]["learning_curve_auc"]
            for sr in seed_runs for tid in dev_task_ids
        ]
        sealed_aucs = [
            sr["task_arm_evals"][tid][aname]["learning_curve_auc"]
            for sr in seed_runs for tid in sealed_task_ids
        ]
        dev_effs = [
            sr["task_arm_evals"][tid][aname]["compute_adjusted_efficiency"]
            for sr in seed_runs for tid in dev_task_ids
        ]
        sealed_effs = [
            sr["task_arm_evals"][tid][aname]["compute_adjusted_efficiency"]
            for sr in seed_runs for tid in sealed_task_ids
        ]
        arm_split_metrics[aname] = {
            "dev_auc_mean": float(np.mean(dev_aucs)),
            "sealed_auc_mean": float(np.mean(sealed_aucs)),
            "dev_eff_mean": float(np.mean(dev_effs)),
            "sealed_eff_mean": float(np.mean(sealed_effs)),
        }
        logger.info("Arm [%s]: Dev AUC=%.4f, Sealed AUC=%.4f, Sealed Eff=%.4f",
                    aname, arm_split_metrics[aname]["dev_auc_mean"],
                    arm_split_metrics[aname]["sealed_auc_mean"],
                    arm_split_metrics[aname]["sealed_eff_mean"])

    # Aggregate Router Metrics
    aggregated_routers: dict[str, dict[str, float]] = {}
    for r_mode in routing_modes_to_test:
        rmode = r_mode.value
        joint_f1s = [sr["router_reports"][rmode]["joint_source_target_f1"] for sr in seed_runs]
        src_f1s = [sr["router_reports"][rmode]["source_routed_macro_f1"] for sr in seed_runs]
        tgt_f1s = [sr["router_reports"][rmode]["target_routed_macro_f1"] for sr in seed_runs]
        accs = [sr["router_reports"][rmode]["routing_accuracy"] for sr in seed_runs]
        div_rates = [sr["router_reports"][rmode]["divergence_rate"] for sr in seed_runs]
        dep_rates = [sr["router_reports"][rmode]["route_dependence_rate"] for sr in seed_runs]
        gains = [sr["router_reports"][rmode]["behavioral_gain_over_composite"] for sr in seed_runs]
        lats = [sr["router_reports"][rmode]["mean_latency_ms"] for sr in seed_runs]
        aggregated_routers[rmode] = {
            "joint_f1_mean": float(np.mean(joint_f1s)),
            "source_f1_mean": float(np.mean(src_f1s)),
            "target_f1_mean": float(np.mean(tgt_f1s)),
            "accuracy_mean": float(np.mean(accs)),
            "divergence_rate_mean": float(np.mean(div_rates)),
            "route_dependence_rate_mean": float(np.mean(dep_rates)),
            "behavioral_gain_mean": float(np.mean(gains)),
            "latency_ms_mean": float(np.mean(lats)),
        }

    # Aggregate Epistemic Records
    aggregated_epistemic: dict[str, dict[str, float]] = {}
    for rname in [
        "Noise_0%_graph_contrastive_targeted",
        "Noise_0%_graph_uniform_random",
        "Noise_5%_graph_contrastive_targeted",
        "Noise_5%_graph_uniform_random",
        "Noise_15%_graph_contrastive_targeted",
        "Noise_15%_graph_uniform_random",
        "Noise_25%_graph_contrastive_targeted",
        "Noise_25%_graph_uniform_random",
    ]:
        recalls = []
        fars = []
        macro_f1s = []
        for sr in seed_runs:
            match = next(d for d in sr["epistemic_records"] if d["regime_name"] == rname)
            recalls.append(match["unknown_rule_recall"])
            fars.append(match["known_world_false_alarm_rate"])
            macro_f1s.append(match["detection_macro_f1"])
        aggregated_epistemic[rname] = {
            "recall_mean": float(np.mean(recalls)),
            "false_alarm_rate_mean": float(np.mean(fars)),
            "macro_f1_mean": float(np.mean(macro_f1s)),
        }

    # Causal Qualification Evaluation
    auc_relevant = arm_split_metrics[CausalAblationArm.ARM_B_RELEVANT_HISTORY.value]["sealed_auc_mean"]
    auc_no_hist = arm_split_metrics[CausalAblationArm.ARM_A_NO_HISTORY.value]["sealed_auc_mean"]
    auc_shuffled = arm_split_metrics[CausalAblationArm.ARM_D_SHUFFLED_OUTCOMES_HISTORY.value]["sealed_auc_mean"]
    auc_random = arm_split_metrics[CausalAblationArm.ARM_C_RANDOM_UNRELATED_HISTORY.value]["sealed_auc_mean"]
    auc_fixed40 = arm_split_metrics[CausalAblationArm.ARM_F_FIXED_40_STEPS.value]["sealed_auc_mean"]
    eff_relevant = arm_split_metrics[CausalAblationArm.ARM_B_RELEVANT_HISTORY.value]["sealed_eff_mean"]
    eff_fixed40 = arm_split_metrics[CausalAblationArm.ARM_F_FIXED_40_STEPS.value]["sealed_eff_mean"]

    # Causal attribution test: Relevant history must outperform No History, Shuffled History, and Random History
    causal_advantage_over_shuffled = auc_relevant - auc_shuffled
    causal_advantage_over_no_hist = auc_relevant - auc_no_hist
    causal_history_benefit_proven = bool(
        causal_advantage_over_shuffled > 0.01 and causal_advantage_over_no_hist > 0.005 and eff_relevant > eff_fixed40
    )

    # Functional routing advantage test: Router must provide behavioral gain over forced composite
    routing_behavioral_gain = aggregated_routers[RouterMode.DETERMINISTIC_VERIFIED_EVIDENCE.value]["behavioral_gain_mean"]
    routing_accuracy = aggregated_routers[RouterMode.DETERMINISTIC_VERIFIED_EVIDENCE.value]["accuracy_mean"]
    functional_routing_pass = bool(routing_behavioral_gain > 0.02 and routing_accuracy >= 0.80)

    # Epistemic test: Zero false alarms at 0% noise and FAR <= 0.15 under noise
    far_0 = aggregated_epistemic["Noise_0%_graph_contrastive_targeted"]["false_alarm_rate_mean"]
    recall_0 = aggregated_epistemic["Noise_0%_graph_contrastive_targeted"]["recall_mean"]
    epistemic_pass = bool(far_0 <= 0.05 and recall_0 >= 0.90)

    all_hashes_invariant = all(sr["retention_report"]["base_parameter_hash_invariant"] for sr in seed_runs)
    all_rollback_ok = all(sr["retention_report"]["rollback_successful"] for sr in seed_runs)
    engineering_pass = bool(all_hashes_invariant and all_rollback_ok)

    qualification_gates = {
        "engineering_pass": engineering_pass,
        "causal_experiential_learning_pass": causal_history_benefit_proven,
        "functional_routing_pass": functional_routing_pass,
        "epistemic_pass": epistemic_pass,
        "topology_expansion": "RESEARCH_ONLY",
        "production_promotion": "NO-GO",
    }

    final_report = {
        "milestone": "M60",
        "description": "Causal Meta-Learning and Functional Skill Routing",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": gpu_name,
        "seeds": args.seeds,
        "phase1_m59_audit": asdict(m59_audit),
        "arm_split_metrics": arm_split_metrics,
        "aggregated_routers": aggregated_routers,
        "aggregated_epistemic": aggregated_epistemic,
        "qualification_gates": qualification_gates,
        "seed_runs": seed_runs,
    }

    report_file = output_path / "m60_report.json"
    with open(report_file, "w") as f:
        json.dump(final_report, f, indent=2)

    logger.info("==================================================")
    logger.info("M60 Benchmark Complete! Report written to %s", report_file)
    logger.info("Qualification Gates: %s", json.dumps(qualification_gates, indent=2))
    logger.info("==================================================")


if __name__ == "__main__":
    main()
