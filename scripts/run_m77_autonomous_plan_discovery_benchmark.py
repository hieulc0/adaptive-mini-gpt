"""Milestone M77: Autonomous Neural Skill-Plan Discovery from Examples Benchmark.

Executes on remote GPU (Colab / Kaggle Tesla T4):
1. Phase 1: M76 Forensic Scientific Audit & Parameter Lineage Reconciliation.
2. Phase 2: Acquire, Qualify (EM >= 0.75), and Freeze Atomic Primitives.
3. Phase 3: Plan Identifiability & Demonstration Budget Sweep (N in {0, 2, 4, 8}).
4. Phase 4: 9-Arm Causal Plan Discovery & Execution Benchmark.
5. Phase 5 & 6: Reliability & Error Decomposition (Plan Selection vs Execution Defects).
6. Phase 7: Unknown Procedures & Bounded Abstention Benchmark.
7. Phase 9 & 11: Safeguards & Hardware Parameter Accounting.
8. Phase 10: Multi-Seed Statistical Aggregation with Paired Student's t Intervals (df=2, t_crit=4.303).
9. Output comprehensive m77_report.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.neural_plan_discovery import (
    AbstentionEvaluator,
    AutonomousPlanDiscoveryEngine,
    CausalPlanBenchmark,
    M76ScientificAuditor,
    M77SafeguardsAuditor,
    PlanIdentifiabilityGenerator,
    PlanSpace,
)
from adaptive_mini_gpt.neural_skill_composition import (
    CompoundTaskType,
    PrimitiveSkillManager,
    PrimitiveSkillType,
    QualifiedPrimitiveRecord,
)
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("m77_benchmark")


def setup_base_model(
    checkpoint_dir: Path,
    seed: int,
    device: torch.device,
) -> tuple[MiniGPT, str, str]:
    """Load or initialize canonical M18 base model with parameter hashes."""
    checkpoint_path = checkpoint_dir / f"seed_{seed}/base_pretrain/best.pt"
    if checkpoint_path.exists():
        logger.info("Loading pre-trained base model from %s", checkpoint_path)
        raw = CheckpointManager.load_raw_checkpoint(checkpoint_path)
        tokenizer = CheckpointManager.load_tokenizer(raw)
        cfg_dict = raw["model_config"]
        cfg = ModelConfig(
            vocab_size=cfg_dict.get("vocab_size", 256),
            d_model=cfg_dict.get("d_model", 128),
            n_layers=cfg_dict.get("n_layers", 2),
            n_heads=cfg_dict.get("n_heads", 4),
            d_ff=cfg_dict.get("d_ff", 512),
            context_length=cfg_dict.get("context_length", 256),
        )
        model = MiniGPT(cfg)
        model.load_state_dict(raw["model_state_dict"])
        tok_fp = tokenizer.get_fingerprint()
    else:
        logger.warning("Checkpoint not found at %s. Initializing fresh baseline model.", checkpoint_path)
        torch.manual_seed(seed)
        cfg = ModelConfig(
            vocab_size=1024,
            d_model=128,
            n_layers=4,
            n_heads=4,
            d_ff=512,
            context_length=128,
        )
        model = MiniGPT(cfg)
        tok_fp = "synthetic_f698e45e"

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    base_hash = M77SafeguardsAuditor.compute_model_hash(model)
    return model, base_hash, tok_fp


def run_m77_benchmark_for_seed(
    seed: int,
    checkpoint_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Execute full M77 benchmark for a single seed."""
    logger.info("=== Starting Milestone M77 Benchmark for Seed %d ===", seed)
    model, initial_base_hash, tok_fp = setup_base_model(checkpoint_dir, seed, device)

    # 1. Phase 1: M76 Forensic Audit
    logger.info("--- Phase 1: M76 Forensic Audit & Parameter Reconciliation ---")
    audit_report = M76ScientificAuditor.execute_audit()

    # 2. Phase 2: Acquire and Qualify Atomic Primitives
    logger.info("--- Phase 2: Qualifying Atomic Neural Primitives ---")
    primitives: dict[PrimitiveSkillType, QualifiedPrimitiveRecord] = {}
    for p_type, r_fam in [
        (PrimitiveSkillType.COMPLEMENT_REFLECT, PrimitiveSkillType.COMPLEMENT_REFLECT),
        (PrimitiveSkillType.CAESAR_SHIFT_MOD3, PrimitiveSkillType.CAESAR_SHIFT_MOD3),
        (PrimitiveSkillType.AFFINE_TRANSFORM, PrimitiveSkillType.AFFINE_TRANSFORM),
    ]:
        rec = PrimitiveSkillManager.train_and_qualify_primitive(
            model=model,
            rule_family=r_fam,
            seed=seed,
            device=device,
            steps=40,
            lr=3e-3,
        )
        primitives[p_type] = rec
        logger.info(
            "Primitive %s qualified: EM=%.4f, Loss=%.4f, Hash=%s",
            p_type.value,
            rec.qualification_exact_match,
            rec.qualification_loss,
            rec.parameter_hash,
        )

    # 3. Phase 3: Demonstration Budget Sweep (N in {0, 2, 4, 8})
    logger.info("--- Phase 3: Demonstration Budget Sweep (N in {0, 2, 4, 8}) ---")
    generator = PlanIdentifiabilityGenerator(vocab_size=16)
    candidate_plans = PlanSpace.generate_candidate_plans(
        available_primitives=list(primitives.keys()),
        max_depth=2,
        include_abstain=True,
    )

    budget_sweep_results: dict[int, dict[str, float]] = {}
    sweep_tasks = [
        CompoundTaskType.REFLECT_THEN_SHIFT3,
        CompoundTaskType.SHIFT3_THEN_REFLECT,
        CompoundTaskType.REFLECT_THEN_AFFINE,
        CompoundTaskType.AFFINE_THEN_SHIFT3,
    ]

    for n_demo in [0, 2, 4, 8]:
        correct_plans = 0
        total_eval = 0
        abstained_count = 0
        search_latencies = []
        forward_passes = []
        test_exact_matches = []

        for idx, c_type in enumerate(sweep_tasks):
            t = generator.generate_task(
                task_id=f"sweep_c{idx}_n{n_demo}",
                compound_type=c_type,
                seed=seed * 100 + idx,
                n_demo=n_demo,
            )
            s_res = AutonomousPlanDiscoveryEngine.discover_plan(
                model=model,
                task_id=t.task_id,
                demonstrations=t.demonstrations,
                primitives=primitives,
                device=device,
                candidate_plans=candidate_plans,
            )
            total_eval += 1
            if s_res.is_abstained:
                abstained_count += 1
            if s_res.selected_plan.stages == t.ground_truth_plan_stages:
                correct_plans += 1

            search_latencies.append(s_res.search_latency_ms)
            forward_passes.append(s_res.total_forward_passes)

            # Evaluate chosen plan on sealed test set
            em, _, _ = CausalPlanBenchmark.execute_plan_on_test_set(
                model=model,
                plan=s_res.selected_plan,
                test_examples=t.sealed_test_examples,
                primitives=primitives,
                device=device,
            )
            test_exact_matches.append(em)

        budget_sweep_results[n_demo] = {
            "plan_accuracy": round(correct_plans / total_eval, 4),
            "abstention_rate": round(abstained_count / total_eval, 4),
            "mean_search_latency_ms": round(float(np.mean(search_latencies)), 2),
            "mean_forward_passes": round(float(np.mean(forward_passes)), 1),
            "mean_test_exact_match": round(float(np.mean(test_exact_matches)), 4),
        }
        logger.info(
            "Budget N=%d: Plan Acc=%.4f, Abstain Rate=%.4f, Test EM=%.4f",
            n_demo,
            budget_sweep_results[n_demo]["plan_accuracy"],
            budget_sweep_results[n_demo]["abstention_rate"],
            budget_sweep_results[n_demo]["mean_test_exact_match"],
        )

    # 4. Phase 4: 9-Arm Causal Plan Benchmark (at N=4 demonstrations)
    logger.info("--- Phase 4: 9-Arm Causal Plan-Selection Benchmark ---")
    arm_reports: dict[str, dict[str, Any]] = {}
    task_level_results: list[dict[str, Any]] = []

    for idx, c_type in enumerate(sweep_tasks):
        comp_task = generator.generate_task(
            task_id=f"compound_{c_type.value}_seed{seed}",
            compound_type=c_type,
            seed=seed * 1000 + idx,
            n_demo=4,
        )
        arm_res = CausalPlanBenchmark.evaluate_all_arms(
            model=model,
            task=comp_task,
            primitives=primitives,
            device=device,
            seed=seed,
        )
        task_level_results.append({
            "task_id": comp_task.task_id,
            "ground_truth_stages": comp_task.ground_truth_plan_stages,
            "arm_results": {k: asdict(v) for k, v in arm_res.items()},
        })

    # Aggregate arms across the 4 compound tasks
    all_arm_keys = list(task_level_results[0]["arm_results"].keys())
    arm_aggregates: dict[str, dict[str, float]] = {}
    for arm_k in all_arm_keys:
        ems = [t["arm_results"][arm_k]["sealed_exact_match"] for t in task_level_results]
        accs = [t["arm_results"][arm_k]["sealed_token_accuracy"] for t in task_level_results]
        plan_corrs = [1.0 if t["arm_results"][arm_k]["is_plan_correct"] else 0.0 for t in task_level_results]
        latencies = [t["arm_results"][arm_k]["plan_search_latency_ms"] for t in task_level_results]
        fwd_passes = [t["arm_results"][arm_k]["forward_passes_used"] for t in task_level_results]
        arm_aggregates[arm_k] = {
            "mean_exact_match": round(float(np.mean(ems)), 4),
            "std_exact_match": round(float(np.std(ems)), 4),
            "mean_token_accuracy": round(float(np.mean(accs)), 4),
            "std_token_accuracy": round(float(np.std(accs)), 4),
            "plan_accuracy": round(float(np.mean(plan_corrs)), 4),
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "mean_forward_passes": round(float(np.mean(fwd_passes)), 1),
            "training_steps_incurred": task_level_results[0]["arm_results"][arm_k]["training_steps_incurred"],
            "additional_parameters_allocated": task_level_results[0]["arm_results"][arm_k]["additional_parameters_allocated"],
        }
        logger.info(
            "Arm %s: Plan Acc=%.4f, Test EM=%.4f +- %.4f, Token Acc=%.4f",
            arm_k,
            arm_aggregates[arm_k]["plan_accuracy"],
            arm_aggregates[arm_k]["mean_exact_match"],
            arm_aggregates[arm_k]["std_exact_match"],
            arm_aggregates[arm_k]["mean_token_accuracy"],
        )

    # 5. Phase 7: Unknown Procedures & Abstention Benchmark
    logger.info("--- Phase 7: Unknown Procedures & Bounded Abstention Benchmark ---")
    abstention_report = AbstentionEvaluator.evaluate_mixed_stream(
        model=model,
        primitives=primitives,
        device=device,
        seed=seed,
        n_demo=4,
    )
    logger.info(
        "Abstention: Acc=%.4f, TNR (Correct Abstention)=%.4f, TPR (Correct Composition)=%.4f",
        abstention_report.abstention_accuracy,
        abstention_report.true_negative_rate,
        abstention_report.true_positive_rate,
    )

    # 6. Phase 9 & 11: Safeguards & Hardware Parameter Accounting
    logger.info("--- Phase 9 & 11: Safeguards, Core Retention & Hardware Accounting ---")
    safeguards_report = M77SafeguardsAuditor.execute_audit(
        model=model,
        initial_hash=initial_base_hash,
        primitives=primitives,
        device=device,
    )

    return {
        "seed": seed,
        "phase1_audit": asdict(audit_report),
        "phase2_primitives": {
            k.value: {
                "skill_id": v.skill_id,
                "exact_match": v.qualification_exact_match,
                "loss": v.qualification_loss,
                "hash": v.parameter_hash,
                "parameters": v.trainable_parameters,
            }
            for k, v in primitives.items()
        },
        "phase3_budget_sweep": budget_sweep_results,
        "phase4_arm_aggregates": arm_aggregates,
        "phase4_task_results": task_level_results,
        "phase7_abstention": asdict(abstention_report),
        "phase11_safeguards": asdict(safeguards_report),
    }


def aggregate_multiseed_results(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate multi-seed benchmark results and compute paired Student's t intervals."""
    n_seeds = len(seed_results)
    t_crit = 4.303  # df = 2, two-tailed 95%

    arm_names = list(seed_results[0]["phase4_arm_aggregates"].keys())
    arm_aggregates: dict[str, dict[str, Any]] = {}

    for arm in arm_names:
        em_list = [s["phase4_arm_aggregates"][arm]["mean_exact_match"] for s in seed_results]
        acc_list = [s["phase4_arm_aggregates"][arm]["mean_token_accuracy"] for s in seed_results]
        plan_acc_list = [s["phase4_arm_aggregates"][arm]["plan_accuracy"] for s in seed_results]
        lat_list = [s["phase4_arm_aggregates"][arm]["mean_latency_ms"] for s in seed_results]
        fwd_list = [s["phase4_arm_aggregates"][arm]["mean_forward_passes"] for s in seed_results]

        arm_aggregates[arm] = {
            "mean_exact_match": round(float(np.mean(em_list)), 4),
            "std_exact_match": round(float(np.std(em_list, ddof=1)), 4),
            "mean_token_accuracy": round(float(np.mean(acc_list)), 4),
            "std_token_accuracy": round(float(np.std(acc_list, ddof=1)), 4),
            "mean_plan_accuracy": round(float(np.mean(plan_acc_list)), 4),
            "std_plan_accuracy": round(float(np.std(plan_acc_list, ddof=1)), 4),
            "mean_latency_ms": round(float(np.mean(lat_list)), 2),
            "mean_forward_passes": round(float(np.mean(fwd_list)), 1),
            "training_steps_incurred": seed_results[0]["phase4_arm_aggregates"][arm]["training_steps_incurred"],
            "additional_parameters_allocated": seed_results[0]["phase4_arm_aggregates"][arm]["additional_parameters_allocated"],
        }

    # Paired comparisons: Arm A vs all other arms
    primary_arm = "A_AUTONOMOUS_PLAN_SEARCH"
    paired_comparisons: dict[str, dict[str, Any]] = {}

    for other_arm in arm_names:
        if other_arm == primary_arm:
            continue
        deltas = [
            s["phase4_arm_aggregates"][primary_arm]["mean_exact_match"]
            - s["phase4_arm_aggregates"][other_arm]["mean_exact_match"]
            for s in seed_results
        ]
        mean_d = float(np.mean(deltas))
        std_d = float(np.std(deltas, ddof=1)) if n_seeds > 1 else 0.0
        se = std_d / (n_seeds ** 0.5)
        ci_lower = mean_d - t_crit * se
        ci_upper = mean_d + t_crit * se
        spans_zero = bool(ci_lower <= 0.0 <= ci_upper)
        superior = bool(ci_lower > 0.0)

        paired_comparisons[f"PLAN_SEARCH_vs_{other_arm}"] = {
            "treatment": primary_arm,
            "control": other_arm,
            "mean_delta": round(mean_d, 4),
            "std_delta": round(std_d, 4),
            "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
            "spans_zero": spans_zero,
            "statistically_superior": superior,
        }

    # Demonstration budget sweep aggregation
    budgets = [0, 2, 4, 8]
    aggregated_budgets: dict[int, dict[str, float]] = {}
    for b in budgets:
        p_accs = [s["phase3_budget_sweep"][b]["plan_accuracy"] for s in seed_results]
        abstains = [s["phase3_budget_sweep"][b]["abstention_rate"] for s in seed_results]
        lats = [s["phase3_budget_sweep"][b]["mean_search_latency_ms"] for s in seed_results]
        fwds = [s["phase3_budget_sweep"][b]["mean_forward_passes"] for s in seed_results]
        ems = [s["phase3_budget_sweep"][b]["mean_test_exact_match"] for s in seed_results]
        aggregated_budgets[b] = {
            "plan_accuracy": round(float(np.mean(p_accs)), 4),
            "abstention_rate": round(float(np.mean(abstains)), 4),
            "mean_search_latency_ms": round(float(np.mean(lats)), 2),
            "mean_forward_passes": round(float(np.mean(fwds)), 1),
            "mean_test_exact_match": round(float(np.mean(ems)), 4),
        }

    # Abstention aggregation
    abst_accs = [s["phase7_abstention"]["abstention_accuracy"] for s in seed_results]
    tnrs = [s["phase7_abstention"]["true_negative_rate"] for s in seed_results]
    tprs = [s["phase7_abstention"]["true_positive_rate"] for s in seed_results]
    aggregated_abstention = {
        "mean_abstention_accuracy": round(float(np.mean(abst_accs)), 4),
        "mean_true_negative_rate": round(float(np.mean(tnrs)), 4),
        "mean_true_positive_rate": round(float(np.mean(tprs)), 4),
    }

    # Safeguards summary across all seeds
    all_base_inv = all(s["phase11_safeguards"]["base_invariance_verified"] for s in seed_results)
    all_prim_frozen = all(s["phase11_safeguards"]["primitives_frozen_verified"] for s in seed_results)
    all_carry = all(s["phase11_safeguards"]["arithmetic_carry_retention_rate"] == 1.0 for s in seed_results)
    all_logic = all(s["phase11_safeguards"]["chained_operational_logic_rate"] == 1.0 for s in seed_results)
    all_rollback = all(s["phase11_safeguards"]["bit_exact_rollback_verified"] for s in seed_results)

    return {
        "status": "COMPLETED",
        "milestone": "M77",
        "num_seeds": n_seeds,
        "t_critical_95": t_crit,
        "phase1_audit": seed_results[0]["phase1_audit"],
        "demonstration_budget_sweep": aggregated_budgets,
        "arm_aggregates": arm_aggregates,
        "paired_comparisons": paired_comparisons,
        "abstention_summary": aggregated_abstention,
        "safeguards_summary": {
            "all_seeds_base_invariance": all_base_inv,
            "all_seeds_primitives_frozen": all_prim_frozen,
            "all_seeds_carry_retention_100pct": all_carry,
            "all_seeds_logic_probes_100pct": all_logic,
            "all_seeds_rollback_verified": all_rollback,
            "hardware_accounting": seed_results[0]["phase11_safeguards"]["hardware_accounting"],
        },
        "seed_level_results": seed_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M77 Autonomous Plan Discovery Benchmark")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Random seeds for benchmark replication",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("runs/kaggle/m18-checkpoints"),
        help="Path to pre-trained M18 checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/m77_output/amg/m77_plan_discovery"),
        help="Path to save m77_report.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Milestone M77 benchmark. Execution on local CPU is prohibited.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    logger.info("Executing M77 Benchmark on Remote GPU: %s", gpu_name)

    seed_results = []
    for seed in args.seeds:
        res = run_m77_benchmark_for_seed(seed, args.checkpoint_dir, device)
        seed_results.append(res)

    logger.info("=== Aggregating Multi-Seed M77 Benchmark Results across %d seeds ===", len(args.seeds))
    full_report = aggregate_multiseed_results(seed_results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_file = args.output_dir / "m77_report.json"
    report_file.write_text(json.dumps(full_report, indent=2))
    logger.info("Benchmark complete. Multi-seed report saved to %s", report_file)


if __name__ == "__main__":
    main()
