"""Milestone M76: Causal Neural Skill Composition and Reusable Procedural Knowledge Benchmark.

Executes on remote Tesla T4 GPU:
1. Phase 1: M75 Forensic Scientific Audit & Hindsight Oracle Diagnosis.
2. Phase 2: Acquisition, Qualification (EM >= 0.75), and Parameter Freezing of Atomic Primitives
   (COMPLEMENT_REFLECT, CAESAR_SHIFT_MOD3, AFFINE_TRANSFORM, and unrelated CAESAR_SHIFT_MOD5).
3. Phase 3 & 6: Noncommutative Compound Tasks Generation (REFLECT_THEN_SHIFT3, SHIFT3_THEN_REFLECT,
   REFLECT_THEN_AFFINE, AFFINE_THEN_SHIFT3).
4. Phase 4: 9-Arm Causal Evaluation (NEURAL_COMPOSITION, WRONG_ORDER, SINGLE_SKILL_A,
   SINGLE_SKILL_B, SHUFFLED_COMPONENT, UNRELATED_COMPONENT, FRESH_MONOLITHIC, UNCHANGED_BASE,
   DETERMINISTIC_ORACLE).
5. Phase 5: Two-Stage Error Diagnosis & Teacher-Forced Decomposition (Stage 1 defect vs
   Stage 2 defect vs Error Propagation).
6. Phase 7: Efficiency, Step Savings, and Parameter Allocation Accounting.
7. Phase 10 & 11: Multi-Seed Statistical Aggregation with Paired Student's t Intervals (Seeds 42, 43, 44)
   and Empirical Safeguards (Double SHA-256 base invariance, primitive hash integrity, carry/logic probes).
8. Outputs comprehensive m76_report.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.neural_skill_composition import (
    CausalCompositionBenchmark,
    CompoundTaskGenerator,
    CompoundTaskType,
    M75ScientificAuditor,
    M76SafeguardsAuditor,
    PrimitiveSkillManager,
    PrimitiveSkillType,
    QualifiedPrimitiveRecord,
)
from adaptive_mini_gpt.neural_weight_learning.procedural_tasks import ProceduralRuleFamily
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("m76_benchmark")


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

    unnamed_hash = M76SafeguardsAuditor.compute_unnamed_param_hash(model)
    named_hash = M76SafeguardsAuditor.compute_named_param_hash(model)
    return model, unnamed_hash, named_hash


def run_m76_benchmark_for_seed(
    seed: int,
    checkpoint_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Execute complete M76 benchmark workflow for a single random seed."""
    logger.info("=== Starting Milestone M76 Benchmark for Seed %d ===", seed)
    model, initial_unnamed_hash, initial_named_hash = setup_base_model(checkpoint_dir, seed, device)

    # 1. Phase 1: M75 Forensic Scientific Audit
    logger.info("--- Phase 1: M75 Forensic Audit & Hindsight Diagnosis ---")
    audit_results = M75ScientificAuditor.run_full_audit()

    # 2. Phase 2: Acquire and Qualify Atomic Primitives
    logger.info("--- Phase 2: Training & Qualifying Atomic Neural Primitives ---")
    primitives: dict[PrimitiveSkillType, QualifiedPrimitiveRecord] = {}

    for p_type, rule_fam in [
        (PrimitiveSkillType.COMPLEMENT_REFLECT, ProceduralRuleFamily.COMPLEMENT_REFLECT),
        (PrimitiveSkillType.CAESAR_SHIFT_MOD3, ProceduralRuleFamily.CAESAR_SHIFT_MOD3),
        (PrimitiveSkillType.AFFINE_TRANSFORM, ProceduralRuleFamily.AFFINE_TRANSFORM),
    ]:
        rec = PrimitiveSkillManager.train_and_qualify_primitive(
            model=model,
            rule_family=rule_fam,
            seed=seed,
            device=device,
            steps=50,
            lr=3e-3,
        )
        primitives[p_type] = rec
        logger.info(
            "Primitive [%s] qualified: EM = %.4f (Loss = %.4f), Hash = %s",
            p_type.value,
            rec.qualification_exact_match,
            rec.qualification_loss,
            rec.parameter_hash[:12],
        )

    # Train unrelated control primitive (CAESAR_SHIFT_MOD5)
    unrelated_rec = PrimitiveSkillManager.train_and_qualify_primitive(
        model=model,
        rule_family=ProceduralRuleFamily.CAESAR_SHIFT_MOD5,
        seed=seed + 999,
        device=device,
        steps=50,
        lr=3e-3,
    )

    initial_primitive_hashes = {
        rec.skill_id: rec.adapter.compute_parameter_hash() for rec in primitives.values()
    }

    # 3. Phase 3 & 4: Evaluate 9 Causal Arms on Noncommutative Compound Tasks
    logger.info("--- Phase 3 & 4: Evaluating 9 Causal Arms on Compound Tasks ---")
    task_gen = CompoundTaskGenerator(vocab_size=16)
    compound_task_types = [
        CompoundTaskType.REFLECT_THEN_SHIFT3,
        CompoundTaskType.SHIFT3_THEN_REFLECT,
        CompoundTaskType.REFLECT_THEN_AFFINE,
        CompoundTaskType.AFFINE_THEN_SHIFT3,
    ]

    compound_reports: dict[str, Any] = {}
    for c_type in compound_task_types:
        task = task_gen.generate_compound_task(
            task_id=f"compound_{c_type.value}_seed{seed}",
            compound_type=c_type,
            seed=seed * 100 + list(CompoundTaskType).index(c_type),
            n_train=48,
            n_dev=16,
            n_qual=16,
            n_sealed_test=48,
        )

        c_rep = CausalCompositionBenchmark.evaluate_all_arms(
            model=model,
            task=task,
            primitives=primitives,
            unrelated_primitive=unrelated_rec,
            device=device,
            seed=seed,
        )

        # Convert report to dictionary
        c_dict = {
            "task_id": c_rep.task_id,
            "compound_type": c_rep.compound_type,
            "stage1_primitive": c_rep.stage1_primitive,
            "stage2_primitive": c_rep.stage2_primitive,
            "arm_results": {k: asdict(v) for k, v in c_rep.arm_results.items()},
            "composition_diagnostics": {
                "cascaded_exact_match": c_rep.composition_execution_details.cascaded_exact_match,
                "token_accuracy": c_rep.composition_execution_details.token_accuracy,
                "stage1_exact_match": c_rep.composition_execution_details.stage1_exact_match,
                "teacher_forced_exact_match": c_rep.composition_execution_details.teacher_forced_exact_match,
                "error_proportions": c_rep.composition_execution_details.error_proportions,
                "mean_stage1_loss": c_rep.composition_execution_details.mean_stage1_loss,
                "mean_stage2_loss": c_rep.composition_execution_details.mean_stage2_loss,
            },
        }
        compound_reports[c_type.value] = c_dict

        logger.info(
            "Compound [%s] | Composition EM: %.4f | Wrong Order EM: %.4f | Monolithic EM: %.4f | Propagation Error: %.2f%%",
            c_type.value,
            c_dict["arm_results"]["A_NEURAL_COMPOSITION"]["exact_match"],
            c_dict["arm_results"]["B_WRONG_ORDER"]["exact_match"],
            c_dict["arm_results"]["G_FRESH_MONOLITHIC"]["exact_match"],
            c_dict["composition_diagnostics"]["error_proportions"].get("ERROR_PROPAGATION", 0.0) * 100,
        )

    # 4. Phase 10 & 11: Safeguards & Invariance Verification
    logger.info("--- Phase 10 & 11: Safeguards, Base & Primitive Invariance ---")
    base_invariance = M76SafeguardsAuditor.verify_base_invariance(
        model, initial_unnamed_hash, initial_named_hash
    )
    prims_invariant = M76SafeguardsAuditor.verify_primitive_invariance(
        primitives, initial_primitive_hashes
    )
    carry_rate = M76SafeguardsAuditor.verify_arithmetic_carry_retention()
    logic_rate = M76SafeguardsAuditor.verify_chained_operational_logic()
    ad_test = copy.deepcopy(primitives[PrimitiveSkillType.COMPLEMENT_REFLECT].adapter)
    rollback_ok = M76SafeguardsAuditor.verify_bit_exact_rollback(ad_test)
    hw_accounting = M76SafeguardsAuditor.audit_hardware_accounting(model, num_primitives=3)

    return {
        "seed": seed,
        "phase1_audit": audit_results,
        "primitives_qualification": {
            p_type.value: {
                "skill_id": rec.skill_id,
                "exact_match": rec.qualification_exact_match,
                "loss": rec.qualification_loss,
                "hash": rec.parameter_hash,
                "parameters": rec.trainable_parameters,
            }
            for p_type, rec in primitives.items()
        },
        "compound_task_benchmarks": compound_reports,
        "phase11_safeguards": {
            "base_invariance_verified": base_invariance,
            "primitives_frozen_verified": prims_invariant,
            "arithmetic_carry_retention_rate": carry_rate,
            "chained_operational_logic_rate": logic_rate,
            "bit_exact_rollback_verified": rollback_ok,
            "hardware_accounting": hw_accounting,
            "initial_unnamed_hash": initial_unnamed_hash,
            "initial_named_hash": initial_named_hash,
        },
    }


def aggregate_multi_seed_results(
    seed_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute multi-seed cross-task statistics and paired Student's t intervals."""
    n_seeds = len(seed_results)
    t_crit = 4.303  # 95% critical value for df=2

    compound_types = list(seed_results[0]["compound_task_benchmarks"].keys())
    arm_names = list(
        seed_results[0]["compound_task_benchmarks"][compound_types[0]]["arm_results"].keys()
    )

    # 1. Aggregate cross-compound, cross-seed performance per arm
    arm_aggregates: dict[str, Any] = {}
    for arm in arm_names:
        seed_macro_ems = []
        seed_macro_toks = []
        seed_steps = []
        seed_params = []

        for s in seed_results:
            c_ems = [
                s["compound_task_benchmarks"][c]["arm_results"][arm]["exact_match"]
                for c in compound_types
            ]
            c_toks = [
                s["compound_task_benchmarks"][c]["arm_results"][arm]["token_accuracy"]
                for c in compound_types
            ]
            seed_macro_ems.append(float(np.mean(c_ems)))
            seed_macro_toks.append(float(np.mean(c_toks)))
            seed_steps.append(
                s["compound_task_benchmarks"][compound_types[0]]["arm_results"][arm][
                    "training_steps_incurred"
                ]
            )
            seed_params.append(
                s["compound_task_benchmarks"][compound_types[0]]["arm_results"][arm][
                    "additional_parameters_allocated"
                ]
            )

        mean_em = float(np.mean(seed_macro_ems))
        std_em = float(np.std(seed_macro_ems, ddof=1)) if n_seeds > 1 else 0.0

        mean_tok = float(np.mean(seed_macro_toks))
        std_tok = float(np.std(seed_macro_toks, ddof=1)) if n_seeds > 1 else 0.0

        arm_aggregates[arm] = {
            "mean_exact_match": round(mean_em, 4),
            "std_exact_match": round(std_em, 4),
            "mean_token_accuracy": round(mean_tok, 4),
            "std_token_accuracy": round(std_tok, 4),
            "training_steps_incurred": int(np.mean(seed_steps)),
            "additional_parameters_allocated": int(np.mean(seed_params)),
        }

    # 2. Paired Student's t comparisons: Arm A (NEURAL_COMPOSITION) vs Controls
    paired_comparisons: dict[str, Any] = {}
    for control_arm in [
        "B_WRONG_ORDER",
        "C_SINGLE_SKILL_A",
        "D_SINGLE_SKILL_B",
        "E_SHUFFLED_COMPONENT",
        "F_UNRELATED_COMPONENT",
        "G_FRESH_MONOLITHIC",
        "H_UNCHANGED_BASE",
    ]:
        deltas = []
        for s in seed_results:
            comp_em = float(
                np.mean([
                    s["compound_task_benchmarks"][c]["arm_results"]["A_NEURAL_COMPOSITION"][
                        "exact_match"
                    ]
                    for c in compound_types
                ])
            )
            ctrl_em = float(
                np.mean([
                    s["compound_task_benchmarks"][c]["arm_results"][control_arm]["exact_match"]
                    for c in compound_types
                ])
            )
            deltas.append(comp_em - ctrl_em)

        mean_d = float(np.mean(deltas))
        std_d = float(np.std(deltas, ddof=1)) if n_seeds > 1 else 0.0
        se_d = std_d / math.sqrt(n_seeds)
        ci_lower = mean_d - t_crit * se_d
        ci_upper = mean_d + t_crit * se_d

        paired_comparisons[f"COMPOSITION_vs_{control_arm}"] = {
            "treatment": "A_NEURAL_COMPOSITION",
            "control": control_arm,
            "mean_delta": round(mean_d, 4),
            "std_delta": round(std_d, 4),
            "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
            "spans_zero": bool(ci_lower <= 0.0 <= ci_upper),
            "statistically_superior": bool(ci_lower > 0.0),
        }

    # 3. Detailed Error Decomposition Breakdown Across Seeds
    aggregated_error_props: dict[str, float] = {}
    err_keys = ["SUCCESS", "STAGE_1_DEFECT", "ERROR_PROPAGATION", "STAGE_2_DEFECT", "DUAL_DEFECT"]
    for k in err_keys:
        vals = [
            s["compound_task_benchmarks"][c]["composition_diagnostics"]["error_proportions"].get(
                k, 0.0
            )
            for s in seed_results
            for c in compound_types
        ]
        aggregated_error_props[k] = round(float(np.mean(vals)), 4)

    # 4. Noncommutativity proof: Compare REFLECT_THEN_SHIFT3 vs SHIFT3_THEN_REFLECT
    noncomm_data: dict[str, Any] = {}
    for s in seed_results:
        em_r_s = s["compound_task_benchmarks"]["REFLECT_THEN_SHIFT3"]["arm_results"][
            "A_NEURAL_COMPOSITION"
        ]["exact_match"]
        em_wrong = s["compound_task_benchmarks"]["REFLECT_THEN_SHIFT3"]["arm_results"][
            "B_WRONG_ORDER"
        ]["exact_match"]
        noncomm_data.setdefault("forward_em", []).append(em_r_s)
        noncomm_data.setdefault("reversed_em", []).append(em_wrong)

    noncommutative_effect = {
        "forward_composition_em": round(float(np.mean(noncomm_data["forward_em"])), 4),
        "reversed_order_em": round(float(np.mean(noncomm_data["reversed_em"])), 4),
        "order_sensitivity_gap": round(
            float(np.mean(noncomm_data["forward_em"]))
            - float(np.mean(noncomm_data["reversed_em"])),
            4,
        ),
    }

    # 5. Safeguards verification across all seeds
    all_base_inv = all(
        s["phase11_safeguards"]["base_invariance_verified"] for s in seed_results
    )
    all_prims_inv = all(
        s["phase11_safeguards"]["primitives_frozen_verified"] for s in seed_results
    )
    all_carry = all(
        s["phase11_safeguards"]["arithmetic_carry_retention_rate"] == 1.0 for s in seed_results
    )
    all_logic = all(
        s["phase11_safeguards"]["chained_operational_logic_rate"] == 1.0 for s in seed_results
    )
    all_rollback = all(
        s["phase11_safeguards"]["bit_exact_rollback_verified"] for s in seed_results
    )

    return {
        "status": "COMPLETED",
        "milestone": "M76",
        "num_seeds": n_seeds,
        "t_critical_95": t_crit,
        "phase1_audit": seed_results[0]["phase1_audit"],
        "arm_aggregates": arm_aggregates,
        "paired_comparisons": paired_comparisons,
        "error_decomposition": aggregated_error_props,
        "noncommutative_analysis": noncommutative_effect,
        "safeguards_summary": {
            "all_seeds_base_invariance": all_base_inv,
            "all_seeds_primitives_frozen": all_prims_inv,
            "all_seeds_carry_retention_100pct": all_carry,
            "all_seeds_logic_probes_100pct": all_logic,
            "all_seeds_rollback_verified": all_rollback,
            "hardware_accounting": seed_results[0]["phase11_safeguards"]["hardware_accounting"],
        },
        "seed_level_results": seed_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M76 Neural Skill Composition Benchmark")
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
        default=Path("runs/kaggle-m76/output/amg/m76_skill_composition"),
        help="Path to save m76_report.json",
    )
    args = parser.parse_args()

    # Preflight runtime check
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for Milestone M76 benchmark. Execution on local CPU is prohibited."
        )

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    logger.info("Executing M76 Benchmark on Remote GPU: %s", gpu_name)

    seed_results = []
    for seed in args.seeds:
        res = run_m76_benchmark_for_seed(seed, args.checkpoint_dir, device)
        seed_results.append(res)

    report = aggregate_multi_seed_results(seed_results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_file = args.output_dir / "m76_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved Milestone M76 Report to %s", report_file)
    logger.info("=== Milestone M76 Benchmark Execution Successfully Finished ===")


if __name__ == "__main__":
    main()
