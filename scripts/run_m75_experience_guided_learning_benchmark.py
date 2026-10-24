"""Milestone M75: Experience-Guided Autonomous Learning and Adaptive Training Budgets Benchmark.

Executes:
1. Phase 1: M74 Forensic Scientific Audit & Parameter Lineage Verification (928,512 -> 974,658).
2. Phase 2: Verified Learning-Experience Records & Manifest Generation.
3. Phase 3 & 6: Adaptive Budget Optimization & Experience-Guided Controller.
4. Phase 4: 7-Arm Causal Evaluation (NO_HISTORY, RELEVANT_HISTORY, SHUFFLED_HISTORY,
   IRRELEVANT_HISTORY, CONSERVATIVE_HISTORY, FIXED_BUDGET_FRESH, OFFLINE_HINDSIGHT).
5. Phase 5: Procedural Generalization Breakdown (4 tiers: Known Rule, New Parameter,
   Compositional, Unfamiliar Family).
6. Phase 7 & 8: Safeguards (Double SHA-256 base invariance, 20 carry cases, 8 logic probes,
   bit-exact rollback, Rank-16 vs Rank-32 capacity diagnostic, hardware resource accounting).
7. Aggregates multi-seed report across Seeds 42, 43, 44 with paired Student's t confidence
   intervals and saves m75_report.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from adaptive_mini_gpt.experience_guided_learning import (
    CausalExperienceBenchmark,
    ControllerPolicyMode,
    LearningExperienceManifest,
    M74ScientificAuditor,
    M75SafeguardsAuditor,
    ProceduralRuleFamily,
    ProceduralTaskGenerator,
)
from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("m75_benchmark")


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

    unnamed_hash = M75SafeguardsAuditor.compute_unnamed_param_hash(model)
    named_hash = M75SafeguardsAuditor.compute_named_param_hash(model)
    return model, unnamed_hash, named_hash


def build_warmup_experience_history(
    model: MiniGPT,
    device: torch.device,
    seed: int,
) -> LearningExperienceManifest:
    """Generate initial warm-up verified experience records from exploratory runs.

    Provides historical outcomes across budgets 20, 40, 60 on exploratory tasks
    so the experience controller has empirical data for nearest-neighbor inference.
    """
    manifest = LearningExperienceManifest()
    gen = ProceduralTaskGenerator(vocab_size=16)

    # 4 warm-up exploration tasks
    warmup_rules = [
        (ProceduralRuleFamily.CAESAR_SHIFT_MOD3, 20, 0.003),
        (ProceduralRuleFamily.CAESAR_SHIFT_MOD5, 40, 0.003),
        (ProceduralRuleFamily.AFFINE_TRANSFORM, 60, 0.003),
        (ProceduralRuleFamily.COMPLEMENT_REFLECT, 40, 0.003),
    ]

    from adaptive_mini_gpt.experience_guided_learning.experience_records import (
        LearningAction,
        LearningActionType,
        LearningExperienceRecord,
        RetentionDecision,
    )
    from adaptive_mini_gpt.neural_weight_learning.pathway_instrumentation import (
        NeuralPathwayInstrumenter,
    )
    from adaptive_mini_gpt.neural_weight_learning.sequence_adapter import (
        Rank16SequenceAdapter,
    )

    for idx, (rule_fam, budget, lr) in enumerate(warmup_rules):
        task = gen.generate_task(
            task_id=f"warmup_{idx}_{rule_fam.value}",
            rule_family=rule_fam,
            seed=seed * 500 + idx,
            n_train=48,
            n_dev=16,
            n_qual=16,
            n_sealed_test=48,
        )
        desc = task.compute_observable_descriptors()
        adapter = Rank16SequenceAdapter().to(device)

        t0 = time.perf_counter()
        trained_ad, metrics = NeuralPathwayInstrumenter.train_and_instrument(
            model=model,
            adapter=adapter,
            task=task,
            adapter_id=f"warmup_{rule_fam.value}",
            device=device,
            steps=budget,
            lr=lr,
        )
        gpu_time_ms = (time.perf_counter() - t0) * 1000.0
        loss_red = (
            (metrics.initial_loss - metrics.final_loss)
            / max(metrics.initial_loss, 1e-4)
            if metrics.initial_loss > 0
            else 0.0
        )

        qual_h, qual_y = NeuralPathwayInstrumenter.extract_symbol_hidden_states(
            model, task.qualification_examples, device
        )
        trained_ad.eval()
        with torch.no_grad():
            q_preds = torch.argmax(trained_ad(qual_h), dim=-1)
            q_em = float((q_preds == qual_y).all(dim=-1).float().mean().item())

        qual_passed = q_em >= 0.80

        rec = LearningExperienceRecord(
            record_id=f"warmup_rec_{idx}",
            task_id=task.task_id,
            episode_index=idx,
            descriptors=desc,
            action=LearningAction(
                action_type=LearningActionType.TRAIN_CANDIDATE,
                step_budget=budget,
                learning_rate=lr,
            ),
            actual_steps=budget,
            gpu_time_ms=round(gpu_time_ms, 2),
            initial_loss=round(metrics.initial_loss, 4),
            final_loss=round(metrics.final_loss, 4),
            loss_reduction_ratio=round(loss_red, 4),
            dev_em=round(q_em, 4),
            qualification_passed=qual_passed,
            retention_decision=(
                RetentionDecision.RETAINED if qual_passed else RetentionDecision.DISCARDED
            ),
            repertoire_size_after=1 if qual_passed else 0,
        )
        manifest.add_record(rec)

    return manifest


def run_m75_benchmark_for_seed(
    seed: int,
    checkpoint_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Execute complete M75 benchmark workflow for a single random seed."""
    logger.info("=== Starting Milestone M75 Benchmark for Seed %d ===", seed)
    model, initial_unnamed_hash, initial_named_hash = setup_base_model(checkpoint_dir, seed, device)

    # 1. Phase 1: M74 Forensic Scientific Audit
    logger.info("--- Phase 1: M74 Forensic Audit & Parameter Lineage ---")
    audit_results = M74ScientificAuditor.run_full_audit()

    # 2. Build Warm-up History
    logger.info("--- Building Initial Verified Learning Experience History ---")
    initial_history = build_warmup_experience_history(model, device, seed)
    logger.info("Initial history contains %d verified records", initial_history.count())

    # 3. Phase 4: Multi-Arm Causal Evaluation across 7 Policy Regimes
    logger.info("--- Phase 4: Evaluating 7 Causal Policy Regimes on Canonical Task Streams ---")
    policies = [
        ControllerPolicyMode.NO_HISTORY,
        ControllerPolicyMode.RELEVANT_HISTORY,
        ControllerPolicyMode.SHUFFLED_HISTORY,
        ControllerPolicyMode.IRRELEVANT_HISTORY,
        ControllerPolicyMode.CONSERVATIVE_HISTORY,
        ControllerPolicyMode.FIXED_BUDGET_FRESH,
        ControllerPolicyMode.OFFLINE_HINDSIGHT,
    ]

    policy_summaries: dict[str, Any] = {}
    for pol in policies:
        logger.info("Running Stream for Policy: %s", pol.value)
        summary = CausalExperienceBenchmark.evaluate_policy_on_stream(
            model=model,
            policy_mode=pol,
            seed=seed,
            device=device,
            initial_history=initial_history,
            num_episodes=12,
        )
        policy_summaries[pol.value] = asdict(summary)
        logger.info(
            "[%s] Macro EM: %.4f, Utility: %.4f, Steps: %d, Reuses: %d",
            pol.value,
            summary.macro_exact_match,
            summary.physical_compute_utility,
            summary.total_training_steps,
            summary.successful_reuses,
        )

    # 4. Phase 7 & 8: Safeguards, Rollback, Capacity & Hardware Accounting
    logger.info("--- Phase 7 & 8: Safeguards, Rollback, Capacity Diagnostics ---")
    base_invariance = M75SafeguardsAuditor.verify_base_invariance(
        model, initial_unnamed_hash, initial_named_hash
    )
    carry_rate = M75SafeguardsAuditor.verify_arithmetic_carry_retention()
    logic_rate = M75SafeguardsAuditor.verify_chained_operational_logic()
    ad_test = copy.deepcopy(model.blocks[0])  # Mock module for rollback test
    rollback_ok = M75SafeguardsAuditor.verify_bit_exact_rollback(ad_test)
    capacity_diag = M75SafeguardsAuditor.compare_capacity_variants(d_model=128, num_symbols=16)
    hw_footprint = M75SafeguardsAuditor.audit_hardware_footprint(
        base_model=model,
        adapter_repertoire_size=policy_summaries[ControllerPolicyMode.RELEVANT_HISTORY.value][
            "final_repertoire_size"
        ],
    )

    return {
        "seed": seed,
        "phase1_audit": audit_results,
        "phase4_stream_policies": policy_summaries,
        "phase8_safeguards": {
            "base_invariance_verified": base_invariance,
            "arithmetic_carry_retention_rate": carry_rate,
            "chained_operational_logic_rate": logic_rate,
            "bit_exact_rollback_verified": rollback_ok,
            "capacity_diagnostics": capacity_diag,
            "hardware_footprint": hw_footprint,
            "initial_unnamed_hash": initial_unnamed_hash,
            "initial_named_hash": initial_named_hash,
        },
    }


def aggregate_multi_seed_results(
    seed_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute cross-seed summary statistics and paired Student's t confidence intervals."""
    n_seeds = len(seed_results)
    t_crit = 4.303  # 95% critical value for df=2

    policy_names = list(seed_results[0]["phase4_stream_policies"].keys())
    multi_seed_policies: dict[str, Any] = {}

    for pol in policy_names:
        ems = [s["phase4_stream_policies"][pol]["macro_exact_match"] for s in seed_results]
        utils = [
            s["phase4_stream_policies"][pol]["physical_compute_utility"] for s in seed_results
        ]
        steps = [s["phase4_stream_policies"][pol]["total_training_steps"] for s in seed_results]
        reuses = [s["phase4_stream_policies"][pol]["successful_reuses"] for s in seed_results]

        mean_em = float(np.mean(ems))
        std_em = float(np.std(ems, ddof=1)) if n_seeds > 1 else 0.0

        mean_u = float(np.mean(utils))
        std_u = float(np.std(utils, ddof=1)) if n_seeds > 1 else 0.0

        mean_steps = float(np.mean(steps))
        std_steps = float(np.std(steps, ddof=1)) if n_seeds > 1 else 0.0

        multi_seed_policies[pol] = {
            "mean_macro_exact_match": round(mean_em, 4),
            "std_macro_exact_match": round(std_em, 4),
            "mean_physical_compute_utility": round(mean_u, 4),
            "std_physical_compute_utility": round(std_u, 4),
            "mean_total_training_steps": round(mean_steps, 1),
            "std_total_training_steps": round(std_steps, 1),
            "mean_successful_reuses": round(float(np.mean(reuses)), 2),
        }

    # Paired Student's t comparisons: RELEVANT_HISTORY vs Controls
    def compute_paired_ci(
        treatment: str, control: str, metric_key: str
    ) -> dict[str, Any]:
        deltas = [
            s["phase4_stream_policies"][treatment][metric_key]
            - s["phase4_stream_policies"][control][metric_key]
            for s in seed_results
        ]
        mean_d = float(np.mean(deltas))
        std_d = float(np.std(deltas, ddof=1)) if n_seeds > 1 else 0.0
        se_d = std_d / math.sqrt(n_seeds)
        ci_lower = mean_d - t_crit * se_d
        ci_upper = mean_d + t_crit * se_d
        return {
            "treatment": treatment,
            "control": control,
            "metric": metric_key,
            "mean_delta": round(mean_d, 4),
            "std_delta": round(std_d, 4),
            "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
            "spans_zero": bool(ci_lower <= 0.0 <= ci_upper),
            "statistically_superior": bool(ci_lower > 0.0),
        }

    paired_comparisons = {
        "RELEVANT_vs_NO_HISTORY_utility": compute_paired_ci(
            "RELEVANT_HISTORY", "NO_HISTORY", "physical_compute_utility"
        ),
        "RELEVANT_vs_SHUFFLED_HISTORY_utility": compute_paired_ci(
            "RELEVANT_HISTORY", "SHUFFLED_HISTORY", "physical_compute_utility"
        ),
        "RELEVANT_vs_IRRELEVANT_HISTORY_utility": compute_paired_ci(
            "RELEVANT_HISTORY", "IRRELEVANT_HISTORY", "physical_compute_utility"
        ),
        "RELEVANT_vs_FIXED_BUDGET_utility": compute_paired_ci(
            "RELEVANT_HISTORY", "FIXED_BUDGET_FRESH", "physical_compute_utility"
        ),
        "CONSERVATIVE_vs_NO_HISTORY_utility": compute_paired_ci(
            "CONSERVATIVE_HISTORY", "NO_HISTORY", "physical_compute_utility"
        ),
    }

    # Safeguards summary
    base_inv = all(
        s["phase8_safeguards"]["base_invariance_verified"] for s in seed_results
    )
    carry_all = all(
        s["phase8_safeguards"]["arithmetic_carry_retention_rate"] == 1.0 for s in seed_results
    )
    logic_all = all(
        s["phase8_safeguards"]["chained_operational_logic_rate"] == 1.0 for s in seed_results
    )
    rollback_all = all(
        s["phase8_safeguards"]["bit_exact_rollback_verified"] for s in seed_results
    )

    # Generalization category breakdown under RELEVANT_HISTORY
    cat_breakdown: dict[str, list[float]] = {}
    for s in seed_results:
        cat_dict = s["phase4_stream_policies"]["RELEVANT_HISTORY"]["category_em"]
        for c_name, val in cat_dict.items():
            cat_breakdown.setdefault(c_name, []).append(val)

    generalization_summary = {
        c_name: round(float(np.mean(vals)), 4) for c_name, vals in cat_breakdown.items()
    }

    return {
        "status": "COMPLETED",
        "milestone": "M75",
        "num_seeds": n_seeds,
        "t_critical_95": t_crit,
        "phase1_audit_lineage": seed_results[0]["phase1_audit"],
        "phase4_multi_seed_policies": multi_seed_policies,
        "phase4_paired_comparisons": paired_comparisons,
        "phase5_generalization_summary": generalization_summary,
        "phase8_safeguards_summary": {
            "all_seeds_base_invariance": base_inv,
            "all_seeds_arithmetic_carry_100pct": carry_all,
            "all_seeds_logic_probes_100pct": logic_all,
            "all_seeds_rollback_verified": rollback_all,
            "capacity_diagnostics": seed_results[0]["phase8_safeguards"]["capacity_diagnostics"],
            "hardware_footprint": seed_results[0]["phase8_safeguards"]["hardware_footprint"],
        },
        "seed_level_results": seed_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M75 Experience-Guided Learning Benchmark")
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
        default=Path("runs/kaggle-m75/output/amg/m75_experience_learning"),
        help="Path to save m75_report.json",
    )
    args = parser.parse_args()

    # Preflight runtime check
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for Milestone M75 benchmark. Execution on local CPU is prohibited."
        )

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    logger.info("Executing M75 Benchmark on Remote GPU: %s", gpu_name)

    seed_results = []
    for seed in args.seeds:
        res = run_m75_benchmark_for_seed(seed, args.checkpoint_dir, device)
        seed_results.append(res)

    report = aggregate_multi_seed_results(seed_results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_file = args.output_dir / "m75_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved Milestone M75 Report to %s", report_file)
    logger.info("=== Milestone M75 Benchmark Execution Successfully Finished ===")


if __name__ == "__main__":
    main()
