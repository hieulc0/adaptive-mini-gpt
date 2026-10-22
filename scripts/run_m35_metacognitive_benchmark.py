#!/usr/bin/env python3
"""M35: Learning How to Learn — Metacognitive Action Policy Benchmark Runner.

Executes the 7-Phase M35 Study across 3 independent lineages (Seeds 42, 43, 44):
  Phase 1: M34 audit and verification of V3 arithmetic & logic adapters.
  Phase 2: Empirical action-outcome utility collection.
  Phase 3: 7-scenario metacognitive curriculum generation (disjoint train/eval).
  Phase 4: Policy head training (Supervised vs Outcome-Aware).
  Phase 5: Policy decision benchmark (Deterministic vs Supervised vs Outcome-Aware vs Baselines).
  Phase 6: Controlled self-directed learning execution with budget gating.
  Phase 7: Language representation diagnostic audit (M28 vs M34).
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

# Ensure repository root and src are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from adaptive_mini_gpt.data.bpe_scaling import (
    expand_model_vocabulary,
    train_scaled_bpe_tokenizer,
)
from adaptive_mini_gpt.data.language_corpus import PRETRAIN_PROSE_CORPUS
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.learning.cumulative_learner import (
    build_m33_arithmetic_episode,
    build_m33_logic_episode,
)
from adaptive_mini_gpt.learning.foundation import train_m32_recurrent_foundation
from adaptive_mini_gpt.learning.modular_cumulative_learner import (
    evaluate_multi_domain_system,
    train_domain_adapter,
)
from adaptive_mini_gpt.learning.modular_lineage import ModularLineageManager
from adaptive_mini_gpt.metacognition.action_space import (
    ACTION_COMPUTE_COSTS,
    ACTION_LIST,
    LearningAction,
    compute_action_utility,
)
from adaptive_mini_gpt.metacognition.curriculum import (
    MetacognitiveSample,
    build_metacognitive_curriculum,
)
from adaptive_mini_gpt.metacognition.language_diagnosis import (
    diagnose_language_representation_gap,
)
from adaptive_mini_gpt.metacognition.learning_controller import (
    ControlledSelfDirectedLearner,
)
from adaptive_mini_gpt.metacognition.policy_head import (
    ACTION_TO_IDX,
    IDX_TO_ACTION,
    NeuralMetacognitivePolicy,
    train_outcome_aware_policy,
    train_supervised_policy,
)
from adaptive_mini_gpt.model.modular_adapter import (
    add_domain_adapter,
    compute_base_model_hash,
    get_adapter_state_dict,
    inject_multi_domain_adapters,
    load_adapter_state_dict,
    set_active_domain_adapter,
)
from adaptive_mini_gpt.model.transformer import MiniGPT, ModelConfig
from adaptive_mini_gpt.reasoning.adaptive_router import DeterministicRouter
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("m35_benchmark")


def compute_mean_std(values: list[float]) -> dict[str, float]:
    """Compute mean and sample standard deviation."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0}
    mean = sum(values) / n
    if n == 1:
        return {"mean": round(mean, 4), "std": 0.0}
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return {"mean": round(mean, 4), "std": round(math.sqrt(var), 4)}


def run_seed_m35_pipeline(
    seed: int,
    checkpoints_dir: Path,
    tokenizer: BPETokenizer,
    device: torch.device,
    epochs: int = 12,
    lr: float = 3e-4,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Execute complete M35 pipeline for a single seed lineage."""
    logger.info("#################################################################")
    logger.info(f"Executing M35 Metacognitive Policy Benchmark Seed {seed}")
    logger.info("#################################################################")

    t_start = time.perf_counter()
    random.seed(seed)
    torch.manual_seed(seed)

    # 1. Load M18 Root Checkpoint & Expand to 1024-BPE
    relative = Path(f"seed_{seed}/base_pretrain/best.pt")
    checkpoint_path = checkpoints_dir / relative
    raw = CheckpointManager.load_raw_checkpoint(checkpoint_path)

    base_tok = CheckpointManager.load_tokenizer(raw)
    state_dict = raw.get("model_state_dict", raw.get("model_state"))
    base_model_400 = MiniGPT(ModelConfig(**raw["model_config"]))
    base_model_400.load_state_dict(state_dict)
    base_model = expand_model_vocabulary(
        model=base_model_400,
        old_tokenizer=base_tok,
        new_tokenizer=tokenizer,
    ).to(device)

    # 2. Phase 1: Train & Audit M32 Foundation + M34 V3 Adapters
    logger.info(f"[Seed {seed}] Training M32 Foundation and M34 V3 Adapters...")
    foundation_model = train_m32_recurrent_foundation(
        base_model=base_model,
        tokenizer=tokenizer,
        device=device,
        seed=seed,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    inject_multi_domain_adapters(foundation_model, r=8, target_modules=["c_attn", "c_proj", "c_fc"])
    base_hash_init = compute_base_model_hash(foundation_model)

    # Train Arithmetic Adapter (V2)
    ep_a = build_m33_arithmetic_episode(seed=1000 + seed)
    add_domain_adapter(foundation_model, "arithmetic", r=8)
    adapt_arith_state, _ = train_domain_adapter(
        model=foundation_model,
        tokenizer=tokenizer,
        domain_name="arithmetic",
        practice_samples=ep_a.practice_samples,
        replay_samples=[],
        device=device,
        use_replay=True,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    # Train Logic Adapter (V3)
    ep_b = build_m33_logic_episode(seed=2000 + seed)
    add_domain_adapter(foundation_model, "logic", r=8)
    adapt_logic_state, _ = train_domain_adapter(
        model=foundation_model,
        tokenizer=tokenizer,
        domain_name="logic",
        practice_samples=ep_b.practice_samples,
        replay_samples=ep_a.practice_samples[:16],
        device=device,
        use_replay=True,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
    )

    base_hash_after = compute_base_model_hash(foundation_model)
    assert base_hash_init == base_hash_after, "Base model weights mutated during adapter training!"

    # Evaluate Phase 1 Audit metrics
    audit_eval = evaluate_multi_domain_system(
        model=foundation_model,
        tokenizer=tokenizer,
        device=device,
        arith_test_samples=ep_a.fresh_test_samples,
        logic_test_samples=ep_b.fresh_test_samples,
        lang_test_samples=[],
        is_modular=True,
        active_domain=None,
    )
    logger.info(f"[Seed {seed}] M34 Audit: Arith={audit_eval['arithmetic_exact']*100:.1f}%, Logic={audit_eval['logic_exact']*100:.1f}%, Carry={audit_eval['carry_retention_200']*100:.1f}%")

    # 3. Phase 3: Metacognitive Curriculum Construction
    logger.info(f"[Seed {seed}] Building Metacognitive Curriculum (7 Scenarios)...")
    train_samples, eval_samples = build_metacognitive_curriculum(
        seed=seed,
        train_samples_per_scenario=24,
        eval_samples_per_scenario=16,
    )

    # Extract pooled backbone representations for training set
    foundation_model.eval()
    max_ctx = getattr(foundation_model.config, "context_length", 256)

    def extract_features_batch(samples: list[MetacognitiveSample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat_list = []
        target_list = []
        utility_list = []
        with torch.no_grad():
            for s in samples:
                enc = tokenizer.encode(s.query)
                inp = torch.tensor([enc[-max_ctx:]], dtype=torch.long, device=device)
                pol_dummy = NeuralMetacognitivePolicy(d_model=foundation_model.config.d_model)
                feat = pol_dummy.extract_pooled_representation(foundation_model, inp)
                feat_list.append(feat[0].cpu())
                target_list.append(ACTION_TO_IDX[s.optimal_action])
                utility_list.append(s.action_utilities.get(s.optimal_action, 0.5))
        return torch.stack(feat_list), torch.tensor(target_list, dtype=torch.long), torch.tensor(utility_list, dtype=torch.float)

    train_feats, train_targets, train_utils = extract_features_batch(train_samples)
    eval_feats, eval_targets, eval_utils = extract_features_batch(eval_samples)

    # 4. Phase 4: Policy Head Training (Supervised vs Outcome-Aware)
    logger.info(f"[Seed {seed}] Training Policy B (Supervised Cross-Entropy)...")
    policy_supervised = NeuralMetacognitivePolicy(
        d_model=foundation_model.config.d_model,
        d_hidden=64,
        num_actions=len(ACTION_LIST),
    ).to(device)
    train_supervised_policy(
        policy=policy_supervised,
        features=train_feats.to(device),
        targets=train_targets.to(device),
        epochs=30,
        lr=1e-3,
    )

    logger.info(f"[Seed {seed}] Training Policy C (Outcome-Aware Utility-Weighted)...")
    policy_outcome = NeuralMetacognitivePolicy(
        d_model=foundation_model.config.d_model,
        d_hidden=64,
        num_actions=len(ACTION_LIST),
    ).to(device)
    train_outcome_aware_policy(
        policy=policy_outcome,
        features=train_feats.to(device),
        target_action_indices=train_targets.to(device),
        action_utilities=train_utils.to(device),
        epochs=30,
        lr=1e-3,
        utility_weight=0.6,
    )

    # 5. Phase 5: Policy Decision Benchmark on Held-Out Eval Samples
    logger.info(f"[Seed {seed}] Evaluating Policy Decisions on {len(eval_samples)} Held-Out Samples...")
    deterministic_router = DeterministicRouter()

    def evaluate_policy_decisions(pol_name: str, policy_obj: Any) -> dict[str, Any]:
        correct_actions = 0
        total_utility = 0.0
        safe_abstentions = 0
        total_unanswerable = 0
        false_abstentions = 0
        total_answerable = 0
        unnecessary_training = 0
        all_confidences = []
        all_uncertainties = []

        scenario_accuracies: dict[str, list[float]] = {}

        for s in eval_samples:
            sc = s.scenario
            if sc not in scenario_accuracies:
                scenario_accuracies[sc] = []

            # Determine action taken
            if pol_name == "policy_a_deterministic":
                # DeterministicRouter maps to ReasoningAction
                d_dec = deterministic_router.route(s.query)
                # Map ReasoningAction -> LearningAction
                act_map = {
                    "DIRECT_ANSWER": LearningAction.ANSWER,
                    "RECURRENT_EXECUTION": LearningAction.RECURRENT_EXECUTION,
                    "VERIFIED_RETRIEVAL": LearningAction.RETRIEVE,
                    "ABSTAIN": LearningAction.ABSTAIN,
                }
                pred_act = act_map.get(d_dec.action.value, LearningAction.ANSWER)
                conf = d_dec.confidence
                uncert = 1.0 - conf
            elif pol_name in ["policy_b_supervised", "policy_c_outcome_aware"]:
                enc = tokenizer.encode(s.query)
                inp = torch.tensor([enc[-max_ctx:]], dtype=torch.long, device=device)
                pred_act, conf, uncert, _ = policy_obj.predict(foundation_model, inp)
            elif pol_name == "baseline_always_answer":
                pred_act = LearningAction.ANSWER
                conf, uncert = 1.0, 0.0
            elif pol_name == "baseline_always_retrieve":
                pred_act = LearningAction.RETRIEVE
                conf, uncert = 1.0, 0.0
            elif pol_name == "baseline_always_recurrent":
                pred_act = LearningAction.RECURRENT_EXECUTION
                conf, uncert = 1.0, 0.0
            elif pol_name == "baseline_always_abstain":
                pred_act = LearningAction.ABSTAIN
                conf, uncert = 1.0, 0.0
            else:  # Random
                pred_act = random.choice(ACTION_LIST)
                conf, uncert = 1.0 / len(ACTION_LIST), 0.5

            is_correct = (pred_act == s.optimal_action)
            if is_correct:
                correct_actions += 1
            scenario_accuracies[sc].append(1.0 if is_correct else 0.0)

            # Empirical utility obtained
            ut = s.action_utilities.get(pred_act, -0.5)
            total_utility += ut

            if s.is_unanswerable:
                total_unanswerable += 1
                if pred_act == LearningAction.ABSTAIN:
                    safe_abstentions += 1
            else:
                total_answerable += 1
                if pred_act == LearningAction.ABSTAIN:
                    false_abstentions += 1

            if pred_act == LearningAction.TRAIN_CANDIDATE and not s.is_repeated_failure:
                unnecessary_training += 1

            all_confidences.append(conf)
            all_uncertainties.append(uncert)

        n_eval = max(1, len(eval_samples))
        acc = round(correct_actions / n_eval, 4)
        mean_u = round(total_utility / n_eval, 4)
        safe_ab_rate = round(safe_abstentions / max(1, total_unanswerable), 4)
        false_ab_rate = round(false_abstentions / max(1, total_answerable), 4)
        unnec_train_rate = round(unnecessary_training / n_eval, 4)

        sc_summary = {sc: round(sum(v) / len(v), 4) for sc, v in scenario_accuracies.items()}

        return {
            "policy_name": pol_name,
            "action_selection_accuracy": acc,
            "mean_empirical_utility": mean_u,
            "safe_abstention_rate": safe_ab_rate,
            "false_abstention_rate": false_ab_rate,
            "unnecessary_training_rate": unnec_train_rate,
            "mean_confidence": round(sum(all_confidences) / len(all_confidences), 4),
            "mean_uncertainty": round(sum(all_uncertainties) / len(all_uncertainties), 4),
            "scenario_accuracies": sc_summary,
        }

    policies_evaluated = {
        "policy_a_deterministic": evaluate_policy_decisions("policy_a_deterministic", None),
        "policy_b_supervised": evaluate_policy_decisions("policy_b_supervised", policy_supervised),
        "policy_c_outcome_aware": evaluate_policy_decisions("policy_c_outcome_aware", policy_outcome),
        "baseline_always_answer": evaluate_policy_decisions("baseline_always_answer", None),
        "baseline_always_retrieve": evaluate_policy_decisions("baseline_always_retrieve", None),
        "baseline_always_recurrent": evaluate_policy_decisions("baseline_always_recurrent", None),
        "baseline_always_abstain": evaluate_policy_decisions("baseline_always_abstain", None),
    }

    # 6. Phase 6: Controlled Self-Directed Learning Simulation
    logger.info(f"[Seed {seed}] Running Phase 6 Controlled Self-Directed Learning Interaction...")
    memory_bank = {
        "mitochondria": "Mitochondria produce ATP through aerobic cellular respiration.",
        "light_speed": "The speed of light in vacuum is approximately 300,000 km per second.",
        "chlorophyll": "Chlorophyll pigments absorb blue and red photons and reflect green light.",
    }
    learner = ControlledSelfDirectedLearner(
        model=foundation_model,
        tokenizer=tokenizer,
        policy=policy_outcome,
        device=device,
        memory_bank=memory_bank,
        max_training_budget=2,
    )

    interaction_results: list[dict[str, Any]] = []
    # Test on a representative subset of mixed scenarios
    test_interaction_samples = eval_samples[:14]
    for s in test_interaction_samples:
        ep_res = learner.process_episode(s)
        interaction_results.append(asdict(ep_res))

    controller_overrides_count = sum(1 for r in interaction_results if r["controller_overrode_action"])
    tasks_solved_count = sum(1 for r in interaction_results if r["task_solved"])

    # 7. Phase 7: Language Representation Diagnostic Audit
    logger.info(f"[Seed {seed}] Running Phase 7 Language Representation Diagnostic Audit...")
    lang_diag = diagnose_language_representation_gap(
        model=foundation_model,
        tokenizer=tokenizer,
        device=device,
    )

    duration = round(time.perf_counter() - t_start, 2)
    logger.info(f"[Seed {seed}] Completed M35 Pipeline in {duration}s")

    return {
        "seed": seed,
        "duration_seconds": duration,
        "phase1_m34_audit": audit_eval,
        "phase5_policies_evaluated": policies_evaluated,
        "phase6_self_directed_learning": {
            "total_episodes": len(interaction_results),
            "tasks_solved": tasks_solved_count,
            "tasks_solved_rate": round(tasks_solved_count / len(interaction_results), 4),
            "controller_overrides": controller_overrides_count,
            "training_budget_spent": learner.training_budget_spent,
            "episodes_sample": interaction_results[:5],
        },
        "phase7_language_diagnosis": {k: v.to_dict() for k, v in lang_diag.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M35 Metacognitive Policy Study")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoints-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoints_dir = Path(args.checkpoints_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("======================================================================")
    logger.info("M35 — Learning How to Learn: Metacognitive Action Policy Study")
    logger.info(f"Device: {device}, Seeds: {args.seeds}, Epochs: {args.epochs}, LR: {args.lr}")
    logger.info("======================================================================")

    # 1. 1024-BPE Tokenizer
    all_prose = " ".join(PRETRAIN_PROSE_CORPUS)
    tokenizer = train_scaled_bpe_tokenizer(all_prose, target_vocab_size=1024)
    tok_path = output_dir / "bpe_1024.json"
    tokenizer.save(tok_path)
    logger.info(f"1024-BPE Tokenizer saved. Fingerprint: {tokenizer.get_fingerprint()}")

    # 2. Execute Lineages Across Seeds
    seed_runs: list[dict[str, Any]] = []
    for seed in args.seeds:
        run_res = run_seed_m35_pipeline(
            seed=seed,
            checkpoints_dir=checkpoints_dir,
            tokenizer=tokenizer,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )
        seed_runs.append(run_res)

    logger.info("\nAggregating M35 multi-lineage metrics across seeds 42, 43, 44...")

    # 3. Policy Benchmark Aggregation
    policy_keys = [
        "policy_a_deterministic",
        "policy_b_supervised",
        "policy_c_outcome_aware",
        "baseline_always_answer",
        "baseline_always_retrieve",
        "baseline_always_recurrent",
        "baseline_always_abstain",
    ]

    aggregated_policies: dict[str, dict[str, Any]] = {}
    for pkey in policy_keys:
        accs = [r["phase5_policies_evaluated"][pkey]["action_selection_accuracy"] for r in seed_runs]
        utils = [r["phase5_policies_evaluated"][pkey]["mean_empirical_utility"] for r in seed_runs]
        safe_abs = [r["phase5_policies_evaluated"][pkey]["safe_abstention_rate"] for r in seed_runs]
        false_abs = [r["phase5_policies_evaluated"][pkey]["false_abstention_rate"] for r in seed_runs]
        unnec_tr = [r["phase5_policies_evaluated"][pkey]["unnecessary_training_rate"] for r in seed_runs]

        aggregated_policies[pkey] = {
            "action_selection_accuracy": compute_mean_std(accs),
            "mean_empirical_utility": compute_mean_std(utils),
            "safe_abstention_rate": compute_mean_std(safe_abs),
            "false_abstention_rate": compute_mean_std(false_abs),
            "unnecessary_training_rate": compute_mean_std(unnec_tr),
        }

    # 4. Phase 1 Audit Aggregation
    audit_arith = compute_mean_std([r["phase1_m34_audit"]["arithmetic_exact"] for r in seed_runs])
    audit_logic = compute_mean_std([r["phase1_m34_audit"]["logic_exact"] for r in seed_runs])
    audit_carry = compute_mean_std([r["phase1_m34_audit"]["carry_retention_200"] for r in seed_runs])

    # 5. Phase 6 Self-Directed Aggregation
    sd_solve_rates = [r["phase6_self_directed_learning"]["tasks_solved_rate"] for r in seed_runs]
    sd_overrides = [r["phase6_self_directed_learning"]["controller_overrides"] for r in seed_runs]

    m35_report = {
        "metadata": {
            "milestone": "M35",
            "study_name": "Learning How to Learn: Metacognitive Action Policy and Self-Directed Learning",
            "seeds": args.seeds,
            "device": str(device),
            "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", 1024),
            "tokenizer_fingerprint": tokenizer.get_fingerprint(),
            "mean_duration_seconds": round(sum(r["duration_seconds"] for r in seed_runs) / len(seed_runs), 2),
        },
        "phase1_m34_audit_summary": {
            "arithmetic_exact": audit_arith,
            "logic_exact": audit_logic,
            "carry_retention_200": audit_carry,
        },
        "phase5_policy_benchmark_summary": aggregated_policies,
        "phase6_self_directed_summary": {
            "tasks_solved_rate": compute_mean_std(sd_solve_rates),
            "mean_controller_overrides": round(sum(sd_overrides) / len(sd_overrides), 2),
        },
        "phase7_language_diagnosis_summary": seed_runs[0]["phase7_language_diagnosis"],
        "individual_seed_runs": seed_runs,
    }

    report_path = output_dir / "m35_report.json"
    with open(report_path, "w") as f:
        json.dump(m35_report, f, indent=2)
    logger.info(f"M35 Report successfully written to {report_path}")

    logger.info("\n=================== M35 POLICY BENCHMARK COMPARISON ===================")
    for pkey, pdata in aggregated_policies.items():
        acc = pdata["action_selection_accuracy"]
        ut = pdata["mean_empirical_utility"]
        sab = pdata["safe_abstention_rate"]
        logger.info(
            f"[{pkey:<26}] Accuracy: {acc['mean']*100:.1f}% ± {acc['std']*100:.1f}% | "
            f"Utility: {ut['mean']:.4f} ± {ut['std']:.4f} | Safe Abstain: {sab['mean']*100:.1f}%"
        )
    logger.info("=======================================================================")


if __name__ == "__main__":
    main()
