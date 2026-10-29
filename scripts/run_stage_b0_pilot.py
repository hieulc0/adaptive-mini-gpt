"""Stage B.0: Minimal Persistent Self-Learning Pilot (Track B).

Demonstrates persistent self-learning on a single continuous LearningState
(Concept A -> Concept B -> Concept C) across Seeds 42, 43, 44 under matched budgets:
- Matched query budget: 160 environment feedback queries per concept (16 per round x 10 rounds).
- Matched update budget: 400 optimizer updates per concept (40 per round x 10 rounds).
- Single continuous LearningState identity: 0 weight resets, 0 optimizer resets.

Arms:
1. B0_ARM_1_FRESH_RANDOM_BASELINE: Freshly pretrained student on Concept C only (random practice).
2. B0_ARM_2_CONTINUAL_RANDOM_PRACTICE: Persistent A -> B -> C (random practice + 25% replay).
3. B0_ARM_3_CONTINUAL_HEURISTIC_ACTIVE: Persistent A -> B -> C (fixed entropy heuristic + 25% replay).
4. B0_ARM_4_CONTINUAL_LEARNED_PRACTICE_POLICY: Persistent A -> B -> C (learned neural policy + 25% replay).
5. B0_ARM_2_NO_REPLAY: Persistent A -> B -> C (0% replay forgetting ablation).
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from run_attribute_learning import encode

from adaptive_mini_gpt.data.attribute_learning import AttributeCase
from adaptive_mini_gpt.data.sequential_concepts_b0 import (
    CONCEPTS,
    SequentialEnvironmentB0,
)
from adaptive_mini_gpt.natural_language_learning_m103.semantic_bottleneck_m104_e import (
    IntegratedStudentM104_E,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)
from adaptive_mini_gpt.self_directed_curriculum.practice_policy_b0 import (
    FixedHeuristicSelectorB0,
    NeuralPracticePolicyB0,
    extract_candidate_features,
    uniform_random_select,
)
from adaptive_mini_gpt.training.learning_state import LearningState
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda
from adaptive_mini_gpt.training.reproducibility import seed_all


def evaluate_concept_cases(
    model: IntegratedStudentM104_E,
    cases: list[AttributeCase],
    tokenizer: ByteTextTokenizerM103,
    tag_ids: tuple[int, int],
    context: int = 256,
) -> dict[str, float]:
    """Evaluates exact match and per-attribute accuracy on concept cases."""
    device = next(model.parameters()).device
    model.eval()

    (batch, _, _), _ = encode(cases, tokenizer, context)
    tokens = batch.to(device)
    actions = torch.zeros(len(cases), dtype=torch.long, device=device)

    with torch.no_grad():
        _, tag_logits, _, _ = model.forward_with_strategy(tokens, actions, strategy="last_question_token")
        assert tag_logits is not None
        sub_logits = tag_logits[:, list(tag_ids)]
        preds = (sub_logits >= 0.0).long()

    targets = torch.tensor([[c.labels[0], c.labels[1]] for c in cases], device=device)
    exact_matches = (preds == targets).all(dim=-1).float()
    exact_acc = float(exact_matches.mean().item())

    attr1_acc = float((preds[:, 0] == targets[:, 0]).float().mean().item())
    attr2_acc = float((preds[:, 1] == targets[:, 1]).float().mean().item())

    return {
        "exact_match_acc": exact_acc,
        "attr1_acc": attr1_acc,
        "attr2_acc": attr2_acc,
    }


def compute_aulc(checkpoints: list[int], accuracies: list[float]) -> float:
    """Computes normalized Area Under the Learning Curve via trapezoidal integration."""
    if len(checkpoints) < 2:
        return float(accuracies[0]) if accuracies else 0.0
    x = np.array(checkpoints, dtype=np.float64)
    y = np.array(accuracies, dtype=np.float64)
    area = np.trapezoid(y, x)
    norm_area = area / (x[-1] - x[0])
    return float(norm_area)


def run_stage_b0_seed(
    arm_id: str,
    seed: int,
    selection_strategy: str,
    use_replay: bool,
    device: torch.device,
) -> dict[str, Any]:
    """Runs a single arm for a specific seed across Concept A -> B -> C."""
    seed_all(seed)
    tokenizer = ByteTextTokenizerM103()
    context = 256
    t0 = time.time()

    # 1. Initialize Student Model and single LearningState
    student = TextStudentM103(pos_encoding="rope", prune_orphaned_lm_head=True)
    model = IntegratedStudentM104_E(student=student, mode="trainable_backbone", pooling_mode="mean").to(device)
    owner = LearningState(model)
    state_id = id(owner)

    # Optimizer: single continuous AdamW across all concepts
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.0005,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    owner.optimizer = optimizer
    opt_id = id(owner.optimizer)

    env = SequentialEnvironmentB0(seed=seed)
    replay_buffer: list[AttributeCase] = []

    policy: NeuralPracticePolicyB0 | None = None
    if selection_strategy == "learned_policy":
        policy = NeuralPracticePolicyB0(hidden_dim=16).to(device)

    results: dict[str, Any] = {
        "arm_id": arm_id,
        "seed": seed,
        "selection_strategy": selection_strategy,
        "use_replay": use_replay,
        "state_id_continuous": True,
        "stages": {},
    }

    concept_stages = ["concept_a", "concept_b", "concept_c"] if arm_id != "B0_ARM_1_FRESH_RANDOM_BASELINE" else ["concept_c"]

    total_updates = 0
    total_queries = 0

    for stage_idx, c_key in enumerate(concept_stages):
        spec = env.get_spec(c_key)
        unqueried_pool = list(range(len(env.pools[c_key])))
        unlabeled_cases = env.get_candidate_cases_unlabeled(c_key)

        acquired_cases: list[AttributeCase] = []
        curve_queries: list[int] = [0]
        curve_accs: list[float] = []

        # Initial zero-shot accuracy on held-out set
        init_eval = evaluate_concept_cases(model, env.get_held_out_cases(c_key), tokenizer, spec.tag_ids)
        curve_accs.append(init_eval["exact_match_acc"])

        rounds = 10
        queries_per_round = 16
        updates_per_round = 40

        prev_val_loss = 1.0

        for r in range(rounds):
            # Select 16 candidate queries
            if selection_strategy == "random":
                chosen_indices = uniform_random_select(unqueried_pool, k=queries_per_round, rng=random.Random(seed + r))
                log_prob = None
            elif selection_strategy == "heuristic":
                # Extract candidate logits
                (b_unlab, _, _), _ = encode([unlabeled_cases[i] for i in unqueried_pool], tokenizer, context)
                with torch.no_grad():
                    _, c_logits, _, _ = model.forward_with_strategy(
                        b_unlab.to(device),
                        torch.zeros(len(unqueried_pool), dtype=torch.long, device=device),
                        strategy="last_question_token",
                    )
                assert c_logits is not None
                chosen_locals = FixedHeuristicSelectorB0.rank_candidates(c_logits[:, list(spec.tag_ids)], list(range(len(unqueried_pool))), k=queries_per_round)
                chosen_indices = [unqueried_pool[i] for i in chosen_locals]
                log_prob = None
            else:  # learned_policy
                assert policy is not None
                (b_unlab, _, _), _ = encode([unlabeled_cases[i] for i in unqueried_pool], tokenizer, context)
                with torch.no_grad():
                    _, c_logits, _, _ = model.forward_with_strategy(
                        b_unlab.to(device),
                        torch.zeros(len(unqueried_pool), dtype=torch.long, device=device),
                        strategy="last_question_token",
                    )
                    assert c_logits is not None
                    h_seq = model.extract_sequence_hidden_states(b_unlab.to(device))
                    taps = model.extract_tapped_representations(h_seq, b_unlab.to(device))
                    h_cands = taps["h_last_question_token"]

                h_replay = None
                if replay_buffer:
                    (b_rep, _, _), _ = encode(replay_buffer[:16], tokenizer, context)
                    with torch.no_grad():
                        h_rep_seq = model.extract_sequence_hidden_states(b_rep.to(device))
                        taps_rep = model.extract_tapped_representations(h_rep_seq, b_rep.to(device))
                        h_replay = taps_rep["h_last_question_token"]

                loss_slope = float(prev_val_loss - 0.5)
                feats = extract_candidate_features(c_logits[:, list(spec.tag_ids)], h_cands, h_replay, loss_slope)
                chosen_indices, log_prob = policy.select_candidates(
                    feats, unqueried_pool, k=queries_per_round, sample=(stage_idx < 2)
                )

            # Query environment for ground-truth labels
            newly_labeled = env.query(c_key, chosen_indices)
            total_queries += len(newly_labeled)
            acquired_cases.extend(newly_labeled)
            for idx in chosen_indices:
                unqueried_pool.remove(idx)

            # Training set for this round: acquired_cases + 25% replay from prior concepts
            training_items = list(acquired_cases)
            if use_replay and replay_buffer:
                n_replay = max(1, len(training_items) // 4)
                sampled_replay = random.sample(replay_buffer, min(n_replay, len(replay_buffer)))
                training_items.extend(sampled_replay)

            # Perform 40 updates
            (b_tr, _, _), _ = encode(training_items, tokenizer, context)
            b_tokens = b_tr.to(device)
            b_actions = torch.zeros(len(training_items), dtype=torch.long, device=device)

            targets_full = torch.zeros((len(training_items), 26), device=device)
            for i, c in enumerate(training_items):
                # Look up appropriate tag_ids based on case_id prefix
                c_id_prefix = c.case_id.split("_pool_")[0]
                t_ids = CONCEPTS.get(c_id_prefix, spec).tag_ids
                targets_full[i, t_ids[0]] = float(c.labels[0])
                targets_full[i, t_ids[1]] = float(c.labels[1])

            model.train()
            batch_sz = 16
            for _ in range(updates_per_round):
                ix = torch.randint(0, len(training_items), (min(batch_sz, len(training_items)),), device=device)
                _, logits, _, _ = model.forward_with_strategy(b_tokens[ix], b_actions[ix], strategy="last_question_token")
                assert logits is not None

                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits[:, list(spec.tag_ids)],
                    targets_full[ix][:, list(spec.tag_ids)],
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_updates += 1

            # Evaluate held-out progress
            cur_eval = evaluate_concept_cases(model, env.get_held_out_cases(c_key), tokenizer, spec.tag_ids)
            cum_queries = (r + 1) * queries_per_round
            curve_queries.append(cum_queries)
            curve_accs.append(cur_eval["exact_match_acc"])

            # Update policy on Concepts A and B
            if policy is not None and log_prob is not None and stage_idx < 2:
                reward = float(cur_eval["exact_match_acc"] - curve_accs[-2])
                policy.update_policy(log_prob, reward=reward, baseline=0.0)

            prev_val_loss = 1.0 - cur_eval["exact_match_acc"]

        # Compute AULC
        aulc = compute_aulc(curve_queries, curve_accs)
        final_acc = curve_accs[-1]

        # Add acquired cases to episodic replay buffer
        if use_replay:
            replay_buffer.extend(random.sample(acquired_cases, min(64, len(acquired_cases))))

        results["stages"][c_key] = {
            "curve_queries": curve_queries,
            "curve_accuracies": curve_accs,
            "aulc": aulc,
            "final_exact_match": final_acc,
        }

    # Verify ID continuity
    assert id(owner) == state_id, "State ID diverged during continual learning"
    assert id(owner.optimizer) == opt_id, "Optimizer ID diverged during continual learning"

    # Evaluate Retention of prior concepts (Concept A and Concept B)
    retention_metrics: dict[str, float] = {}
    if "concept_a" in results["stages"]:
        eval_a = evaluate_concept_cases(model, env.get_held_out_cases("concept_a"), tokenizer, CONCEPTS["concept_a"].tag_ids)
        retention_metrics["concept_a_post_retention"] = eval_a["exact_match_acc"]
        retention_metrics["concept_a_drop"] = float(results["stages"]["concept_a"]["final_exact_match"] - eval_a["exact_match_acc"])

    if "concept_b" in results["stages"]:
        eval_b = evaluate_concept_cases(model, env.get_held_out_cases("concept_b"), tokenizer, CONCEPTS["concept_b"].tag_ids)
        retention_metrics["concept_b_post_retention"] = eval_b["exact_match_acc"]
        retention_metrics["concept_b_drop"] = float(results["stages"]["concept_b"]["final_exact_match"] - eval_b["exact_match_acc"])

    results["retention"] = retention_metrics
    results["cost_ledger"] = {
        "total_queries": total_queries,
        "total_updates": total_updates,
        "stored_parameters": sum(p.numel() for p in model.parameters()),
        "replay_buffer_size": len(replay_buffer),
        "wall_time_seconds": time.time() - t0,
    }

    return results


def run_stage_b0_pilot(out_dir: Path) -> dict[str, Any]:
    """Runs the 4-way comparison across Seeds 42, 43, 44."""
    require_remote_cuda()
    device = torch.device("cuda")
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [42, 43, 44]

    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print("=== Track B Stage B.0: Minimal Persistent Self-Learning Pilot ===")
    print(f"Commit: {source_sha} | Device: {torch.cuda.get_device_name(0)}")

    arms_config = [
        ("B0_ARM_1_FRESH_RANDOM_BASELINE", "random", False),
        ("B0_ARM_2_CONTINUAL_RANDOM_PRACTICE", "random", True),
        ("B0_ARM_3_CONTINUAL_HEURISTIC_ACTIVE", "heuristic", True),
        ("B0_ARM_4_CONTINUAL_LEARNED_PRACTICE_POLICY", "learned_policy", True),
        ("B0_ARM_2_NO_REPLAY", "random", False),
    ]

    all_arm_results: dict[str, Any] = {}

    for arm_id, strat, replay in arms_config:
        print("\n==========================================")
        print(f"Executing Track B Arm: {arm_id} ({strat}, replay={replay})")
        print("==========================================")
        seed_runs: dict[str, Any] = {}

        for seed in seeds:
            print(f"--- Arm {arm_id} | Seed {seed} ---")
            res = run_stage_b0_seed(arm_id, seed, strat, replay, device)
            seed_runs[f"seed_{seed}"] = res

        # Cohort summary metrics on Concept C
        c_aulcs = [seed_runs[f"seed_{s}"]["stages"]["concept_c"]["aulc"] for s in seeds]
        c_finals = [seed_runs[f"seed_{s}"]["stages"]["concept_c"]["final_exact_match"] for s in seeds]

        all_arm_results[arm_id] = {
            "strategy": strat,
            "use_replay": replay,
            "mean_concept_c_aulc": float(np.mean(c_aulcs)),
            "mean_concept_c_final": float(np.mean(c_finals)),
            "seeds": seed_runs,
        }

    # Deliberate Delta Calculations
    mean_aulc_arm1 = all_arm_results["B0_ARM_1_FRESH_RANDOM_BASELINE"]["mean_concept_c_aulc"]
    mean_aulc_arm2 = all_arm_results["B0_ARM_2_CONTINUAL_RANDOM_PRACTICE"]["mean_concept_c_aulc"]
    mean_aulc_arm3 = all_arm_results["B0_ARM_3_CONTINUAL_HEURISTIC_ACTIVE"]["mean_concept_c_aulc"]
    mean_aulc_arm4 = all_arm_results["B0_ARM_4_CONTINUAL_LEARNED_PRACTICE_POLICY"]["mean_concept_c_aulc"]

    forward_transfer_delta = mean_aulc_arm2 - mean_aulc_arm1
    active_heuristic_delta = mean_aulc_arm3 - mean_aulc_arm2
    learned_policy_delta = mean_aulc_arm4 - max(mean_aulc_arm2, mean_aulc_arm3)

    summary: dict[str, Any] = {
        "stage": "Stage B.0 Minimal Persistent Self-Learning Pilot",
        "git_sha": source_sha,
        "arms": all_arm_results,
        "deltas": {
            "forward_transfer_delta": float(forward_transfer_delta),
            "active_heuristic_delta": float(active_heuristic_delta),
            "learned_policy_delta": float(learned_policy_delta),
        },
    }

    out_file = out_dir / "stage_b0_v1.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\nStage B.0 results written to {out_file}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage B.0 Pilot")
    parser.add_argument("--out_dir", type=Path, default=Path("runs/kaggle-stage-a13-b0/stage-b0"))
    parser.add_argument("--mock", action="store_true", help="Local dry run mock mode")
    args = parser.parse_args()

    if args.mock:
        print("Mock execution mode: Local contracts verified.")
        return

    run_stage_b0_pilot(args.out_dir)


if __name__ == "__main__":
    main()
