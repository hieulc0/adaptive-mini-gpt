"""Phase 5: Structural Minimal-Pair Benchmark and Paired Supervision Engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.guided_experiment.environment import (
    AutonomousEnvironment,
    EnvironmentQuery,
    MinimalPairControl,
    RuleHypothesis,
)


@dataclass
class MinimalPairEvaluationResult:
    """Outcome of evaluating minimal graph pairs."""
    total_pairs: int
    both_correct_count: int
    paired_correctness_rate: float
    base_accuracy: float
    pert_accuracy: float
    predicted_distribution: dict[str, int]
    failure_breakdown: dict[str, int] = field(default_factory=dict)


class MinimalPairEngine:
    """Constructs, validates, and supervises balanced minimal graph pairs."""

    @classmethod
    def generate_controlled_minimal_pairs(
        cls,
        env: AutonomousEnvironment,
        max_pairs: int = 12,
    ) -> list[MinimalPairControl]:
        """Synthesize verified minimal pairs where a single edge perturbation flips ground-truth reachability."""
        pairs: list[MinimalPairControl] = []
        entities = env.entities
        base_edges = list(env.base_edges)
        edge_set = set(base_edges)

        # 1. Bridge Deletions: If u -> v is a critical edge, removing it turns YES -> NO or UNKNOWN
        for u, v in base_edges:
            ans_base = env.oracle_evaluate_query(u, v)
            if ans_base == "YES":
                perturbed_edges = [e for e in base_edges if e != (u, v)]
                ans_pert = env.hypothesis.evaluate_query(
                    source=u,
                    target=v,
                    edges=perturbed_edges,
                    entity_levels=env.entity_levels,
                    blocked_nodes=env.blocked_nodes,
                    negated_edges=env.negated_edges,
                )
                if ans_pert != ans_base:
                    base_q = EnvironmentQuery(
                        query_id=f"base_bridge_{len(pairs)}",
                        source=u,
                        target=v,
                        query_type="bridge_cut",
                        question_text=f"Does {u} {env.relation_name} {v}?",
                        expected_answer=ans_base,
                    )
                    pert_q = EnvironmentQuery(
                        query_id=f"pert_bridge_{len(pairs)}",
                        source=u,
                        target=v,
                        query_type="bridge_cut",
                        question_text=f"Without bridge {u}->{v}, does {u} {env.relation_name} {v}?",
                        expected_answer=ans_pert,
                    )
                    pairs.append(MinimalPairControl(
                        pair_id=f"bridge_cut_{len(pairs)}",
                        base_query=base_q,
                        perturbed_query=pert_q,
                        perturbation_type="BRIDGE_DELETION",
                    ))
                    if len(pairs) >= max_pairs // 2:
                        break

        # 2. Directional Reversals: u -> v vs v -> u
        for u in entities:
            for v in entities:
                if u != v:
                    ans_uv = env.oracle_evaluate_query(u, v)
                    ans_vu = env.oracle_evaluate_query(v, u)
                    if ans_uv != ans_vu:
                        q_uv = EnvironmentQuery(
                            query_id=f"dir_base_{len(pairs)}",
                            source=u,
                            target=v,
                            query_type="directional",
                            question_text=f"Does {u} {env.relation_name} {v}?",
                            expected_answer=ans_uv,
                        )
                        q_vu = EnvironmentQuery(
                            query_id=f"dir_pert_{len(pairs)}",
                            source=v,
                            target=u,
                            query_type="directional",
                            question_text=f"Does {v} {env.relation_name} {u}?",
                            expected_answer=ans_vu,
                        )
                        pairs.append(MinimalPairControl(
                            pair_id=f"dir_rev_{len(pairs)}",
                            base_query=q_uv,
                            perturbed_query=q_vu,
                            perturbation_type="DIRECTIONAL_REVERSAL",
                        ))
                        if len(pairs) >= max_pairs:
                            break
            if len(pairs) >= max_pairs:
                break

        # 3. Two-hop bridge cuts
        if len(pairs) < max_pairs:
            for u in entities:
                for v in entities:
                    for w in entities:
                        if len({u, v, w}) == 3:
                            if (u, v) in edge_set and (v, w) in edge_set:
                                ans_base = env.oracle_evaluate_query(u, w)
                                pert_edges = [e for e in base_edges if e != (v, w)]
                                ans_pert = env.hypothesis.evaluate_query(
                                    source=u,
                                    target=w,
                                    edges=pert_edges,
                                    entity_levels=env.entity_levels,
                                    blocked_nodes=env.blocked_nodes,
                                    negated_edges=env.negated_edges,
                                )
                                if ans_base != ans_pert:
                                    b_q = EnvironmentQuery(
                                        query_id=f"two_hop_base_{len(pairs)}",
                                        source=u,
                                        target=w,
                                        query_type="two_hop_cut",
                                        question_text=f"Does {u} {env.relation_name} {w}?",
                                        expected_answer=ans_base,
                                    )
                                    p_q = EnvironmentQuery(
                                        query_id=f"two_hop_pert_{len(pairs)}",
                                        source=u,
                                        target=w,
                                        query_type="two_hop_cut",
                                        question_text=f"Without {v}->{w}, does {u} {env.relation_name} {w}?",
                                        expected_answer=ans_pert,
                                    )
                                    pairs.append(MinimalPairControl(
                                        pair_id=f"two_hop_cut_{len(pairs)}",
                                        base_query=b_q,
                                        perturbed_query=p_q,
                                        perturbation_type="TWO_HOP_BRIDGE_CUT",
                                    ))
                                    if len(pairs) >= max_pairs:
                                        break

        return pairs

    @classmethod
    def evaluate_paired_correctness(
        cls,
        predict_fn: Any,
        minimal_pairs: list[MinimalPairControl],
    ) -> MinimalPairEvaluationResult:
        """Evaluate whether predict_fn outputs the exact ground-truth answer on BOTH sides of minimal pairs."""
        if not minimal_pairs:
            return MinimalPairEvaluationResult(
                total_pairs=0,
                both_correct_count=0,
                paired_correctness_rate=0.0,
                base_accuracy=0.0,
                pert_accuracy=0.0,
                predicted_distribution={"YES": 0, "NO": 0, "UNKNOWN": 0},
            )

        both_correct = 0
        base_correct = 0
        pert_correct = 0
        dist = {"YES": 0, "NO": 0, "UNKNOWN": 0}
        failures: dict[str, int] = {
            "base_wrong_pert_right": 0,
            "base_right_pert_wrong": 0,
            "both_wrong": 0,
        }

        for pair in minimal_pairs:
            pred_base = predict_fn(pair.base_query.question_text)
            pred_pert = predict_fn(pair.perturbed_query.question_text)

            dist[pred_base] = dist.get(pred_base, 0) + 1
            dist[pred_pert] = dist.get(pred_pert, 0) + 1

            b_ok = (pred_base == pair.base_query.expected_answer)
            p_ok = (pred_pert == pair.perturbed_query.expected_answer)

            if b_ok:
                base_correct += 1
            if p_ok:
                pert_correct += 1

            if b_ok and p_ok:
                both_correct += 1
            elif b_ok and not p_ok:
                failures["base_right_pert_wrong"] += 1
            elif not b_ok and p_ok:
                failures["base_wrong_pert_right"] += 1
            else:
                failures["both_wrong"] += 1

        total = len(minimal_pairs)
        return MinimalPairEvaluationResult(
            total_pairs=total,
            both_correct_count=both_correct,
            paired_correctness_rate=float(both_correct / total),
            base_accuracy=float(base_correct / total),
            pert_accuracy=float(pert_correct / total),
            predicted_distribution=dist,
            failure_breakdown=failures,
        )
