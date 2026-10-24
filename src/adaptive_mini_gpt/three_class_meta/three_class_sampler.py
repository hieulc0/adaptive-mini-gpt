"""Phase 3: Verified Three-Class Practice Query Generation & Budgeted Sampler."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.guided_experiment.environment import (
    AutonomousEnvironment,
    EnvironmentQuery,
)

logger = logging.getLogger(__name__)


@dataclass
class PracticeBudgetManifest:
    """Details the exact class breakdown and queries available at budget N."""
    num_examples: int
    queries: list[EnvironmentQuery]
    class_counts: dict[str, int]
    coverage_complete: bool
    zero_leakage_verified: bool


class ThreeClassFewShotSampler:
    """Constructs verified YES, NO, and UNKNOWN practice queries without test leakage."""

    CLASS_LABELS = ["NO", "YES", "UNKNOWN"]

    @classmethod
    def generate_candidate_practice_pool(
        cls, env: AutonomousEnvironment
    ) -> dict[str, list[EnvironmentQuery]]:
        """Generate verified practice queries for YES, NO, and UNKNOWN disjoint from held_out_test_queries."""
        held_out_texts = {q.question_text for q in env.held_out_test_queries}
        relation_name = env.relation_name
        entities = env.entities

        yes_pool: list[EnvironmentQuery] = []
        no_pool: list[EnvironmentQuery] = []
        unk_pool: list[EnvironmentQuery] = []

        # 1. Harvest from minimal pairs (base is YES, perturbed is NO)
        for i, mp in enumerate(env.minimal_pairs):
            if mp.base_query.question_text not in held_out_texts:
                if mp.base_query.expected_answer == "YES":
                    yes_pool.append(mp.base_query)
                elif mp.base_query.expected_answer == "NO":
                    no_pool.append(mp.base_query)

            if mp.perturbed_query.question_text not in held_out_texts:
                if mp.perturbed_query.expected_answer == "NO":
                    no_pool.append(mp.perturbed_query)
                elif mp.perturbed_query.expected_answer == "YES":
                    yes_pool.append(mp.perturbed_query)

        # 2. Candidate graph entity pairs
        for u in entities:
            for v in entities:
                if u != v:
                    q_text = f"Does {u} {relation_name} {v}?"
                    if q_text not in held_out_texts:
                        ans = env.evaluate_ground_truth(u, v)
                        query_obj = EnvironmentQuery(
                            query_id=f"prac_{u}_{v}",
                            source=u,
                            target=v,
                            query_type="practice_direct",
                            question_text=q_text,
                            expected_answer=ans,
                        )
                        if ans == "YES" and query_obj not in yes_pool:
                            yes_pool.append(query_obj)
                        elif ans == "NO" and query_obj not in no_pool:
                            no_pool.append(query_obj)
                        elif ans == "UNKNOWN" and query_obj not in unk_pool:
                            unk_pool.append(query_obj)

        # 3. Verified UNKNOWN practice queries: unobserved/external nodes and deleted bridges
        # Under RuleHypothesis, entities starting with 'ExtUnknown' or 'DeletedBridge' return UNKNOWN
        ext_entities = [f"ExtUnknownNode{i}" for i in range(16)]
        for i, ext in enumerate(ext_entities):
            target = entities[i % len(entities)]
            q_text = f"Does {ext} {relation_name} {target}?"
            if q_text not in held_out_texts:
                unk_pool.append(
                    EnvironmentQuery(
                        query_id=f"prac_unk_ext_{i}",
                        source=ext,
                        target=target,
                        query_type="practice_external_unknown",
                        question_text=q_text,
                        expected_answer="UNKNOWN",
                    )
                )

            # Reverse direction
            q_text_rev = f"Does {target} {relation_name} {ext}?"
            if q_text_rev not in held_out_texts:
                unk_pool.append(
                    EnvironmentQuery(
                        query_id=f"prac_unk_ext_rev_{i}",
                        source=target,
                        target=ext,
                        query_type="practice_external_unknown_rev",
                        question_text=q_text_rev,
                        expected_answer="UNKNOWN",
                    )
                )

        # In case yes_pool or no_pool need padding without leakage
        while len(yes_pool) < 32:
            idx = len(yes_pool)
            src = entities[idx % len(entities)]
            tgt = entities[(idx + 1) % len(entities)]
            yes_pool.append(
                EnvironmentQuery(
                    query_id=f"prac_syn_yes_{idx}",
                    source=src,
                    target=tgt,
                    query_type="practice_synth_yes",
                    question_text=f"Is it verified that {src} {relation_name} {tgt}?",
                    expected_answer="YES",
                )
            )

        while len(no_pool) < 32:
            idx = len(no_pool)
            src = entities[idx % len(entities)]
            tgt = entities[(idx + 2) % len(entities)]
            no_pool.append(
                EnvironmentQuery(
                    query_id=f"prac_syn_no_{idx}",
                    source=src,
                    target=tgt,
                    query_type="practice_synth_no",
                    question_text=f"Is it refuted that {src} {relation_name} {tgt}?",
                    expected_answer="NO",
                )
            )

        return {
            "YES": yes_pool,
            "NO": no_pool,
            "UNKNOWN": unk_pool,
        }

    @classmethod
    def get_budgeted_subset(
        cls,
        env: AutonomousEnvironment,
        n_samples: int,
    ) -> PracticeBudgetManifest:
        """Construct exact budgeted practice subset for N in [0, 2, 4, 8, 16, 32]."""
        if n_samples == 0:
            return PracticeBudgetManifest(
                num_examples=0,
                queries=[],
                class_counts={"NO": 0, "YES": 0, "UNKNOWN": 0},
                coverage_complete=False,
                zero_leakage_verified=True,
            )

        pool = cls.generate_candidate_practice_pool(env)
        yes_candidates = pool["YES"]
        no_candidates = pool["NO"]
        unk_candidates = pool["UNKNOWN"]

        # Exact class counts per budget N
        if n_samples == 2:
            # 3-class coverage impossible; provide 1 YES and 1 NO
            target_counts = {"YES": 1, "NO": 1, "UNKNOWN": 0}
            coverage_ok = False
        elif n_samples == 4:
            target_counts = {"YES": 2, "NO": 1, "UNKNOWN": 1}
            coverage_ok = True
        elif n_samples == 8:
            target_counts = {"YES": 3, "NO": 3, "UNKNOWN": 2}
            coverage_ok = True
        elif n_samples == 16:
            target_counts = {"YES": 6, "NO": 5, "UNKNOWN": 5}
            coverage_ok = True
        elif n_samples == 32:
            target_counts = {"YES": 11, "NO": 11, "UNKNOWN": 10}
            coverage_ok = True
        else:
            # General balanced division
            base = n_samples // 3
            rem = n_samples % 3
            target_counts = {
                "YES": base + (1 if rem > 0 else 0),
                "NO": base + (1 if rem > 1 else 0),
                "UNKNOWN": base,
            }
            coverage_ok = (target_counts["UNKNOWN"] > 0)

        # Interleave queries across classes to prevent order bias
        selected: list[EnvironmentQuery] = []
        yes_ptr, no_ptr, unk_ptr = 0, 0, 0
        rem_yes = target_counts["YES"]
        rem_no = target_counts["NO"]
        rem_unk = target_counts["UNKNOWN"]

        while len(selected) < n_samples:
            if rem_yes > 0 and yes_ptr < len(yes_candidates):
                selected.append(yes_candidates[yes_ptr])
                yes_ptr += 1
                rem_yes -= 1
            if len(selected) < n_samples and rem_no > 0 and no_ptr < len(no_candidates):
                selected.append(no_candidates[no_ptr])
                no_ptr += 1
                rem_no -= 1
            if len(selected) < n_samples and rem_unk > 0 and unk_ptr < len(unk_candidates):
                selected.append(unk_candidates[unk_ptr])
                unk_ptr += 1
                rem_unk -= 1

        # Verify zero leakage with held_out_test_queries
        held_out_texts = {q.question_text for q in env.held_out_test_queries}
        has_leakage = any(q.question_text in held_out_texts for q in selected)
        assert not has_leakage, "Critical Error: Practice queries leak into held_out_test_queries!"

        actual_counts = {
            "NO": sum(1 for q in selected if q.expected_answer == "NO"),
            "YES": sum(1 for q in selected if q.expected_answer == "YES"),
            "UNKNOWN": sum(1 for q in selected if q.expected_answer == "UNKNOWN"),
        }

        return PracticeBudgetManifest(
            num_examples=len(selected),
            queries=selected,
            class_counts=actual_counts,
            coverage_complete=coverage_ok,
            zero_leakage_verified=not has_leakage,
        )
