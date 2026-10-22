"""Phase 6: Selective Consolidation, Archival, and Memory Lifecycle Benchmark.

Implements and evaluates reversible memory lifecycle management:
  - Active Working Memory vs Long-Term Archive Pool.
  - Consolidating redundant experiences while preserving complete provenance lineages.
  - Archiving low-utility / low-recency records when capacity limits are reached.
  - Restoring archived memories when relevant to an unfamiliar domain.
  - Correcting superseded or contradicted claims.
  - Controlled 3-Way Policy Benchmark under Matched Storage Budget (Selective vs Full vs Recency).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from adaptive_mini_gpt.memory.episodic_memory import EpisodicMemoryBank, LearningExperience
from adaptive_mini_gpt.memory.retrieval import MemoryRetriever

logger = logging.getLogger("m43_lifecycle")


@dataclass
class LifecycleBenchmarkReport:
    initial_active_count: int
    initial_archived_count: int
    consolidated_redundant_count: int
    superseded_corrected_count: int
    post_consolidation_active_count: int
    post_consolidation_archived_count: int
    provenance_integrity_verified: bool
    reversibility_verified: bool
    selective_policy_precision: float
    full_policy_precision: float
    recency_policy_precision: float
    selective_contradiction_rejection: float
    full_contradiction_rejection: float
    recency_contradiction_rejection: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_active_count": self.initial_active_count,
            "initial_archived_count": self.initial_archived_count,
            "consolidated_redundant_count": self.consolidated_redundant_count,
            "superseded_corrected_count": self.superseded_corrected_count,
            "post_consolidation_active_count": self.post_consolidation_active_count,
            "post_consolidation_archived_count": self.post_consolidation_archived_count,
            "provenance_integrity_verified": self.provenance_integrity_verified,
            "reversibility_verified": self.reversibility_verified,
            "selective_policy_precision": self.selective_policy_precision,
            "full_policy_precision": self.full_policy_precision,
            "recency_policy_precision": self.recency_policy_precision,
            "selective_contradiction_rejection": self.selective_contradiction_rejection,
            "full_contradiction_rejection": self.full_contradiction_rejection,
            "recency_contradiction_rejection": self.recency_contradiction_rejection,
        }


class MemoryLifecycleBenchmark:
    """Evaluates memory lifecycle operations and compares storage selection policies."""

    @staticmethod
    def run_lifecycle_evaluation(
        memory_bank: EpisodicMemoryBank,
        test_queries: list[dict[str, Any]],
    ) -> LifecycleBenchmarkReport:
        init_act = len(memory_bank.get_all_active())
        init_arc = len(memory_bank.get_all_archived())

        # 1. Consolidate redundant experiences
        consolidated = memory_bank.consolidate_redundant_experiences()

        # 2. Test superseding a contradicted memory
        active_list = memory_bank.get_all_active()
        superseded_count = 0
        if active_list:
            target_to_revise = active_list[0]
            new_exp = memory_bank.revise_experience(
                old_experience_id=target_to_revise.experience_id,
                new_verified_answer="NO",
                new_procedure="Revised by oracle: updated relation directionality",
                new_evidence_source="verified_oracle:graph_oracle_v2_recheck",
            )
            superseded_count += 1

        # 3. Verify provenance integrity
        prov_ok = True
        for exp in memory_bank.get_all_active():
            if exp.consolidated_from:
                for prior_id in exp.consolidated_from:
                    archived = memory_bank.inspect_experience(prior_id)
                    if not archived or not archived.is_archived:
                        prov_ok = False

        # 4. Verify reversibility: archive then restore
        reversibility_ok = True
        if memory_bank.get_all_active():
            sample_exp = memory_bank.get_all_active()[-1]
            sid = sample_exp.experience_id
            archived_ok = memory_bank.archive_experience(sid)
            restored_ok = memory_bank.restore_experience(sid)
            reversibility_ok = (archived_ok and restored_ok and sid in memory_bank.active_experiences)

        # 5. Evaluate Retrieval under Matched Capacity: Selective vs Full vs Recency
        retriever = MemoryRetriever(memory_bank, max_exemplars=2)

        sel_correct = 0
        full_correct = 0
        rec_correct = 0
        sel_contra_rej = 0
        full_contra_rej = 0
        rec_contra_rej = 0
        total_q = max(1, len(test_queries))

        for tq in test_queries:
            q = tq["question"]
            p = tq["premises"]
            expected_verdict = tq.get("expected_verdict", "YES")
            is_contradictory = tq.get("is_contradictory", False)

            res_sel = retriever.retrieve_selective(q, p)
            res_full = retriever.retrieve_full_baseline(q, p)
            res_rec = retriever.retrieve_recency_baseline(q, p)

            # Check if retrieved exemplars are accurate & not superseded
            if res_sel and all(not e.is_superseded and e.verified_answer == expected_verdict for e in res_sel):
                sel_correct += 1
            if res_full and all(not e.is_superseded and e.verified_answer == expected_verdict for e in res_full):
                full_correct += 1
            if res_rec and all(not e.is_superseded and e.verified_answer == expected_verdict for e in res_rec):
                rec_correct += 1

            # Check contradiction rejection
            if all(not e.is_superseded for e in res_sel):
                sel_contra_rej += 1
            if all(not e.is_superseded for e in res_full):
                full_contra_rej += 1
            if all(not e.is_superseded for e in res_rec):
                rec_contra_rej += 1

        return LifecycleBenchmarkReport(
            initial_active_count=init_act,
            initial_archived_count=init_arc,
            consolidated_redundant_count=consolidated,
            superseded_corrected_count=superseded_count,
            post_consolidation_active_count=len(memory_bank.get_all_active()),
            post_consolidation_archived_count=len(memory_bank.get_all_archived()),
            provenance_integrity_verified=prov_ok,
            reversibility_verified=reversibility_ok,
            selective_policy_precision=round(sel_correct / total_q, 4),
            full_policy_precision=round(full_correct / total_q, 4),
            recency_policy_precision=round(rec_correct / total_q, 4),
            selective_contradiction_rejection=round(sel_contra_rej / total_q, 4),
            full_contradiction_rejection=round(full_contra_rej / total_q, 4),
            recency_contradiction_rejection=round(rec_contra_rej / total_q, 4),
        )
