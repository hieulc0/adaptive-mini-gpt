"""Phase 2: Formal Three-Valued Observation Semantics & Dataset Validator."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from adaptive_mini_gpt.guided_experiment.environment import (
    AutonomousEnvironment,
    EnvironmentQuery,
    MinimalPairControl,
    RuleHypothesis,
)

logger = logging.getLogger(__name__)\


class TriValuedState(str, Enum):
    """Rigorous three-valued logical states under open-world epistemic semantics."""
    YES = "YES"          # Supported by observable path or rule
    NO = "NO"            # Refuted by structural constraint or explicit negative edge
    UNKNOWN = "UNKNOWN"  # Indeterminate; insufficient evidence to prove or refute


@dataclass
class ObservableEvidenceContext:
    """Explicitly represents the evidence available to the learner at inference time."""
    observed_edges: list[tuple[str, str]]
    negated_edges: list[tuple[str, str]]
    blocked_nodes: list[str]
    unobserved_entities: list[str]
    domain_rule_name: str
    is_open_world: bool

    def to_evidence_prefix(self, max_edges: int = 5) -> str:
        """Construct compact, non-leaking evidence string from observable graph facts."""
        facts = []
        for u, v in self.observed_edges[:max_edges]:
            facts.append(f"{u}->{v}")
        if self.negated_edges:
            for u, v in self.negated_edges[:2]:
                facts.append(f"NOT({u}->{v})")
        if self.blocked_nodes:
            facts.append(f"BLOCK({','.join(self.blocked_nodes[:2])})")
        if self.unobserved_entities:
            facts.append(f"UNOBS({','.join(self.unobserved_entities[:2])})")

        return f"Evidence: [{', '.join(facts)}]. " if facts else ""


@dataclass
class DatasetValidationReport:
    """Audit report from dataset validator asserting information identifiability."""
    total_queries: int
    class_counts: dict[str, int]
    identifiability_score: float
    conflicts_detected: list[dict[str, str]]
    validation_passed: bool
    diagnostics: str


class ThreeValuedDatasetValidator:
    """Verifies that all query inputs are identifiable, sound, and non-conflicting."""

    @classmethod
    def validate_query_set(
        cls,
        queries: list[EnvironmentQuery],
        dataset_name: str = "dataset",
    ) -> DatasetValidationReport:
        """Audit query set for identical input texts assigned conflicting labels."""
        total = len(queries)
        if total == 0:
            return DatasetValidationReport(
                total_queries=0,
                class_counts={"NO": 0, "YES": 0, "UNKNOWN": 0},
                identifiability_score=1.0,
                conflicts_detected=[],
                validation_passed=True,
                diagnostics="Empty query set.",
            )

        text_to_labels: dict[str, set[str]] = {}
        class_counts = {"NO": 0, "YES": 0, "UNKNOWN": 0}

        for q in queries:
            if q.expected_answer in class_counts:
                class_counts[q.expected_answer] += 1
            if q.question_text not in text_to_labels:
                text_to_labels[q.question_text] = set()
            text_to_labels[q.question_text].add(q.expected_answer)

        conflicts: list[dict[str, str]] = []
        for text, labels in text_to_labels.items():
            if len(labels) > 1:
                conflicts.append({
                    "question_text": text,
                    "conflicting_labels": ", ".join(sorted(labels)),
                })

        num_unique_texts = len(text_to_labels)
        identifiability = float(num_unique_texts / total) if total > 0 else 1.0
        passed = (len(conflicts) == 0)

        diag = (
            f"Dataset '{dataset_name}': {total} queries ({class_counts}). "
            f"Unique inputs: {num_unique_texts}/{total} (Identifiability: {identifiability:.4f}). "
            f"Conflicts: {len(conflicts)}."
        )

        return DatasetValidationReport(
            total_queries=total,
            class_counts=class_counts,
            identifiability_score=identifiability,
            conflicts_detected=conflicts,
            validation_passed=passed,
            diagnostics=diag,
        )


class GroundedThreeValuedEnvironment:
    """Wraps an environment with explicit evidence semantics, eliminating identifiability flaws."""

    def __init__(
        self,
        base_env: AutonomousEnvironment,
        use_evidence_prefix: bool = False,
    ) -> None:
        self.base_env = base_env
        self.use_evidence_prefix = use_evidence_prefix
        self.env_id = base_env.env_id
        self.domain_name = base_env.domain_name
        self.relation_name = base_env.relation_name
        self.entities = list(base_env.entities)
        self.base_edges = list(base_env.base_edges)
        self.true_hypothesis = base_env.true_hypothesis

        # Evidence context
        unobserved = [f"UnobservedEntity{i}" for i in range(8)]
        self.evidence_context = ObservableEvidenceContext(
            observed_edges=list(base_env.base_edges),
            negated_edges=list(base_env.negated_edges),
            blocked_nodes=list(base_env.blocked_nodes),
            unobserved_entities=unobserved,
            domain_rule_name=getattr(base_env.true_hypothesis, "hypothesis_id", "UNKNOWN_RULE"),
            is_open_world=getattr(base_env.true_hypothesis, "is_open_world", False),
        )

        # Build clean, sound queries
        self.held_out_test_queries = self._build_sound_test_queries()
        self.minimal_pairs = self._build_sound_minimal_pairs()

    def _format_query_text(self, u: str, v: str, evidence_note: str | None = None) -> str:
        """Format query text optionally prepending observable evidence."""
        core = f"Does {u} {self.relation_name} {v}?"
        if not self.use_evidence_prefix:
            return core
        prefix = self.evidence_context.to_evidence_prefix()
        note = f" Note: {evidence_note}." if evidence_note else ""
        return f"{prefix}{core}{note}"

    def _build_sound_test_queries(self) -> list[EnvironmentQuery]:
        """Construct non-conflicting held-out queries (8 YES, 8 NO, 8 UNKNOWN)."""
        yes_queries: list[EnvironmentQuery] = []
        no_queries: list[EnvironmentQuery] = []
        unk_queries: list[EnvironmentQuery] = []

        used_pairs: set[tuple[str, str]] = set()
        all_entities = self.entities

        # 1. Reachable YES pairs
        for u, v in self.base_edges:
            if len(yes_queries) < 8 and (u, v) not in used_pairs:
                ans = self.evaluate_ground_truth(u, v)
                if ans == "YES":
                    q_text = self._format_query_text(u, v, evidence_note=f"direct link {u}->{v}")
                    yes_queries.append(
                        EnvironmentQuery(
                            query_id=f"test_yes_{u}_{v}",
                            source=u,
                            target=v,
                            query_type="OBSERVED_POSITIVE_EDGE",
                            question_text=q_text,
                            expected_answer="YES",
                        )
                    )
                    used_pairs.add((u, v))

        for i in range(len(all_entities)):
            for j in range(len(all_entities)):
                if i != j and len(yes_queries) < 8:
                    u, v = all_entities[i], all_entities[j]
                    if (u, v) not in used_pairs:
                        ans = self.evaluate_ground_truth(u, v)
                        if ans == "YES":
                            q_text = self._format_query_text(u, v, evidence_note=f"path {u}->{v}")
                            yes_queries.append(
                                EnvironmentQuery(
                                    query_id=f"test_trans_yes_{u}_{v}",
                                    source=u,
                                    target=v,
                                    query_type="TRANSITIVE_PATH_REACHABLE",
                                    question_text=q_text,
                                    expected_answer="YES",
                                )
                            )
                            used_pairs.add((u, v))

        # 2. Refuted NO pairs
        for i in range(len(all_entities)):
            for j in range(len(all_entities)):
                if i != j and len(no_queries) < 8:
                    u, v = all_entities[i], all_entities[j]
                    if (u, v) not in used_pairs:
                        ans = self.evaluate_ground_truth(u, v)
                        if ans == "NO":
                            q_text = self._format_query_text(u, v, evidence_note=f"refuted {u}->{v}")
                            no_queries.append(
                                EnvironmentQuery(
                                    query_id=f"test_no_{u}_{v}",
                                    source=u,
                                    target=v,
                                    query_type="STRUCTURAL_NO",
                                    question_text=q_text,
                                    expected_answer="NO",
                                )
                            )
                            used_pairs.add((u, v))

        # 3. Indeterminate UNKNOWN pairs
        # First check if any observed pairs have ground truth UNKNOWN (e.g. open world incomplete)
        for i in range(len(all_entities)):
            for j in range(len(all_entities)):
                if i != j and len(unk_queries) < 8:
                    u, v = all_entities[i], all_entities[j]
                    if (u, v) not in used_pairs:
                        ans = self.evaluate_ground_truth(u, v)
                        if ans == "UNKNOWN":
                            q_text = self._format_query_text(u, v, evidence_note=f"indeterminate {u}->{v}")
                            unk_queries.append(
                                EnvironmentQuery(
                                    query_id=f"test_unk_internal_{u}_{v}",
                                    source=u,
                                    target=v,
                                    query_type="INSUFFICIENT_EVIDENCE_UNKNOWN",
                                    question_text=q_text,
                                    expected_answer="UNKNOWN",
                                )
                            )
                            used_pairs.add((u, v))

        # Unobserved external entities outside the observed graph
        ext_nodes = [f"UnobservedEntity{i}" for i in range(8)]
        for i, ext in enumerate(ext_nodes):
            tgt = all_entities[i % len(all_entities)]
            if (ext, tgt) not in used_pairs and len(unk_queries) < 8:
                q_text = self._format_query_text(ext, tgt, evidence_note=f"entity {ext} unobserved")
                unk_queries.append(
                    EnvironmentQuery(
                        query_id=f"test_unk_ext_{i}",
                        source=ext,
                        target=tgt,
                        query_type="INSUFFICIENT_EVIDENCE_UNKNOWN",
                        question_text=q_text,
                        expected_answer="UNKNOWN",
                    )
                )
                used_pairs.add((ext, tgt))

        assert len(yes_queries) == 8, f"Env {self.env_id} only yielded {len(yes_queries)} YES queries (expected 8)"
        assert len(no_queries) == 8, f"Env {self.env_id} only yielded {len(no_queries)} NO queries (expected 8)"
        assert len(unk_queries) == 8, f"Env {self.env_id} only yielded {len(unk_queries)} UNKNOWN queries (expected 8)"

        test_pool = yes_queries + no_queries + unk_queries
        report = ThreeValuedDatasetValidator.validate_query_set(test_pool, f"test_pool_{self.env_id}")
        assert report.validation_passed, f"Validation failed: {report.diagnostics}"
        return test_pool

    def _build_sound_minimal_pairs(self) -> list[MinimalPairControl]:
        """Construct balanced tri-valued minimal pairs."""
        pairs: list[MinimalPairControl] = []
        if len(self.base_edges) > 0:
            u, v = self.base_edges[0]
            q_base = EnvironmentQuery(
                query_id="mp1_base",
                source=u,
                target=v,
                query_type="DIRECT_FORWARD",
                question_text=self._format_query_text(u, v, "observed edge"),
                expected_answer="YES",
            )
            q_pert = EnvironmentQuery(
                query_id="mp1_pert",
                source=v,
                target=u,
                query_type="DIRECTIONAL_REVERSAL",
                question_text=self._format_query_text(v, u, "refuted reversal"),
                expected_answer="NO",
            )
            pairs.append(MinimalPairControl("mp_dir_reversal", q_base, q_pert, "DIRECTIONAL_REVERSAL"))

        if len(self.base_edges) > 1:
            u_src = self.base_edges[0][0]
            v_tgt = self.base_edges[-1][1]
            q_base2 = EnvironmentQuery(
                query_id="mp2_base",
                source=u_src,
                target=v_tgt,
                query_type="MULTI_HOP_CHAIN",
                question_text=self._format_query_text(u_src, v_tgt, "multi-hop reachability"),
                expected_answer="YES",
            )
            ext_u = f"UnobservedEntity0"
            q_pert2 = EnvironmentQuery(
                query_id="mp2_pert",
                source=u_src,
                target=ext_u,
                query_type="UNOBSERVED_NODE",
                question_text=self._format_query_text(u_src, ext_u, "unobserved target"),
                expected_answer="UNKNOWN",
            )
            pairs.append(MinimalPairControl("mp_unobserved_contrast", q_base2, q_pert2, "EVIDENCE_INSUFFICIENCY"))

        return pairs

    def evaluate_ground_truth(self, source: str, target: str) -> str:
        """Ground truth reachability query."""
        if "Unobserved" in source or "Unobserved" in target or "ExtUnknown" in source or "ExtUnknown" in target:
            return "UNKNOWN"
        return self.base_env.evaluate_ground_truth(source, target)
