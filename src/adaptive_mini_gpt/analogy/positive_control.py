"""Phase 2: Positive Controls for Episodic Memory and Structural Analogy Retrieval.

Constructs three independent memory scenarios:
  Scenario A: Same task family with different entities (e.g. containment hierarchy with novel entities).
  Scenario B: Different vocabulary but structurally equivalent relational rules (cross-domain structural analogy).
  Scenario C: Structurally incompatible or contradictory memories (distractor & contradiction rejection).

Compares 5 retrieval methods:
  1. No memory (baseline).
  2. Lexical retrieval (keyword / token overlap).
  3. Simple structure-aware retrieval (canonical graph signature & relational property match).
  4. Full-memory baseline (uncurated candidate dump up to budget).
  5. Recency baseline (most recently added experiences).

Measures:
  - Precision (fraction of retrieved memories that are structurally valid analogs)
  - Recall (fraction of relevant analogs in memory retrieved)
  - Correct rejection rate (rate of rejecting distractors/contradictions)
  - End-to-end task accuracy (verdict correctness when provided the retrieved context)
"""
from __future__ import annotations

import collections
import copy
import logging
from dataclasses import dataclass, field
from typing import Any

from adaptive_mini_gpt.analogy.structural_schema import (
    CanonicalGraphSignature,
    DeterministicStructuralMatcher,
    RelationProperties,
    StructuralAnalogyExperience,
)
from adaptive_mini_gpt.meta.benchmark_suite import MetaTask

logger = logging.getLogger("m44_positive_control")


@dataclass
class ScenarioRetrievalMetric:
    scenario_name: str
    method_name: str
    total_queries: int
    precision: float
    recall: float
    correct_rejection_rate: float
    mean_retrieved_count: float
    task_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "method_name": self.method_name,
            "total_queries": self.total_queries,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "correct_rejection_rate": round(self.correct_rejection_rate, 4),
            "mean_retrieved_count": round(self.mean_retrieved_count, 2),
            "task_accuracy": round(self.task_accuracy, 4),
        }


@dataclass
class PositiveControlSuiteReport:
    scenario_a_same_family: dict[str, ScenarioRetrievalMetric]
    scenario_b_cross_domain_analogy: dict[str, ScenarioRetrievalMetric]
    scenario_c_distractor_rejection: dict[str, ScenarioRetrievalMetric]
    overall_structure_aware_advantage: float
    overall_summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_a_same_family": {k: v.to_dict() for k, v in self.scenario_a_same_family.items()},
            "scenario_b_cross_domain_analogy": {k: v.to_dict() for k, v in self.scenario_b_cross_domain_analogy.items()},
            "scenario_c_distractor_rejection": {k: v.to_dict() for k, v in self.scenario_c_distractor_rejection.items()},
            "overall_structure_aware_advantage": round(self.overall_structure_aware_advantage, 4),
            "overall_summary": self.overall_summary,
        }


class MemoryPositiveControlEvaluator:
    """Evaluates retrieval policies and downstream task execution on controlled scenarios."""

    def __init__(self, max_exemplars: int = 2) -> None:
        self.max_exemplars = max_exemplars

    @staticmethod
    def _extract_lexical_tokens(text: str) -> set[str]:
        words = [w.strip("?,.:;![]").lower() for w in text.split()]
        stopwords = {"does", "is", "it", "verified", "that", "the", "a", "an", "of", "to", "and", "or", "in", "pattern", "rule", "ex"}
        return {w for w in words if w and w not in stopwords}

    @classmethod
    def _lexical_retrieve(
        cls,
        query_task: MetaTask,
        memory_bank: list[StructuralAnalogyExperience],
        max_exemplars: int,
    ) -> list[StructuralAnalogyExperience]:
        q_text = f"{' '.join(query_task.premises)} {query_task.question}"
        q_tokens = cls._extract_lexical_tokens(q_text)
        scored: list[tuple[int, StructuralAnalogyExperience]] = []
        for exp in memory_bank:
            exp_text = f"{' '.join(exp.exemplar_premises)} {exp.exemplar_question}"
            exp_tokens = cls._extract_lexical_tokens(exp_text)
            overlap = len(q_tokens.intersection(exp_tokens))
            if overlap > 0:
                scored.append((overlap, exp))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:max_exemplars]]

    @staticmethod
    def _structure_aware_retrieve(
        query_task: MetaTask,
        query_props: RelationProperties,
        memory_bank: list[StructuralAnalogyExperience],
        max_exemplars: int,
        threshold: float = 0.5,
    ) -> list[StructuralAnalogyExperience]:
        scored: list[tuple[float, StructuralAnalogyExperience]] = []
        for exp in memory_bank:
            score = DeterministicStructuralMatcher.match_analogy(query_task, query_props, exp)
            if score >= threshold:
                scored.append((score, exp))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:max_exemplars]]

    @staticmethod
    def _full_memory_retrieve(
        memory_bank: list[StructuralAnalogyExperience],
        max_exemplars: int,
    ) -> list[StructuralAnalogyExperience]:
        return memory_bank[:max_exemplars]

    @staticmethod
    def _recency_retrieve(
        memory_bank: list[StructuralAnalogyExperience],
        max_exemplars: int,
    ) -> list[StructuralAnalogyExperience]:
        return list(reversed(memory_bank))[:max_exemplars]

    @staticmethod
    def _solve_task_with_exemplars(
        task: MetaTask,
        props: RelationProperties,
        exemplars: list[StructuralAnalogyExperience],
    ) -> str:
        """Determines verdict using structural logic guided by analogical exemplars if present."""
        query_sig = DeterministicStructuralMatcher.extract_signature(task, props)

        for exp in exemplars:
            if exp.canonical_signature.pattern_type == query_sig.pattern_type:
                if "CHAIN" in query_sig.pattern_type and props.is_transitive:
                    return exp.verified_verdict
                if query_sig.pattern_type == "ASYMMETRIC_REVERSAL" and not props.is_symmetric:
                    return exp.verified_verdict
                if query_sig.pattern_type == "UNGROUNDED":
                    return "UNKNOWN"

        if query_sig.pattern_type == "UNGROUNDED" or query_sig.pattern_type == "CONTRADICTORY":
            return "UNKNOWN"
        if "CHAIN" in query_sig.pattern_type:
            return "YES" if props.is_transitive else "NO"
        if query_sig.pattern_type == "ASYMMETRIC_REVERSAL":
            return "YES" if props.is_symmetric else "NO"
        return "NO"

    def evaluate_scenario(
        self,
        scenario_name: str,
        queries: list[tuple[MetaTask, RelationProperties, set[str]]],
        memory_bank: list[StructuralAnalogyExperience],
        distractor_exp_ids: set[str],
    ) -> dict[str, ScenarioRetrievalMetric]:
        methods = ["no_memory", "lexical", "structure_aware", "full_memory", "recency"]
        results: dict[str, ScenarioRetrievalMetric] = {}

        for method in methods:
            total_relevant_retrieved = 0
            total_retrieved = 0
            total_relevant_in_bank = sum(len(q[2]) for q in queries)
            total_distractors_rejected = 0
            total_distractors_present = len(distractor_exp_ids) * len(queries)
            correct_task_answers = 0

            for task, props, relevant_ids in queries:
                if method == "no_memory":
                    retrieved = []
                elif method == "lexical":
                    retrieved = self._lexical_retrieve(task, memory_bank, self.max_exemplars)
                elif method == "structure_aware":
                    retrieved = self._structure_aware_retrieve(task, props, memory_bank, self.max_exemplars)
                elif method == "full_memory":
                    retrieved = self._full_memory_retrieve(memory_bank, self.max_exemplars)
                elif method == "recency":
                    retrieved = self._recency_retrieve(memory_bank, self.max_exemplars)
                else:
                    retrieved = []

                ret_ids = {e.experience_id for e in retrieved}
                total_retrieved += len(ret_ids)
                total_relevant_retrieved += len(ret_ids.intersection(relevant_ids))

                rejected_distractors = len(distractor_exp_ids.difference(ret_ids))
                total_distractors_rejected += rejected_distractors

                pred_verdict = self._solve_task_with_exemplars(task, props, retrieved)
                if pred_verdict == task.expected_verdict:
                    correct_task_answers += 1

            precision = (
                total_relevant_retrieved / max(1, total_retrieved)
                if total_retrieved > 0
                else (1.0 if total_relevant_in_bank == 0 else 0.0)
            )
            recall = (
                total_relevant_retrieved / max(1, total_relevant_in_bank)
                if total_relevant_in_bank > 0
                else 1.0
            )
            rejection_rate = (
                total_distractors_rejected / max(1, total_distractors_present)
                if total_distractors_present > 0
                else 1.0
            )
            task_acc = correct_task_answers / max(1, len(queries))
            mean_ret = total_retrieved / max(1, len(queries))

            results[method] = ScenarioRetrievalMetric(
                scenario_name=scenario_name,
                method_name=method,
                total_queries=len(queries),
                precision=precision,
                recall=recall,
                correct_rejection_rate=rejection_rate,
                mean_retrieved_count=mean_ret,
                task_accuracy=task_acc,
            )

        return results

    def run_all_positive_controls(self) -> PositiveControlSuiteReport:
        # Scenario A: Same task family with different entities
        rel_containment = RelationProperties("is_contained_in", is_transitive=True, is_symmetric=False)
        exp_a1 = StructuralAnalogyExperience(
            experience_id="exp_a1",
            original_family="spatial_containment",
            relation_name="is_contained_in",
            relation_properties=rel_containment,
            canonical_signature=CanonicalGraphSignature("CHAIN_2HOP", 3, 2, 2, False, True),
            abstract_rule="Transitive 2-hop containment",
            exemplar_premises=["BoxA is_contained_in BoxB", "BoxB is_contained_in BoxC"],
            exemplar_question="Does BoxA is_contained_in BoxC?",
            verified_verdict="YES",
        )
        exp_a2 = StructuralAnalogyExperience(
            experience_id="exp_a2",
            original_family="spatial_containment",
            relation_name="is_contained_in",
            relation_properties=rel_containment,
            canonical_signature=CanonicalGraphSignature("ASYMMETRIC_REVERSAL", 2, 1, 1, True, True),
            abstract_rule="Asymmetric containment reversal entails false",
            exemplar_premises=["ChestX is_contained_in RoomY"],
            exemplar_question="Does RoomY is_contained_in ChestX?",
            verified_verdict="NO",
        )
        bank_a = [exp_a1, exp_a2]

        queries_a = [
            (
                MetaTask("qa_1", "containment_test", "test", ["VesselAlpha is_contained_in VesselBeta", "VesselBeta is_contained_in VesselGamma"], "Does VesselAlpha is_contained_in VesselGamma?", "YES", "two_hop", ["VesselAlpha", "VesselGamma"]),
                rel_containment,
                {"exp_a1"},
            ),
            (
                MetaTask("qa_2", "containment_test", "test", ["Chamber1 is_contained_in Chamber2"], "Does Chamber2 is_contained_in Chamber1?", "NO", "asymmetric", ["Chamber2", "Chamber1"]),
                rel_containment,
                {"exp_a2"},
            ),
        ]
        res_a = self.evaluate_scenario("Scenario_A_Same_Domain", queries_a, bank_a, distractor_exp_ids=set())

        # Scenario B: Cross-Domain Structural Analogy (0% word overlap)
        rel_reports = RelationProperties("reports_to", is_transitive=True, is_symmetric=False)
        exp_b1 = StructuralAnalogyExperience(
            experience_id="exp_b1",
            original_family="corporate_hierarchy",
            relation_name="reports_to",
            relation_properties=rel_reports,
            canonical_signature=CanonicalGraphSignature("CHAIN_2HOP", 3, 2, 2, False, True),
            abstract_rule="Transitive 2-hop hierarchy",
            exemplar_premises=["JuniorDev reports_to TechLead", "TechLead reports_to EngineeringVP"],
            exemplar_question="Does JuniorDev reports_to EngineeringVP?",
            verified_verdict="YES",
        )
        exp_b2 = StructuralAnalogyExperience(
            experience_id="exp_b2",
            original_family="corporate_hierarchy",
            relation_name="reports_to",
            relation_properties=rel_reports,
            canonical_signature=CanonicalGraphSignature("ASYMMETRIC_REVERSAL", 2, 1, 1, True, True),
            abstract_rule="Asymmetric hierarchy reversal entails false",
            exemplar_premises=["Analyst reports_to Manager"],
            exemplar_question="Does Manager reports_to Analyst?",
            verified_verdict="NO",
        )
        bank_b = [exp_b1, exp_b2]

        rel_metabolic = RelationProperties("catalyzes", is_transitive=True, is_symmetric=False)
        queries_b = [
            (
                MetaTask("qb_1", "metabolic_test", "test", ["EnzymeAlpha catalyzes SubstrateBeta", "SubstrateBeta catalyzes ProductGamma"], "Does EnzymeAlpha catalyzes ProductGamma?", "YES", "two_hop", ["EnzymeAlpha", "ProductGamma"]),
                rel_metabolic,
                {"exp_b1"},
            ),
            (
                MetaTask("qb_2", "metabolic_test", "test", ["ProteinX catalyzes KinaseY"], "Does KinaseY catalyzes ProteinX?", "NO", "asymmetric", ["KinaseY", "ProteinX"]),
                rel_metabolic,
                {"exp_b2"},
            ),
        ]
        res_b = self.evaluate_scenario("Scenario_B_Cross_Domain_Analogy", queries_b, bank_b, distractor_exp_ids=set())

        # Scenario C: Incompatible or contradictory distractors
        exp_c_valid = StructuralAnalogyExperience(
            experience_id="exp_c_valid",
            original_family="kinship_ancestor",
            relation_name="ancestor_of",
            relation_properties=RelationProperties("ancestor_of", is_transitive=True, is_symmetric=False),
            canonical_signature=CanonicalGraphSignature("CHAIN_2HOP", 3, 2, 2, False, True),
            abstract_rule="Transitive 2-hop ancestry",
            exemplar_premises=["ParentA ancestor_of ChildB", "ChildB ancestor_of GrandchildC"],
            exemplar_question="Does ParentA ancestor_of GrandchildC?",
            verified_verdict="YES",
        )
        exp_c_distractor1 = StructuralAnalogyExperience(
            experience_id="exp_c_distractor1",
            original_family="unrelated_physics",
            relation_name="collided_with",
            relation_properties=RelationProperties("collided_with", is_transitive=False, is_symmetric=True),
            canonical_signature=CanonicalGraphSignature("CHAIN_1HOP", 2, 1, 1, False, True),
            abstract_rule="Non-transitive symmetric collision",
            exemplar_premises=["Proton1 collided_with Neutron2"],
            exemplar_question="Does Neutron2 collided_with Proton1?",
            verified_verdict="YES",
        )
        exp_c_distractor2 = StructuralAnalogyExperience(
            experience_id="exp_c_distractor2",
            original_family="contradictory_loop",
            relation_name="causes",
            relation_properties=RelationProperties("causes", is_transitive=True, is_symmetric=False),
            canonical_signature=CanonicalGraphSignature("CONTRADICTORY", 2, 2, 0, False, True),
            abstract_rule="Inconsistent circular causality",
            exemplar_premises=["StateA causes StateA"],
            exemplar_question="Does StateA causes StateA?",
            verified_verdict="UNKNOWN",
        )
        bank_c = [exp_c_valid, exp_c_distractor1, exp_c_distractor2]
        distractors_c = {"exp_c_distractor1", "exp_c_distractor2"}

        rel_quantum = RelationProperties("entangled_with", is_transitive=True, is_symmetric=True)
        queries_c = [
            (
                MetaTask("qc_1", "quantum_test", "test", ["Qubit0 entangled_with Qubit1"], "Does Qubit1 entangled_with Qubit0?", "YES", "single_hop", ["Qubit1", "Qubit0"]),
                rel_quantum,
                set(),
            ),
        ]
        res_c = self.evaluate_scenario("Scenario_C_Distractor_Rejection", queries_c, bank_c, distractor_exp_ids=distractors_c)

        struct_rec_b = res_b["structure_aware"].recall
        lex_rec_b = res_b["lexical"].recall
        struct_advantage = struct_rec_b - lex_rec_b

        summary = (
            f"Scenario A (Same Domain): Lexical recall={res_a['lexical'].recall*100:.1f}%, Structure recall={res_a['structure_aware'].recall*100:.1f}%. "
            f"Scenario B (Cross-Domain Analogy): Lexical recall={res_b['lexical'].recall*100:.1f}%, Structure recall={res_b['structure_aware'].recall*100:.1f}%. "
            f"Scenario C (Noise Rejection): Structure rejection rate={res_c['structure_aware'].correct_rejection_rate*100:.1f}%."
        )

        return PositiveControlSuiteReport(
            scenario_a_same_family=res_a,
            scenario_b_cross_domain_analogy=res_b,
            scenario_c_distractor_rejection=res_c,
            overall_structure_aware_advantage=struct_advantage,
            overall_summary=summary,
        )
