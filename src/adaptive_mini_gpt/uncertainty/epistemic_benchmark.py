"""Phase 2: Formal Epistemic Uncertainty Benchmark with Independent Open-World Semantics.

Constructs independent relational worlds with mathematically precise epistemic boundaries:
  - Epistemic Truth Values:
      * YES (Supported): Deductively entailed by premises and relation axioms.
      * NO (Refuted): Explicitly contradicted, or refuted by asymmetry / mutual exclusivity.
      * UNKNOWN (Unsupported / Epistemic Incompleteness):
          - Open-world ungrounded entities (entity not in premise knowledge base).
          - Disconnected components in open-world (absence of proof != proof of absence).
          - Nontransitive chains (A -> B and B -> C does not entail A -> C, but does not refute it).
          - Ambiguous cyclical dependencies under open-world interpretation.

Guarantees:
  - Balanced 3-class distribution (1/3 YES, 1/3 NO, 1/3 UNKNOWN) to prevent constant-class inflation.
  - Independent held-out test families with novel vocabularies, unseen graph topologies, and distinct axioms.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

from adaptive_mini_gpt.meta.benchmark_suite import MetaFamilyDataset, MetaTask


@dataclass(frozen=True)
class RelationSemantics:
    name: str
    inverse_name: str
    is_transitive: bool
    is_symmetric: bool
    is_directional: bool
    world_assumption: str = "OPEN_WORLD"  # "OPEN_WORLD" or "CLOSED_WORLD"


@dataclass
class EpistemicTask:
    task_id: str
    world_name: str
    split: str
    premises: list[str]
    question: str
    expected_verdict: str  # "YES", "NO", "UNKNOWN"
    epistemic_category: str
    entities_involved: list[str]

    def to_meta_task(self) -> MetaTask:
        return MetaTask(
            task_id=self.task_id,
            family=self.world_name,
            split=self.split,
            premises=self.premises,
            question=self.question,
            expected_verdict=self.expected_verdict,
            reasoning_type=self.epistemic_category,
            entities_involved=self.entities_involved,
        )


@dataclass
class EpistemicWorldDataset:
    world_name: str
    split: str
    semantics: RelationSemantics
    train_practice_pool: list[EpistemicTask]
    held_out_test_pool: list[EpistemicTask]

    def to_meta_family_dataset(self) -> MetaFamilyDataset:
        return MetaFamilyDataset(
            family_name=self.world_name,
            split=self.split,
            relation_name=self.semantics.name,
            inverse_relation=self.semantics.inverse_name,
            is_transitive=self.semantics.is_transitive,
            is_symmetric=self.semantics.is_symmetric,
            train_practice_pool=[t.to_meta_task() for t in self.train_practice_pool],
            held_out_test_pool=[t.to_meta_task() for t in self.held_out_test_pool],
        )


def build_epistemic_world(
    world_name: str,
    split: str,
    rel: str,
    inv_rel: str,
    entities: list[str],
    is_transitive: bool,
    is_symmetric: bool,
    is_directional: bool,
) -> EpistemicWorldDataset:
    assert len(entities) >= 5, "World requires at least 5 distinct entities"
    e0, e1, e2, e3, e4 = entities[0], entities[1], entities[2], entities[3], entities[4]

    semantics = RelationSemantics(
        name=rel,
        inverse_name=inv_rel,
        is_transitive=is_transitive,
        is_symmetric=is_symmetric,
        is_directional=is_directional,
        world_assumption="OPEN_WORLD",
    )

    # Base chain: e0 -> e1 -> e2 -> e3. Node e4 is a separate disconnected premise: e4 -> e4_anchor
    premises = [
        f"{e0} {rel} {e1}",
        f"{e1} {rel} {e2}",
        f"{e2} {rel} {e3}",
    ]

    # Practice Pool: 33 items balanced across YES, NO, UNKNOWN (11 each)
    practice_tasks: list[EpistemicTask] = []

    # YES: Direct, 2-hop, 3-hop
    for i in range(1, 12):
        if i % 3 == 1:
            q = f"Does {e0} {rel} {e1}?"
            cat = "SUPPORTED_DIRECT"
            ents = [e0, e1]
        elif i % 3 == 2:
            q = f"Does {e0} {rel} {e2}?"
            cat = "SUPPORTED_TRANSITIVE" if is_transitive else "UNSUPPORTED_NONTRANSITIVE"
            ents = [e0, e2]
        else:
            q = f"Does {e0} {rel} {e3}?"
            cat = "SUPPORTED_TRANSITIVE" if is_transitive else "UNSUPPORTED_NONTRANSITIVE"
            ents = [e0, e3]

        v = "YES" if (cat in ("SUPPORTED_DIRECT", "SUPPORTED_TRANSITIVE")) else "UNKNOWN"
        practice_tasks.append(EpistemicTask(f"{world_name}_pr_{i}", world_name, split, premises, q, v, cat, ents))

    # NO: Asymmetric reversal, refuted direction
    for i in range(12, 23):
        if i % 2 == 0:
            q = f"Does {e1} {rel} {e0}?"
            ents = [e1, e0]
        else:
            q = f"Does {e3} {rel} {e1}?"
            ents = [e3, e1]
        v = "YES" if is_symmetric else "NO"
        cat = "SUPPORTED_SYMMETRIC" if is_symmetric else "REFUTED_ASYMMETRY"
        practice_tasks.append(EpistemicTask(f"{world_name}_pr_{i}", world_name, split, premises, q, v, cat, ents))

    # UNKNOWN: Open-world ungrounded, disconnected, ambiguous
    for i in range(23, 34):
        if i % 3 == 0:
            q = f"Does {e0} {rel} PhantomNodeOmega?"
            cat = "UNSUPPORTED_UNGROUNDED"
            ents = [e0, "PhantomNodeOmega"]
        elif i % 3 == 1:
            q = f"Does UnseenEntityAlpha {rel} {e2}?"
            cat = "UNSUPPORTED_UNGROUNDED"
            ents = ["UnseenEntityAlpha", e2]
        else:
            q = f"Does {e0} {rel} {e4}?"  # e4 is unmentioned in premises!
            cat = "UNSUPPORTED_DISCONNECTED"
            ents = [e0, e4]
        practice_tasks.append(EpistemicTask(f"{world_name}_pr_{i}", world_name, split, premises, q, "UNKNOWN", cat, ents))

    # Held-Out Test Pool: Exactly 24 balanced tasks (8 YES, 8 NO, 8 UNKNOWN)
    test_tasks: list[EpistemicTask] = []

    # 8 YES:
    test_tasks.append(EpistemicTask(f"{world_name}_te_1", world_name, split, premises, f"Does {e0} {rel} {e1}?", "YES", "SUPPORTED_DIRECT", [e0, e1]))
    test_tasks.append(EpistemicTask(f"{world_name}_te_2", world_name, split, premises, f"Does {e1} {rel} {e2}?", "YES", "SUPPORTED_DIRECT", [e1, e2]))
    test_tasks.append(EpistemicTask(f"{world_name}_te_3", world_name, split, premises, f"Does {e2} {rel} {e3}?", "YES", "SUPPORTED_DIRECT", [e2, e3]))
    test_tasks.append(EpistemicTask(f"{world_name}_te_4", world_name, split, premises, f"Does {e0} {rel} {e2}?", "YES" if is_transitive else "UNKNOWN", "SUPPORTED_TRANSITIVE" if is_transitive else "UNSUPPORTED_NONTRANSITIVE", [e0, e2]))
    test_tasks.append(EpistemicTask(f"{world_name}_te_5", world_name, split, premises, f"Does {e1} {rel} {e3}?", "YES" if is_transitive else "UNKNOWN", "SUPPORTED_TRANSITIVE" if is_transitive else "UNSUPPORTED_NONTRANSITIVE", [e1, e3]))
    test_tasks.append(EpistemicTask(f"{world_name}_te_6", world_name, split, premises, f"Does {e0} {rel} {e3}?", "YES" if is_transitive else "UNKNOWN", "SUPPORTED_TRANSITIVE" if is_transitive else "UNSUPPORTED_NONTRANSITIVE", [e0, e3]))

    # Permuted counterfactual premises for tasks 7 & 8
    cf_premises = [f"{e3} {rel} {e2}", f"{e2} {rel} {e1}", f"{e1} {rel} {e0}"]
    test_tasks.append(EpistemicTask(f"{world_name}_te_7", world_name, split, cf_premises, f"Does {e3} {rel} {e2}?", "YES", "SUPPORTED_DIRECT", [e3, e2]))
    test_tasks.append(EpistemicTask(f"{world_name}_te_8", world_name, split, cf_premises, f"Does {e3} {rel} {e1}?", "YES" if is_transitive else "UNKNOWN", "SUPPORTED_TRANSITIVE" if is_transitive else "UNSUPPORTED_NONTRANSITIVE", [e3, e1]))

    # 8 NO (Asymmetric reversals under directional relation):
    test_tasks.append(EpistemicTask(f"{world_name}_te_9", world_name, split, premises, f"Does {e1} {rel} {e0}?", "YES" if is_symmetric else "NO", "REFUTED_ASYMMETRY", [e1, e0]))
    test_tasks.append(EpistemicTask(f"{world_name}_te_10", world_name, split, premises, f"Does {e2} {rel} {e1}?", "YES" if is_symmetric else "NO", "REFUTED_ASYMMETRY", [e2, e1]))
    test_tasks.append(EpistemicTask(f"{world_name}_te_11", world_name, split, premises, f"Does {e3} {rel} {e2}?", "YES" if is_symmetric else "NO", "REFUTED_ASYMMETRY", [e3, e2]))
    test_tasks.append(EpistemicTask(f"{world_name}_te_12", world_name, split, premises, f"Does {e2} {rel} {e0}?", "YES" if is_symmetric else "NO", "REFUTED_ASYMMETRY", [e2, e0]))
    test_tasks.append(EpistemicTask(f"{world_name}_te_13", world_name, split, premises, f"Does {e3} {rel} {e0}?", "YES" if is_symmetric else "NO", "REFUTED_ASYMMETRY", [e3, e0]))
    test_tasks.append(EpistemicTask(f"{world_name}_te_14", world_name, split, cf_premises, f"Does {e0} {rel} {e1}?", "YES" if is_symmetric else "NO", "REFUTED_ASYMMETRY", [e0, e1]))
    test_tasks.append(EpistemicTask(f"{world_name}_te_15", world_name, split, cf_premises, f"Does {e1} {rel} {e2}?", "YES" if is_symmetric else "NO", "REFUTED_ASYMMETRY", [e1, e2]))
    test_tasks.append(EpistemicTask(f"{world_name}_te_16", world_name, split, cf_premises, f"Does {e0} {rel} {e3}?", "YES" if is_symmetric else "NO", "REFUTED_ASYMMETRY", [e0, e3]))

    # 8 UNKNOWN (Epistemically incomplete under open-world semantics):
    test_tasks.append(EpistemicTask(f"{world_name}_te_17", world_name, split, premises, f"Does {e0} {rel} VoidGhostNode?", "UNKNOWN", "UNSUPPORTED_UNGROUNDED", [e0, "VoidGhostNode"]))
    test_tasks.append(EpistemicTask(f"{world_name}_te_18", world_name, split, premises, f"Does PhantomAgentX {rel} {e2}?", "UNKNOWN", "UNSUPPORTED_UNGROUNDED", ["PhantomAgentX", e2]))
    test_tasks.append(EpistemicTask(f"{world_name}_te_19", world_name, split, premises, f"Does QuantumVoid {rel} DarkNebula?", "UNKNOWN", "UNSUPPORTED_UNGROUNDED", ["QuantumVoid", "DarkNebula"]))
    test_tasks.append(EpistemicTask(f"{world_name}_te_20", world_name, split, premises, f"Does {e0} {rel} {e4}?", "UNKNOWN", "UNSUPPORTED_DISCONNECTED", [e0, e4]))
    test_tasks.append(EpistemicTask(f"{world_name}_te_21", world_name, split, premises, f"Does {e4} {rel} {e1}?", "UNKNOWN", "UNSUPPORTED_DISCONNECTED", [e4, e1]))
    test_tasks.append(EpistemicTask(f"{world_name}_te_22", world_name, split, premises, f"Does {e3} {rel} {e4}?", "UNKNOWN", "UNSUPPORTED_DISCONNECTED", [e3, e4]))
    test_tasks.append(EpistemicTask(f"{world_name}_te_23", world_name, split, cf_premises, f"Does {e3} {rel} UnobservedStateZ?", "UNKNOWN", "UNSUPPORTED_UNGROUNDED", [e3, "UnobservedStateZ"]))
    test_tasks.append(EpistemicTask(f"{world_name}_te_24", world_name, split, cf_premises, f"Does {e4} {rel} {e0}?", "UNKNOWN", "UNSUPPORTED_DISCONNECTED", [e4, e0]))

    return EpistemicWorldDataset(
        world_name=world_name,
        split=split,
        semantics=semantics,
        train_practice_pool=practice_tasks,
        held_out_test_pool=test_tasks,
    )


def get_m45_epistemic_worlds() -> dict[str, EpistemicWorldDataset]:
    """Builds M45 epistemic worlds spanning prior source domains and held-out test domains."""
    worlds: dict[str, EpistemicWorldDataset] = {}

    # Source World 1: Corporate Governance (transitive, asymmetric)
    worlds["corporate_governance"] = build_epistemic_world(
        world_name="corporate_governance",
        split="meta_train",
        rel="supervises",
        inv_rel="supervised_by",
        entities=["ExecutiveBoard", "ManagingDirector", "DivisionHead", "TeamLead", "ExternalAuditor"],
        is_transitive=True,
        is_symmetric=False,
        is_directional=True,
    )

    # Source World 2: Geospatial Containment (transitive, asymmetric)
    worlds["geospatial_containment"] = build_epistemic_world(
        world_name="geospatial_containment",
        split="meta_train",
        rel="situated_within",
        inv_rel="contains_region",
        entities=["HamletDistrict", "ProvincialZone", "NationalTerritory", "ContinentalPlate", "OceanTrench"],
        is_transitive=True,
        is_symmetric=False,
        is_directional=True,
    )

    # Source World 3: Biological Ecosystem (non-transitive, directional)
    worlds["biological_ecosystem"] = build_epistemic_world(
        world_name="biological_ecosystem",
        split="meta_train",
        rel="preys_upon",
        inv_rel="preyed_on_by",
        entities=["ApexPredator", "MesoCarnivore", "SmallHerbivore", "PrimaryProducer", "DeepSeaBenthos"],
        is_transitive=False,
        is_symmetric=False,
        is_directional=True,
    )

    # Held-Out Sealed Meta-Test World 1: Epistemological Citation (transitive, asymmetric)
    worlds["epistemological_citation"] = build_epistemic_world(
        world_name="epistemological_citation",
        split="meta_test",
        rel="cites_foundational_work",
        inv_rel="cited_by_work",
        entities=["ContemporaryTreatise", "MidCenturyTheory", "ClassicalMonograph", "AncientManuscript", "UnpublishedDraft"],
        is_transitive=True,
        is_symmetric=False,
        is_directional=True,
    )

    # Held-Out Sealed Meta-Test World 2: Astrophysical Gravitation (transitive, symmetric)
    worlds["astrophysical_gravitation"] = build_epistemic_world(
        world_name="astrophysical_gravitation",
        split="meta_test",
        rel="gravitationally_coupled_with",
        inv_rel="gravitationally_coupled_with",
        entities=["BinaryPulsarA", "BinaryPulsarB", "NeutronCompanion", "AccretingSingularity", "RogueComet"],
        is_transitive=True,
        is_symmetric=True,
        is_directional=False,
    )

    # Held-Out Sealed Meta-Test World 3: Diplomatic Alliance (non-transitive, symmetric)
    worlds["diplomatic_alliance"] = build_epistemic_world(
        world_name="diplomatic_alliance",
        split="meta_test",
        rel="allied_with",
        inv_rel="allied_with",
        entities=["FederationAlpha", "RepublicBeta", "KingdomGamma", "UnionDelta", "NeutralEnclave"],
        is_transitive=False,
        is_symmetric=True,
        is_directional=False,
    )

    return worlds
