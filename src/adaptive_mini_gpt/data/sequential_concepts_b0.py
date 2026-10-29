"""Stage B.0 Sequential 3-Concept Environment and Portable Data Pipeline.

Constructs a leak-free 3-stage sequential relational concept stream:
1. Concept A (Thermal & Electrical Domain): traits ("flammable", "conductive"), tag_ids (10, 15)
2. Concept B (Structural & Magnetic Domain): traits ("metallic", "magnetic"), tag_ids (0, 18)
3. Concept C (Optical & Crystalline Domain): traits ("luminous", "crystalline"), tag_ids (12, 6)

Features:
- Single-entity + negation ("not", "never", "without") + absent + distractor-clause syntax.
- Unlabeled candidate pools (480 items each) queried in active rounds of 16 queries (total budget: 160).
- Strictly held-out validation sets (unseen entities and unseen grammatical templates) with 0% leak.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.data.attribute_learning import AttributeCase


@dataclass(frozen=True)
class ConceptSpec:
    concept_id: str
    concept_name: str
    traits: tuple[str, str]
    tag_ids: tuple[int, int]
    train_entities: tuple[str, ...]
    held_out_entities: tuple[str, ...]


CONCEPTS: dict[str, ConceptSpec] = {
    "concept_a": ConceptSpec(
        concept_id="concept_a",
        concept_name="Thermal & Electrical Domain",
        traits=("flammable", "conductive"),
        tag_ids=(10, 15),
        train_entities=("Aero", "Bora", "Cato", "Dorn"),
        held_out_entities=("Elan", "Faye"),
    ),
    "concept_b": ConceptSpec(
        concept_id="concept_b",
        concept_name="Structural & Magnetic Domain",
        traits=("metallic", "magnetic"),
        tag_ids=(0, 18),
        train_entities=("Ferr", "Grom", "Hale", "Ivar"),
        held_out_entities=("Jorn", "Kael"),
    ),
    "concept_c": ConceptSpec(
        concept_id="concept_c",
        concept_name="Optical & Crystalline Domain",
        traits=("luminous", "crystalline"),
        tag_ids=(12, 6),
        train_entities=("Lyra", "Miri", "Nyx", "Orin"),
        held_out_entities=("Pike", "Quinn"),
    ),
}

DISTRACTOR_CLAUSES = (
    "dense shell",
    "smooth surface",
    "high mass",
    "rough texture",
    "compact structure",
    "porous boundary",
)


def _build_description(
    trait1: str,
    trait2: str,
    status1: str,
    status2: str,
    negation: str,
    distractor: str | None,
    reverse: bool,
) -> str:
    def _format(trait: str, status: str) -> str | None:
        if status == "positive":
            return trait
        elif status == "negated":
            if negation == "not":
                return f"not {trait}"
            elif negation == "never":
                return f"never {trait}"
            else:
                return f"without {trait} traits"
        return None

    c1 = _format(trait1, status1)
    c2 = _format(trait2, status2)

    parts = [c for c in (c1, c2) if c is not None]
    if reverse:
        parts.reverse()

    if distractor is not None:
        parts.append(distractor)

    if not parts:
        return "neutral form"
    return "; ".join(parts)


def _format_question(template_idx: int, name: str, description: str) -> str:
    if template_idx == 0:
        return f"Object {name}: {description}."
    elif template_idx == 1:
        return f"{name} is described as {description}."
    elif template_idx == 2:
        return f"For object {name}, the description is: {description}."
    elif template_idx == 3:
        return f"The properties described for {name} are: {description}."
    elif template_idx == 4:
        return f"Diagnostic analysis of {name} confirms: {description}."
    else:
        return f"Sensor readouts for {name} record: {description}."


def generate_concept_dataset(
    spec: ConceptSpec,
    seed: int = 42,
    pool_size: int = 480,
    val_size: int = 120,
) -> tuple[list[AttributeCase], list[AttributeCase]]:
    """Generates candidate pool (pool_size) and held-out validation set (val_size)."""
    rng = random.Random(seed)
    statuses_list = [
        ("positive", "positive"),
        ("positive", "negated"),
        ("positive", "absent"),
        ("negated", "positive"),
        ("negated", "negated"),
        ("negated", "absent"),
        ("absent", "positive"),
        ("absent", "negated"),
        ("absent", "absent"),
    ]
    negations = ("not", "never", "without")

    # 1. Candidate Pool using train_entities and templates 0..3
    candidate_cases: list[AttributeCase] = []
    case_idx = 0

    while len(candidate_cases) < pool_size:
        ent = rng.choice(spec.train_entities)
        st1, st2 = rng.choice(statuses_list)
        neg = rng.choice(negations)
        rev = rng.choice([False, True])
        dist = rng.choice(DISTRACTOR_CLAUSES) if rng.random() < 0.5 else None
        tmpl = rng.randint(0, 3)

        desc = _build_description(spec.traits[0], spec.traits[1], st1, st2, neg, dist, rev)
        q_text = _format_question(tmpl, ent, desc)
        label1 = int(st1 == "positive")
        label2 = int(st2 == "positive")

        case_id = f"{spec.concept_id}_pool_{case_idx:04d}"
        c = AttributeCase(
            case_id=case_id,
            question=q_text,
            statuses=(st1, st2),
            labels=(label1, label2),
            family=f"{st1}+{st2}",
        )
        candidate_cases.append(c)
        case_idx += 1

    # 2. Held-out Validation using held_out_entities and held-out templates 4..5
    val_cases: list[AttributeCase] = []
    val_idx = 0

    while len(val_cases) < val_size:
        ent = rng.choice(spec.held_out_entities)
        st1, st2 = rng.choice(statuses_list)
        neg = rng.choice(negations)
        rev = rng.choice([False, True])
        dist = rng.choice(DISTRACTOR_CLAUSES) if rng.random() < 0.5 else None
        tmpl = rng.choice([4, 5])

        desc = _build_description(spec.traits[0], spec.traits[1], st1, st2, neg, dist, rev)
        q_text = _format_question(tmpl, ent, desc)
        label1 = int(st1 == "positive")
        label2 = int(st2 == "positive")

        case_id = f"{spec.concept_id}_val_{val_idx:04d}"
        c = AttributeCase(
            case_id=case_id,
            question=q_text,
            statuses=(st1, st2),
            labels=(label1, label2),
            family=f"{st1}+{st2}",
        )
        val_cases.append(c)
        val_idx += 1

    return candidate_cases, val_cases


class SequentialEnvironmentB0:
    """Queryable environment maintaining unlabeled pools and strict feedback ledger."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.pools: dict[str, list[AttributeCase]] = {}
        self.held_out: dict[str, list[AttributeCase]] = {}
        self.queries_executed: dict[str, int] = {}
        self.queried_indices: dict[str, set[int]] = {}

        for k, spec in CONCEPTS.items():
            cand, val = generate_concept_dataset(spec, seed=seed)
            self.pools[k] = cand
            self.held_out[k] = val
            self.queries_executed[k] = 0
            self.queried_indices[k] = set()

    def get_candidate_questions(self, concept_key: str) -> list[str]:
        """Returns candidate text prompts without ground-truth labels."""
        return [c.question for c in self.pools[concept_key]]

    def get_candidate_cases_unlabeled(self, concept_key: str) -> list[AttributeCase]:
        """Returns candidate cases with dummy zero labels (prevents leakage)."""
        return [
            AttributeCase(
                case_id=c.case_id,
                question=c.question,
                statuses=("masked", "masked"),
                labels=(0, 0),
                family="unlabeled",
            )
            for c in self.pools[concept_key]
        ]

    def query(self, concept_key: str, indices: list[int]) -> list[AttributeCase]:
        """Executes environment query, returning ground-truth labeled cases."""
        results: list[AttributeCase] = []
        pool = self.pools[concept_key]
        for idx in indices:
            if idx < 0 or idx >= len(pool):
                raise IndexError(f"Index {idx} out of range for pool {concept_key}")
            self.queried_indices[concept_key].add(idx)
            results.append(pool[idx])
        self.queries_executed[concept_key] += len(indices)
        return results

    def get_held_out_cases(self, concept_key: str) -> list[AttributeCase]:
        """Returns strictly held-out evaluation set for this concept."""
        return self.held_out[concept_key]

    def get_spec(self, concept_key: str) -> ConceptSpec:
        return CONCEPTS[concept_key]

    def summary(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "queries_executed": dict(self.queries_executed),
            "pool_sizes": {k: len(v) for k, v in self.pools.items()},
            "held_out_sizes": {k: len(v) for k, v in self.held_out.items()},
        }
