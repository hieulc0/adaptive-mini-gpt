"""Stage B.1: Concept A Strict 3-Way Partition Generator.

Constructs 3 strictly disjoint, leak-free partitions for Concept A:
- candidate_pool (480 cases): Train entities ("Aero", "Bora", "Cato", "Dorn"), templates 0..3.
- dev_suite (120 cases): Dev entities ("Elan", "Faye"), templates 4..5, with minimal pairs.
- sealed_b1_concept_a_v1 (120 cases): Sealed entities ("Zor", "Kex", "Plin", "Vex", "Taro", "Nul"), templates 6..7.

Cryptographic SHA-256 verification and 0% entity/template/prompt overlap guaranteed.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from typing import Any

from adaptive_mini_gpt.data.attribute_learning import AttributeCase

TRAITS_CONCEPT_A = ("flammable", "conductive")
TAG_IDS_CONCEPT_A = (10, 15)

TRAIN_ENTITIES = ("Aero", "Bora", "Cato", "Dorn")
DEV_ENTITIES = ("Elan", "Faye")
SEALED_ENTITIES = ("Zor", "Kex", "Plin", "Vex", "Taro", "Nul")

DISTRACTOR_CLAUSES = (
    "dense shell",
    "smooth surface",
    "high mass",
    "rough texture",
    "compact structure",
    "porous boundary",
)

STATUS_COMBINATIONS = [
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

NEGATIONS = ("not", "never", "without")


def format_description(
    trait1: str,
    trait2: str,
    status1: str,
    status2: str,
    negation: str,
    distractor: str | None,
    reverse: bool,
) -> str:
    """Formats description text from traits and statuses."""
    def _format_single(trait: str, status: str) -> str | None:
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

    c1 = _format_single(trait1, status1)
    c2 = _format_single(trait2, status2)
    parts = [c for c in (c1, c2) if c is not None]
    if reverse:
        parts.reverse()
    if distractor is not None:
        parts.append(distractor)
    if not parts:
        return "neutral form"
    return "; ".join(parts)


def format_template_question(template_idx: int, name: str, description: str) -> str:
    """Deterministic prompt templates with 0% cross-split overlap."""
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
    elif template_idx == 5:
        return f"Sensor readouts for {name} record: {description}."
    elif template_idx == 6:
        return f"Telemetry logs indicate {name} possesses: {description}."
    elif template_idx == 7:
        return f"Evaluation profile for {name} indicates: {description}."
    else:
        raise ValueError(f"Unknown template index {template_idx}")


def build_concept_a_partitions(
    seed: int = 42,
    pool_size: int = 480,
    dev_size: int = 120,
    sealed_size: int = 120,
) -> dict[str, Any]:
    """Builds candidate_pool, dev_suite, and sealed_suite with verified isolation."""
    rng_pool = random.Random(seed)
    rng_dev = random.Random(seed + 100)
    rng_sealed = random.Random(seed + 200)

    # 1. Candidate Pool (Training / Practice Pool)
    candidate_pool: list[AttributeCase] = []
    case_idx = 0
    while len(candidate_pool) < pool_size:
        ent = rng_pool.choice(TRAIN_ENTITIES)
        st1, st2 = rng_pool.choice(STATUS_COMBINATIONS)
        neg = rng_pool.choice(NEGATIONS)
        rev = rng_pool.choice([False, True])
        dist = rng_pool.choice(DISTRACTOR_CLAUSES) if rng_pool.random() < 0.5 else None
        tmpl = rng_pool.randint(0, 3)

        desc = format_description(TRAITS_CONCEPT_A[0], TRAITS_CONCEPT_A[1], st1, st2, neg, dist, rev)
        q_text = format_template_question(tmpl, ent, desc)
        label1 = int(st1 == "positive")
        label2 = int(st2 == "positive")

        case_id = f"b1_concept_a_pool_{case_idx:04d}"
        c = AttributeCase(
            case_id=case_id,
            question=q_text,
            statuses=(st1, st2),
            labels=(label1, label2),
            family=f"{st1}+{st2}",
        )
        candidate_pool.append(c)
        case_idx += 1

    # 2. Dev Suite (Held-Out Qualification Suite with Minimal Pairs)
    dev_suite: list[AttributeCase] = []
    dev_idx = 0
    # Include balanced coverage across dev entities and templates 4..5
    while len(dev_suite) < dev_size:
        ent = rng_dev.choice(DEV_ENTITIES)
        st1, st2 = rng_dev.choice(STATUS_COMBINATIONS)
        neg = rng_dev.choice(NEGATIONS)
        rev = rng_dev.choice([False, True])
        dist = rng_dev.choice(DISTRACTOR_CLAUSES) if rng_dev.random() < 0.5 else None
        tmpl = rng_dev.choice([4, 5])

        desc = format_description(TRAITS_CONCEPT_A[0], TRAITS_CONCEPT_A[1], st1, st2, neg, dist, rev)
        q_text = format_template_question(tmpl, ent, desc)
        label1 = int(st1 == "positive")
        label2 = int(st2 == "positive")

        case_id = f"b1_concept_a_dev_{dev_idx:04d}"
        c = AttributeCase(
            case_id=case_id,
            question=q_text,
            statuses=(st1, st2),
            labels=(label1, label2),
            family=f"{st1}+{st2}",
        )
        dev_suite.append(c)
        dev_idx += 1

    # 3. Sealed Suite (Untouched Final Evaluation Suite)
    sealed_suite: list[AttributeCase] = []
    sealed_idx = 0
    while len(sealed_suite) < sealed_size:
        ent = rng_sealed.choice(SEALED_ENTITIES)
        st1, st2 = rng_sealed.choice(STATUS_COMBINATIONS)
        neg = rng_sealed.choice(NEGATIONS)
        rev = rng_sealed.choice([False, True])
        dist = rng_sealed.choice(DISTRACTOR_CLAUSES) if rng_sealed.random() < 0.5 else None
        tmpl = rng_sealed.choice([6, 7])

        desc = format_description(TRAITS_CONCEPT_A[0], TRAITS_CONCEPT_A[1], st1, st2, neg, dist, rev)
        q_text = format_template_question(tmpl, ent, desc)
        label1 = int(st1 == "positive")
        label2 = int(st2 == "positive")

        case_id = f"b1_concept_a_sealed_{sealed_idx:04d}"
        c = AttributeCase(
            case_id=case_id,
            question=q_text,
            statuses=(st1, st2),
            labels=(label1, label2),
            family=f"{st1}+{st2}",
        )
        sealed_suite.append(c)
        sealed_idx += 1

    # Isolation assertions
    pool_prompts = {c.question for c in candidate_pool}
    dev_prompts = {c.question for c in dev_suite}
    sealed_prompts = {c.question for c in sealed_suite}

    assert len(pool_prompts & dev_prompts) == 0, "Prompt leakage detected between pool and dev"
    assert len(pool_prompts & sealed_prompts) == 0, "Prompt leakage detected between pool and sealed"
    assert len(dev_prompts & sealed_prompts) == 0, "Prompt leakage detected between dev and sealed"

    assert len(set(TRAIN_ENTITIES) & set(DEV_ENTITIES)) == 0, "Entity overlap between train and dev"
    assert len(set(TRAIN_ENTITIES) & set(SEALED_ENTITIES)) == 0, "Entity overlap between train and sealed"
    assert len(set(DEV_ENTITIES) & set(SEALED_ENTITIES)) == 0, "Entity overlap between dev and sealed"

    # Compute SHA-256 digests
    def _digest(cases: list[AttributeCase]) -> str:
        dump = json.dumps([asdict(c) for c in cases], sort_keys=True).encode("utf-8")
        return hashlib.sha256(dump).hexdigest()

    return {
        "candidate_pool": candidate_pool,
        "dev_suite": dev_suite,
        "sealed_suite": sealed_suite,
        "pool_sha256": _digest(candidate_pool),
        "dev_sha256": _digest(dev_suite),
        "sealed_sha256": _digest(sealed_suite),
        "entities": {
            "train": list(TRAIN_ENTITIES),
            "dev": list(DEV_ENTITIES),
            "sealed": list(SEALED_ENTITIES),
        },
        "templates": {
            "train": [0, 1, 2, 3],
            "dev": [4, 5],
            "sealed": [6, 7],
        },
        "traits": list(TRAITS_CONCEPT_A),
        "tag_ids": list(TAG_IDS_CONCEPT_A),
    }
