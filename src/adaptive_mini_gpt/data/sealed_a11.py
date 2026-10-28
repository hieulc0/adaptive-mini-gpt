"""Milestone Stage A.11 Independent Sealed Language Qualification Suite.

Permanently retires Stage A.10 sealed suite ('64daf317...').
Introduces 360 pristine sealed test cases across seeds 42, 43, 44:
- Unseen entity names: Drav, Sorn, Kael, Vond, Lira, Meko.
- Four distinct relational families:
  * Family 0: Subordination with distractor clauses.
  * Family 1: Swapped-clause counterfactual minimal pairs (100% Bag-of-Words multiset equality).
  * Family 2: Three-clause distractors with multiple negative clauses.
  * Family 3: Multi-entity relational query binding.
- Hard isolation assertion against training, dev, pretraining, and retired sealed_a10 suites.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.data.attribute_learning import AttributeCase

RETIRED_SEALED_A10_SHA256 = "64daf317c659417327713357449967a9d24c16a3cfe618dd714556c687c14403"

SEALED_NAMES_SEED_42 = ["Drav", "Sorn"]
SEALED_NAMES_SEED_43 = ["Kael", "Vond"]
SEALED_NAMES_SEED_44 = ["Lira", "Meko"]
ALL_SEALED_A11_NAMES = SEALED_NAMES_SEED_42 + SEALED_NAMES_SEED_43 + SEALED_NAMES_SEED_44

FORBIDDEN_ENTITY_NAMES = {
    # Train / dev
    "rax", "vim", "tuv", "wex", "j", "k", "y", "z",
    # Pretrain dummy nouns
    "alpha", "beta", "gamma", "delta", "epsilon", "monolith", "prism",
    # Retired Stage A.10 sealed names
    "zor", "kex", "plin", "vex", "taro", "nul",
}

FORBIDDEN_PHRASES = [
    # Train / dev templates
    "has properties:",
    "description of",
    "for specimen",
    "attribute report",
    "characteristics are given",
    "this entry concerns",
    "concerning",
    "inventory for",
    "catalogued with",
    "properties recorded:",
    "a note on",
    "the following description applies",
    # Retired Stage A.10 templates
    "technical dossier on",
    "in contrast to previous samples",
    "laboratory protocol certifies",
    "although the auxiliary casing is",
]

SEALED_PARTS = ["strut", "flange", "valve", "bracket", "gasket", "piston"]
SEALED_ADJS = ["ductile", "tarnished", "crystalline", "porous", "metallic", "fibrous"]

SEALED_MODIFIERS = [
    "unequivocally and genuinely",
    "manifestly and reliably",
    "authentically and steadily",
]


@dataclass(frozen=True)
class SealedA11Case(AttributeCase):
    name: str
    seed: int
    family_id: int
    family_name: str
    is_counterfactual_pair: bool
    pair_id: str | None
    member: str | None  # "A" or "B" if pair


def _bow(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def generate_sealed_cases_for_seed_a11(seed: int) -> list[SealedA11Case]:
    """Generates exactly 120 sealed evaluation cases for a specific seed."""
    if seed == 42:
        names = SEALED_NAMES_SEED_42
    elif seed == 43:
        names = SEALED_NAMES_SEED_43
    elif seed == 44:
        names = SEALED_NAMES_SEED_44
    else:
        raise ValueError(f"Unauthorized sealed evaluation seed: {seed}. Must be 42, 43, or 44.")

    cases: list[SealedA11Case] = []

    # -----------------------------------------------------------------
    # Family 0: Subordination with Distractor Clause (30 cases)
    # 10 configurations x 3 phrasings = 30 cases
    # -----------------------------------------------------------------
    fam0_configs = [
        # (p1, p2, st, lbl)
        ("flammable", "conductive", ("positive", "positive"), (1, 1)),
        ("flammable", "not conductive", ("positive", "negated"), (1, 0)),
        ("flammable", None, ("positive", "absent"), (1, 0)),
        ("not flammable", "conductive", ("negated", "positive"), (0, 1)),
        ("not flammable", "not conductive", ("negated", "negated"), (0, 0)),
        ("not flammable", None, ("negated", "absent"), (0, 0)),
        (None, "conductive", ("absent", "positive"), (0, 1)),
        (None, "not conductive", ("absent", "negated"), (0, 0)),
        (None, None, ("absent", "absent"), (0, 0)),
        ("flammable", "conductive", ("positive", "positive"), (1, 1)),  # balanced anchor
    ]

    for c_idx, (p1, p2, st, lbl) in enumerate(fam0_configs):
        if p1 and p2:
            body = f"{p1} and {p2}"
        elif p1:
            body = p1
        elif p2:
            body = p2
        else:
            body = "inert with neither trait logged"

        for v_idx in range(3):
            name = names[(c_idx + v_idx) % len(names)]
            part = SEALED_PARTS[(c_idx + v_idx) % len(SEALED_PARTS)]
            adj = SEALED_ADJS[(c_idx + v_idx) % len(SEALED_ADJS)]

            if v_idx == 0:
                q = f"Provided that the {part} is not {adj}, for {name} the finding is {body}."
            elif v_idx == 1:
                q = f"Assuming that the {part} remains without {adj} status, observations of {name} show {body}."
            else:
                q = f"Given that the {part} is never {adj}, evidence on {name} records {body}."

            case_id = f"sealed_a11:{seed}:fam0:{c_idx}_{v_idx}"
            cases.append(
                SealedA11Case(
                    case_id=case_id,
                    question=q,
                    statuses=st,
                    labels=lbl,
                    family="fam0_subordination_distractor",
                    name=name,
                    seed=seed,
                    family_id=0,
                    family_name="subordination_distractor",
                    is_counterfactual_pair=False,
                    pair_id=None,
                    member=None,
                )
            )

    assert len(cases) == 30, f"Expected 30 cases for Family 0, got {len(cases)}"

    # -----------------------------------------------------------------
    # Family 1: Swapped-Clause Counterfactual Minimal Pairs (30 cases = 15 pairs)
    # Exactly identical Bag-of-Words multiset for (case_a, case_b)
    # -----------------------------------------------------------------
    fam1_pair_configs = [
        # (part, adj, p1, p2, st_a, st_b, lbl_a, lbl_b)
        ("strut", "ductile", "flammable", "conductive", ("positive", "positive"), ("negated", "positive"), (1, 1), (0, 1)),
        ("flange", "tarnished", "flammable", "conductive", ("positive", "positive"), ("positive", "negated"), (1, 1), (1, 0)),
        ("valve", "crystalline", "flammable", "conductive", ("positive", "absent"), ("negated", "absent"), (1, 0), (0, 0)),
        ("bracket", "porous", "conductive", "conductive", ("absent", "positive"), ("absent", "negated"), (0, 1), (0, 0)),
        ("gasket", "metallic", "flammable", "conductive", ("positive", "positive"), ("negated", "positive"), (1, 1), (0, 1)),
        ("piston", "fibrous", "flammable", "conductive", ("positive", "positive"), ("positive", "negated"), (1, 1), (1, 0)),
        ("strut", "porous", "flammable", "conductive", ("positive", "absent"), ("negated", "absent"), (1, 0), (0, 0)),
        ("flange", "ductile", "conductive", "conductive", ("absent", "positive"), ("absent", "negated"), (0, 1), (0, 0)),
        ("valve", "tarnished", "flammable", "conductive", ("positive", "positive"), ("negated", "positive"), (1, 1), (0, 1)),
        ("bracket", "crystalline", "flammable", "conductive", ("positive", "positive"), ("positive", "negated"), (1, 1), (1, 0)),
        ("gasket", "fibrous", "flammable", "conductive", ("positive", "absent"), ("negated", "absent"), (1, 0), (0, 0)),
        ("piston", "metallic", "conductive", "conductive", ("absent", "positive"), ("absent", "negated"), (0, 1), (0, 0)),
        ("strut", "tarnished", "flammable", "conductive", ("positive", "positive"), ("negated", "positive"), (1, 1), (0, 1)),
        ("flange", "porous", "flammable", "conductive", ("positive", "positive"), ("positive", "negated"), (1, 1), (1, 0)),
        ("valve", "ductile", "flammable", "conductive", ("positive", "absent"), ("negated", "absent"), (1, 0), (0, 0)),
    ]
    assert len(fam1_pair_configs) == 15

    for p_idx, (part, adj, p1, p2, st_a, st_b, lbl_a, lbl_b) in enumerate(fam1_pair_configs):
        name = names[p_idx % len(names)]
        pid = f"sealed_a11:{seed}:fam1_pair_{p_idx:02d}"

        if st_a == ("positive", "absent") or st_b == ("negated", "absent"):
            q_a = f"The {part} is not {adj}; for {name}, the core is {p1}."
            q_b = f"The {part} is {adj}; for {name}, the core is not {p1}."
        elif st_a == ("absent", "positive") or st_b == ("absent", "negated"):
            q_a = f"The {part} is not {adj}; for {name}, the core is {p2}."
            q_b = f"The {part} is {adj}; for {name}, the core is not {p2}."
        elif st_b == ("negated", "positive"):
            q_a = f"The {part} is not {adj}; for {name}, the core is {p1} and {p2}."
            q_b = f"The {part} is {adj}; for {name}, the core is not {p1} and {p2}."
        else:
            q_a = f"The {part} is not {adj}; for {name}, the core is {p1} and {p2}."
            q_b = f"The {part} is {adj}; for {name}, the core is {p1} and not {p2}."

        case_a = SealedA11Case(
            case_id=f"{pid}_A",
            question=q_a,
            statuses=st_a,
            labels=lbl_a,
            family="fam1_swapped_minimal_pairs",
            name=name,
            seed=seed,
            family_id=1,
            family_name="swapped_minimal_pairs",
            is_counterfactual_pair=True,
            pair_id=pid,
            member="A",
        )
        case_b = SealedA11Case(
            case_id=f"{pid}_B",
            question=q_b,
            statuses=st_b,
            labels=lbl_b,
            family="fam1_swapped_minimal_pairs",
            name=name,
            seed=seed,
            family_id=1,
            family_name="swapped_minimal_pairs",
            is_counterfactual_pair=True,
            pair_id=pid,
            member="B",
        )
        assert Counter(_bow(q_a)) == Counter(_bow(q_b)), f"BoW inequality in {pid}!"
        cases.extend([case_a, case_b])

    assert len(cases) == 60, f"Expected 60 cases after Family 1, got {len(cases)}"

    # -----------------------------------------------------------------
    # Family 2: Three-Clause Distractors (30 cases)
    # Two distractor clauses with negative operators + one target clause
    # -----------------------------------------------------------------
    for c_idx, (p1, p2, st, lbl) in enumerate(fam0_configs):
        if p1 and p2:
            body = f"{p1} and {p2}"
        elif p1:
            body = p1
        elif p2:
            body = p2
        else:
            body = "inert with neither trait logged"

        for v_idx in range(3):
            name = names[(c_idx + v_idx) % len(names)]
            part1 = SEALED_PARTS[(c_idx + v_idx) % len(SEALED_PARTS)]
            part2 = SEALED_PARTS[(c_idx + v_idx + 1) % len(SEALED_PARTS)]
            adj1 = SEALED_ADJS[(c_idx + v_idx) % len(SEALED_ADJS)]
            adj2 = SEALED_ADJS[(c_idx + v_idx + 1) % len(SEALED_ADJS)]

            if v_idx == 0:
                q = f"Whereas component {part1} is not {adj1} and segment {part2} is never {adj2}, inspection of {name} confirms it is {body}."
            elif v_idx == 1:
                q = f"While section {part1} is without {adj1} and fitting {part2} is not {adj2}, analysis of {name} confirms it is {body}."
            else:
                q = f"Though piece {part1} stays never {adj1} and module {part2} is without {adj2}, evidence for {name} confirms it is {body}."

            case_id = f"sealed_a11:{seed}:fam2:{c_idx}_{v_idx}"
            cases.append(
                SealedA11Case(
                    case_id=case_id,
                    question=q,
                    statuses=st,
                    labels=lbl,
                    family="fam2_three_clause_distractor",
                    name=name,
                    seed=seed,
                    family_id=2,
                    family_name="three_clause_distractor",
                    is_counterfactual_pair=False,
                    pair_id=None,
                    member=None,
                )
            )

    assert len(cases) == 90, f"Expected 90 cases after Family 2, got {len(cases)}"

    # -----------------------------------------------------------------
    # Family 3: Multi-Entity Relational Query Binding (30 cases = 15 pairs)
    # Two entities described with contrasting properties, queried at prompt conclusion.
    # Exactly identical Bag-of-Words multiset for (case_a, case_b)
    # -----------------------------------------------------------------
    fam3_configs = [
        # (desc1, desc2, st1, st2, lbl1, lbl2)
        ("flammable", "conductive", ("positive", "absent"), ("absent", "positive"), (1, 0), (0, 1)),
        ("conductive", "flammable", ("absent", "positive"), ("positive", "absent"), (0, 1), (1, 0)),
        ("flammable and conductive", "inert", ("positive", "positive"), ("absent", "absent"), (1, 1), (0, 0)),
        ("flammable and not conductive", "not flammable and conductive", ("positive", "negated"), ("negated", "positive"), (1, 0), (0, 1)),
        ("not flammable and conductive", "flammable and not conductive", ("negated", "positive"), ("positive", "negated"), (0, 1), (1, 0)),
    ]

    pair_idx = 0
    for round_idx in range(3):
        for desc1, desc2, st1, st2, lbl1, lbl2 in fam3_configs:
            pair_idx += 1
            pid = f"sealed_a11:{seed}:fam3_pair_{pair_idx:02d}"
            e1, e2 = names[0], names[1]

            if round_idx == 0:
                q_a = f"Regarding {e1}, the status is {desc1}; whereas for {e2}, the status is {desc2}. Target is {e1}, not {e2}."
                q_b = f"Regarding {e1}, the status is {desc1}; whereas for {e2}, the status is {desc2}. Target is {e2}, not {e1}."
            elif round_idx == 1:
                q_a = f"While {e1} is found to be {desc1}, {e2} remains {desc2}. Query focuses on {e1}, not {e2}."
                q_b = f"While {e1} is found to be {desc1}, {e2} remains {desc2}. Query focuses on {e2}, not {e1}."
            else:
                q_a = f"Report for {e1} indicates {desc1}; in contrast, {e2} shows {desc2}. Evaluated subject is {e1}, not {e2}."
                q_b = f"Report for {e1} indicates {desc1}; in contrast, {e2} shows {desc2}. Evaluated subject is {e2}, not {e1}."

            case_a = SealedA11Case(
                case_id=f"{pid}_A",
                question=q_a,
                statuses=st1,
                labels=lbl1,
                family="fam3_multi_entity_binding",
                name=e1,
                seed=seed,
                family_id=3,
                family_name="multi_entity_binding",
                is_counterfactual_pair=True,
                pair_id=pid,
                member="A",
            )
            case_b = SealedA11Case(
                case_id=f"{pid}_B",
                question=q_b,
                statuses=st2,
                labels=lbl2,
                family="fam3_multi_entity_binding",
                name=e2,
                seed=seed,
                family_id=3,
                family_name="multi_entity_binding",
                is_counterfactual_pair=True,
                pair_id=pid,
                member="B",
            )
            assert Counter(_bow(q_a)) == Counter(_bow(q_b)), f"BoW inequality in {pid}!"
            cases.extend([case_a, case_b])

    assert len(cases) == 120, f"Expected 120 cases for seed {seed}, got {len(cases)}"
    return cases


def generate_master_sealed_suite_a11() -> list[SealedA11Case]:
    """Generates complete 360-case sealed evaluation suite across seeds 42, 43, 44."""
    all_cases: list[SealedA11Case] = []
    for s in (42, 43, 44):
        all_cases.extend(generate_sealed_cases_for_seed_a11(s))
    assert len(all_cases) == 360, f"Expected 360 master sealed cases, got {len(all_cases)}"
    return all_cases


def verify_sealed_a11_isolation(cases: list[SealedA11Case]) -> dict[str, Any]:
    """Strictly audits template, phrase, and entity isolation from training, dev, and retired sealed_a10."""
    seen_questions: set[str] = set()
    hasher = hashlib.sha256()

    for c in cases:
        q_lower = c.question.lower()

        # Check entity names
        name_lower = c.name.lower()
        if name_lower in FORBIDDEN_ENTITY_NAMES:
            raise ValueError(f"Sealed case leaks forbidden entity name: {c.name}")

        # Check forbidden phrases from training, dev, and retired sealed_a10
        for fp in FORBIDDEN_PHRASES:
            if fp.lower() in q_lower:
                raise ValueError(f"Sealed case leaks forbidden training/dev/a10 template phrase: {fp}")

        hasher.update(c.question.encode("utf-8"))
        hasher.update(str(c.labels).encode("utf-8"))
        seen_questions.add(c.question)

    digest = hasher.hexdigest()
    assert digest != RETIRED_SEALED_A10_SHA256, "Pristine sealed suite matches retired sealed_a10 hash!"

    return {
        "isolation_verified": True,
        "total_cases": len(cases),
        "unique_questions": len(seen_questions),
        "sealed_suite_sha256": digest,
        "retired_sealed_a10_sha256": RETIRED_SEALED_A10_SHA256,
        "seeds": sorted(list({c.seed for c in cases})),
        "families": sorted(list({c.family for c in cases})),
        "names": sorted(list({c.name for c in cases})),
        "num_counterfactual_pairs": sum(1 for c in cases if c.is_counterfactual_pair) // 2,
    }
