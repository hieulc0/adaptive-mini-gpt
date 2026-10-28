"""Milestone Stage A.10 Sealed Evaluation Suite Generator and Isolation Verifier.

Features:
1. 4 Brand-New Grammatical Template Families:
   - Never seen in training, rehearsal, pretraining, or development sets.
2. Disjoint Novel Entity Names:
   - ['Zor', 'Kex', 'Plin', 'Vex', 'Taro', 'Nul'].
3. Long-Range and Distractor Negation Scope ('relationship_sensitive' split):
   - Introduces distractor negation in irrelevant introductory clauses.
   - Separates negators from target properties by 3-5 word modifier phrases
     ('never at all conductive', 'not in any way flammable').
4. Strict Zero-Leakage & Template Isolation Verification with SHA-256 Digest.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from adaptive_mini_gpt.data.attribute_learning import AttributeCase

SEALED_NAMES_SEED_42 = ["Zor", "Kex"]
SEALED_NAMES_SEED_43 = ["Plin", "Vex"]
SEALED_NAMES_SEED_44 = ["Taro", "Nul"]
ALL_SEALED_NAMES = SEALED_NAMES_SEED_42 + SEALED_NAMES_SEED_43 + SEALED_NAMES_SEED_44

FORBIDDEN_ENTITY_NAMES = {
    "rax", "vim", "tuv", "wex", "j", "k", "y", "z", "alpha", "beta", "monolith", "prism"
}

FORBIDDEN_PHRASES = [
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
]

SEALED_TEMPLATES = [
    # Family 0: Formal Technical Dossier
    "Technical dossier on {name}: testing establishes that this material is {content}.",
    # Family 1: Comparative Assessment
    "In contrast to previous samples, evaluation of {name} confirms it is {content}.",
    # Family 2: Certified Laboratory Protocol
    "Laboratory protocol certifies that sample {name} remains {content} under standard conditions.",
    # Family 3: Long-Range & Distractor Negation Scope (Relationship Sensitive)
    "Although the auxiliary casing is {distractor}, diagnostic analysis for object {name} concludes the inner core is {content}.",
]

DISTRACTORS = [
    "not heavy",
    "without flaw",
    "never brittle",
    "not porous",
    "without defect",
    "never unstable",
]

# Modifier phrases that separate negation from property by >= 3 words
LONG_RANGE_PHRASES = {
    ("positive", "positive"): [
        "decidedly flammable and unquestionably conductive",
        "consistently flammable as well as thoroughly conductive",
        "demonstrably flammable and unmistakably conductive",
    ],
    ("positive", "negated"): [
        "decidedly flammable and never at all conductive",
        "unquestionably flammable, but in no respect conductive",
        "thoroughly flammable while without any trace of conductive",
    ],
    ("positive", "absent"): [
        "decidedly flammable",
        "unquestionably flammable with no other attributes noted",
        "consistently flammable while secondary traits are unlisted",
    ],
    ("negated", "positive"): [
        "not in any way flammable, yet decidedly conductive",
        "without any trace of flammable, but unquestionably conductive",
        "never at all flammable, while thoroughly conductive",
    ],
    ("negated", "negated"): [
        "not in any way flammable and never at all conductive",
        "without any trace of flammable, and in no respect conductive",
        "never at all flammable nor in any manner conductive",
    ],
    ("negated", "absent"): [
        "not in any way flammable",
        "without any trace of flammable with no further properties logged",
        "never at all flammable and without secondary listings",
    ],
    ("absent", "positive"): [
        "decidedly conductive",
        "unquestionably conductive with initial traits unrecorded",
        "thoroughly conductive while primary status is omitted",
    ],
    ("absent", "negated"): [
        "never at all conductive",
        "not in any way conductive with initial traits unrecorded",
        "without any trace of conductive and no other properties",
    ],
    ("absent", "absent"): [
        "inert with no affirmative or negated traits reported",
        "completely unlisted with neither quality detected",
        "devoid of reactive traits under standardized analysis",
    ],
}


@dataclass(frozen=True)
class SealedCase(AttributeCase):
    name: str
    seed: int
    family_id: int
    is_relationship_sensitive: bool
    distractor: str | None


def generate_sealed_cases_for_seed(seed: int) -> list[SealedCase]:
    """Generates exactly 108 sealed test cases for a specific seed."""
    if seed == 42:
        names = SEALED_NAMES_SEED_42
    elif seed == 43:
        names = SEALED_NAMES_SEED_43
    elif seed == 44:
        names = SEALED_NAMES_SEED_44
    else:
        raise ValueError(f"Unauthorized sealed evaluation seed: {seed}. Must be 42, 43, or 44.")

    status_pairs = [
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

    cases: list[SealedCase] = []

    # Exactly 108 cases per seed:
    # 4 families x 9 status pairs = 36 cells.
    # Each cell has 3 distinct phrasing/name variations:
    # 36 x 3 = 108 cases.
    for fam_id, template in enumerate(SEALED_TEMPLATES):
        is_rel = fam_id == 3
        for status_pair in status_pairs:
            phrasings = LONG_RANGE_PHRASES[status_pair]
            for var_idx in range(3):
                name = names[var_idx % len(names)]
                content = phrasings[var_idx]
                distractor = DISTRACTORS[(fam_id + var_idx) % len(DISTRACTORS)] if is_rel else None

                if is_rel:
                    assert distractor is not None
                    question = template.format(name=name, content=content, distractor=distractor)
                else:
                    question = template.format(name=name, content=content)

                flam_label = 1 if status_pair[0] == "positive" else 0
                cond_label = 1 if status_pair[1] == "positive" else 0

                case_id = f"sealed:{seed}:{fam_id}:{var_idx}:{status_pair[0]}_{status_pair[1]}"
                case = SealedCase(
                    case_id=case_id,
                    question=question,
                    statuses=status_pair,
                    labels=(flam_label, cond_label),
                    family=f"sealed_family_{fam_id}",
                    name=name,
                    seed=seed,
                    family_id=fam_id,
                    is_relationship_sensitive=is_rel,
                    distractor=distractor,
                )
                cases.append(case)

    assert len(cases) == 108, f"Expected 108 cases, got {len(cases)}"
    return cases


def generate_master_sealed_suite() -> list[SealedCase]:
    """Generates complete 324-case sealed evaluation suite across seeds 42, 43, 44."""
    all_cases: list[SealedCase] = []
    for s in (42, 43, 44):
        all_cases.extend(generate_sealed_cases_for_seed(s))
    assert len(all_cases) == 324, f"Expected 324 master sealed cases, got {len(all_cases)}"
    return all_cases


def verify_sealed_isolation(cases: list[SealedCase]) -> dict[str, Any]:
    """Strictly audits template, phrase, and entity isolation from training & development."""
    seen_questions: set[str] = set()
    hasher = hashlib.sha256()

    for c in cases:
        q_lower = c.question.lower()

        # Check entity names
        name_lower = c.name.lower()
        if name_lower in FORBIDDEN_ENTITY_NAMES:
            raise ValueError(f"Sealed case leaks forbidden entity name: {c.name}")

        # Check forbidden phrases from training and dev
        for fp in FORBIDDEN_PHRASES:
            if fp.lower() in q_lower:
                raise ValueError(f"Sealed case leaks forbidden training/dev template phrase: {fp}")

        hasher.update(c.question.encode("utf-8"))
        hasher.update(str(c.labels).encode("utf-8"))
        seen_questions.add(c.question)

    # In 324 cases, each seed generates 108 distinct questions
    digest = hasher.hexdigest()
    return {
        "isolation_verified": True,
        "total_cases": len(cases),
        "unique_questions": len(seen_questions),
        "sealed_suite_sha256": digest,
        "seeds": sorted(list({c.seed for c in cases})),
        "families": sorted(list({c.family for c in cases})),
        "names": sorted(list({c.name for c in cases})),
    }
