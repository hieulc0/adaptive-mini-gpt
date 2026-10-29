"""Multi-Clause Relational Pretraining Corpus with Repeated-Name Induction for Stage A.12.

Generates synthetic sentences containing:
1. Multi-clause contrast structures with varied subordinating conjunctions.
2. Repeated-entity-name co-occurrences across clauses so self-attention heads learn
   to form token-matching induction circuits during pretraining.

Guarantees ZERO task attribute words ('flammable', 'conductive'),
ZERO task entity names, and ZERO evaluation templates.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

FORBIDDEN_WORDS = {
    # Task attribute words
    "flammable",
    "conductive",
    # Training entity names
    "rax",
    "vim",
    "kip",
    "jor",
    "qel",
    "zyp",
    "mox",
    "fex",
    "hul",
    "dak",
    "ryn",
    "gom",
    "pek",
    "yor",
    "vek",
    "zan",
    "tep",
    "lox",
    "nok",
    "coz",
    "biv",
    "wup",
    "jad",
    "fyn",
    "gek",
    "pel",
    "mub",
    "rix",
    # Development entity names
    "tuv",
    "wex",
    "zen",
    "bax",
    "cip",
    "gid",
    "tor",
    "lum",
    "fen",
    # Sealed entity names (active and retired)
    "drav",
    "sorn",
    "kael",
    "vond",
    "lira",
    "meko",
    "zor",
    "kex",
    "plin",
    "vex",
    "taro",
    "nul",
}

DUMMY_NOUNS = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
    "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Omicron", "Pi",
    "Crystal", "Matrix", "Vortex", "Prism", "Beacon", "Pylon", "Relic",
    "Sphere", "Conduit", "Nexus", "Monolith", "Obelisk", "Node", "Core",
]

DUMMY_ADJECTIVES = [
    "crystalline", "opaque", "heavy", "hollow", "dense", "fragile",
    "resilient", "permeable", "rigid", "flexible", "porous", "metallic",
    "fibrous", "viscous", "elastic", "inert", "stable", "volatile",
    "luminescent", "magnetic", "acoustic", "thermal", "static", "dynamic",
]

NEGATORS = ["not", "without", "never"]

# Templates with repeated entity occurrences to cultivate name-matching induction heads
REPEATED_NAME_INDUCTION_TEMPLATES = [
    "Regarding {noun1}, it is {p1}; whereas for {noun2}, it is {p2}. Notice concerning {noun1}: {noun1} remains {p1}.",
    "While {noun1} has a structure that is {p1}, {noun2} is reported {p2}. In summary, {noun1} is decidedly {p1}.",
    "{noun1} exhibits {p1} and {neg1} {p2}; meanwhile {noun2} exhibits {p3}. Analysis of {noun1}: {noun1} is confirmed {p1}.",
    "Record for {noun1} indicates {p1}; in contrast, {noun2} shows {p2}. Focus is on {noun1}: {noun1} stays {p1}.",
    "Although {noun1} appears {neg1} {p1}, {noun2} is {p2} and {p3}. Observation regarding {noun2}: {noun2} exhibits {p2}.",
    "The unit {noun1} is {p1}; nevertheless, {noun2} is {neg1} {p2}. Evaluated status of {noun2}: {noun2} is {neg1} {p2}.",
    "Both {noun1} and {noun2} are tested: {noun1} is {p1}, while {noun2} is {p2}. Target assessment: {noun1} demonstrates {p1}.",
    "Whereas {noun1} is described as {p1}, {noun2} is described as {p2}. Verified trait of {noun2}: {noun2} is definitely {p2}.",
    "For {noun1}, the trait is {p1}; for {noun2}, the trait is {p2}. Query check on {noun1}: {noun1} is indeed {p1}.",
    "Notice shows {noun1} has trait {p1}; conversely {noun2} has trait {p2}. Report for {noun2}: {noun2} proves {p2}.",
]

STANDARD_MULTI_CLAUSE_TEMPLATES = [
    "{noun1} is {p1} and {neg1} {p2}; whereas {noun2} is {p3} and {p4}.",
    "Although {noun1} is {neg1} {p1}, {noun2} remains {p2} and {neg2} {p3}.",
    "While the outer {noun1} is {p1}, for {noun2} the inner core is {neg1} {p2}.",
    "{noun1} is described as {p1}; in contrast, {noun2} is definitely {neg1} {p2}.",
    "Regarding {noun1}: it stays {neg1} {p1}; however, {noun2} is found to be {p2} and {p3}.",
    "Even though {noun1} appears {neg1} {p1}, {noun2} exhibits being {p2} and {p4}.",
    "{noun1} has properties {p1} and {p2}; meanwhile, {noun2} remains {neg1} {p3}.",
    "Whereas {noun1} is {neg1} {p1} and {p2}, {noun2} stays {p3} and {neg2} {p4}.",
    "The component {noun1} is {p1}; nevertheless, {noun2} is {neg1} {p2}.",
    "Analysis indicates {noun1} is {neg1} {p1}, while {noun2} is {p2} and {p3}.",
]


def generate_a12_relational_pretraining_corpus(
    num_sentences: int = 2000,
    seed: int = 42,
    induction_ratio: float = 0.5,
) -> list[str]:
    """Generates synthetic pretraining corpus with repeated-entity induction patterns."""
    rng = random.Random(seed)
    sentences: list[str] = []

    for _ in range(num_sentences):
        noun1, noun2 = rng.sample(DUMMY_NOUNS, 2)
        p1, p2, p3, p4 = rng.sample(DUMMY_ADJECTIVES, 4)
        neg1 = rng.choice(NEGATORS)
        neg2 = rng.choice(NEGATORS)

        if rng.random() < induction_ratio:
            template = rng.choice(REPEATED_NAME_INDUCTION_TEMPLATES)
            sentence = template.format(
                noun1=noun1,
                noun2=noun2,
                p1=p1,
                p2=p2,
                p3=p3,
                neg1=neg1,
                neg2=neg2,
            )
        else:
            template = rng.choice(STANDARD_MULTI_CLAUSE_TEMPLATES)
            sentence = template.format(
                noun1=noun1,
                noun2=noun2,
                p1=p1,
                p2=p2,
                p3=p3,
                p4=p4,
                neg1=neg1,
                neg2=neg2,
            )
        sentences.append(sentence)

    return sentences


def verify_a12_pretraining_corpus_isolation(corpus: list[str]) -> dict[str, Any]:
    """Verifies complete word-level isolation between pretraining corpus and task attributes/names."""
    violations: list[str] = []
    hasher = hashlib.sha256()

    for i, s in enumerate(corpus):
        hasher.update(s.encode("utf-8"))
        words = [w.strip(".,;:?!\"'()").lower() for w in s.split()]
        for w in words:
            if w in FORBIDDEN_WORDS:
                violations.append(f"Sentence {i}: forbidden word '{w}' in '{s}'")

    if violations:
        raise ValueError(
            f"Pretraining corpus isolation violated: {len(violations)} errors found. First: {violations[0]}"
        )

    return {
        "status": "PASS",
        "num_sentences": len(corpus),
        "sha256": hasher.hexdigest(),
        "forbidden_words_checked": len(FORBIDDEN_WORDS),
    }
