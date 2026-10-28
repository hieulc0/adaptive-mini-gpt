"""Multi-Clause Relational Pretraining Corpus for Stage A.11.

Generates 2,000 synthetic sentences containing multi-clause contrast structures,
subordinating conjunctions ('although', 'whereas', 'while', 'however', 'meanwhile'),
and two-subject descriptions with varied negation scope.
Guarantees ZERO task attribute words ('flammable', 'conductive'), ZERO task names,
and ZERO evaluation templates.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

FORBIDDEN_WORDS = {
    "flammable",
    "conductive",
    "rax",
    "vim",
    "tuv",
    "wex",
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

MULTI_CLAUSE_RELATIONAL_TEMPLATES = [
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
    "In comparison to {noun1} being {p1}, the unit {noun2} is {neg1} {p2} and {p3}.",
    "Though {noun1} remains {neg1} {p1}, reports verify {noun2} is {p2} and {p4}.",
]


def generate_multi_clause_pretraining_corpus(
    num_sentences: int = 2000,
    seed: int = 42,
) -> list[str]:
    """Generates deterministic zero-leakage multi-clause descriptive sentences."""
    rng = random.Random(seed)
    sentences: list[str] = []

    for _ in range(num_sentences):
        noun1, noun2 = rng.sample(DUMMY_NOUNS, 2)
        p1, p2, p3, p4 = rng.sample(DUMMY_ADJECTIVES, 4)
        neg1 = rng.choice(NEGATORS)
        neg2 = rng.choice(NEGATORS)
        template = rng.choice(MULTI_CLAUSE_RELATIONAL_TEMPLATES)

        sent = template.format(
            noun1=noun1,
            noun2=noun2,
            p1=p1,
            p2=p2,
            p3=p3,
            p4=p4,
            neg1=neg1,
            neg2=neg2,
        )
        words = set(sent.lower().replace(".", "").replace(",", "").replace(";", "").replace(":", "").split())
        assert not words.intersection(FORBIDDEN_WORDS), f"Leakage detected in sentence: {sent}"
        sentences.append(sent)

    return sentences


def verify_multi_clause_pretraining_corpus_isolation(corpus: list[str]) -> dict[str, Any]:
    """Verifies complete vocabulary and template isolation from task evaluation splits."""
    joined = " ".join(corpus).lower()
    for word in FORBIDDEN_WORDS:
        if word in joined:
            raise ValueError(f"Forbidden task word '{word}' found in pretraining corpus!")

    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return {
        "num_sentences": len(corpus),
        "corpus_sha256": digest,
        "forbidden_words_found": False,
        "isolation_verified": True,
    }
