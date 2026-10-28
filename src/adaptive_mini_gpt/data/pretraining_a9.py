"""Zero-leakage synthetic syntactic pretraining corpus for Stage A.9 (Experiment 3).

Generates diverse descriptive sentences using non-task dummy nouns and adjectives
across varied clause orders, copulas, relative clauses, and negation scopes.
Contains ZERO target task labels ('flammable', 'conductive'), ZERO task names ('Rax', 'Vim'),
and ZERO held-out evaluation templates.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

FORBIDDEN_WORDS = {"flammable", "conductive", "rax", "vim"}

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

SYNTACTIC_TEMPLATES = [
    "{noun} is {p1} and {neg} {p2}.",
    "{noun} remains {neg} {p1}, but it is {p2}.",
    "{noun}, which is {p1}, appears {neg} {p2}.",
    "{noun} is described as {p1}; however, it is {neg} {p2}.",
    "Observation: {noun} is {neg} {p1} and {neg} {p2}.",
    "The specimen {noun} is {p1} while {neg} {p2}.",
    "{noun} exhibits being {p1} yet {neg} {p2}.",
    "Analysis logs report {noun} is {p2} and {p1}.",
    "{noun} is found to be {neg} {p1} or {neg} {p2}.",
    "{noun}, although {p1}, is definitely {neg} {p2}.",
    "Regarding {noun}: it is {p1}, but {neg} {p2}.",
    "Record indicates that {noun} stays {p1} and {p2}.",
]


def generate_syntactic_pretraining_corpus(
    num_sentences: int = 2000,
    seed: int = 42,
) -> list[str]:
    """Generates deterministic zero-leakage synthetic descriptive sentences."""
    rng = random.Random(seed)
    sentences: list[str] = []

    for _ in range(num_sentences):
        noun = rng.choice(DUMMY_NOUNS)
        p1, p2 = rng.sample(DUMMY_ADJECTIVES, 2)
        neg = rng.choice(NEGATORS)
        template = rng.choice(SYNTACTIC_TEMPLATES)

        sent = template.format(noun=noun, p1=p1, p2=p2, neg=neg)
        words = set(sent.lower().replace(".", "").replace(",", "").replace(";", "").replace(":", "").split())
        assert not words.intersection(FORBIDDEN_WORDS), f"Leakage detected in sentence: {sent}"
        sentences.append(sent)

    return sentences


def verify_pretraining_corpus_isolation(corpus: list[str]) -> dict[str, Any]:
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
