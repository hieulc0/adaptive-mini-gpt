"""Complete-observation validation for controlled text experiments.

This validates researcher-owned annotations. It is never a student extractor or
an inference-time source of tags, target states or simulator features.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ObservedCase:
    case_id: str
    question: str
    rules: str
    action: str
    tags: tuple[str, ...]
    state: str
    evidence: dict[str, str]


def validate_cases(cases, tag_vocab, action_vocab, state_vocab):
    seen = {}
    for case in cases:
        if case.action not in action_vocab or case.state not in state_vocab:
            raise ValueError("Unknown action/state annotation")
        if not case.rules.strip() or not case.question.strip():
            raise ValueError("Missing required observation")
        for tag in case.tags:
            if tag not in tag_vocab or tag not in case.evidence:
                raise ValueError(f"Unknown or unobservable tag: {tag}")
            evidence = case.evidence[tag]
            if not evidence or evidence not in case.question:
                raise ValueError(f"Missing observable evidence for {tag}")
            # This controlled protocol uses explicit positive declarations only.
            if f"not {evidence.lower()}" in case.question.lower():
                raise ValueError(f"Contradictory positive evidence for {tag}")
        identity = (case.rules, case.question, case.action)
        labels = (tuple(sorted(case.tags)), case.state)
        if identity in seen and seen[identity] != labels:
            raise ValueError("Indistinguishable observations have contradictory labels")
        seen[identity] = labels


def encode_observation(case, tokenizer, context_length):
    """Only observable fields cross this boundary; annotations never enter tokens."""
    ids = tokenizer.encode_prompt(case.question, case.rules, max_length=context_length)
    q = ids.index(tokenizer.QUESTION_ID)
    r = ids.index(tokenizer.RULE_ID)
    sep = ids.index(tokenizer.SEP_ID)
    assert tokenizer.decode(ids[r + 1 : sep]) == case.rules
    assert tokenizer.decode(ids[q + 1 : -1]) == case.question
    assert ids[-1] == tokenizer.ANSWER_ID and len(ids) <= context_length
    return ids


def dataset_hash(splits):
    return hashlib.sha256(
        json.dumps({k: [asdict(c) for c in v] for k, v in splits.items()}, sort_keys=True).encode()
    ).hexdigest()
