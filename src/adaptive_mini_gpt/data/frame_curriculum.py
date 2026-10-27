"""Stage A.2 observable-language curriculum; sealed cases generated only after admission."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from adaptive_mini_gpt.data.attribute_learning import TRAITS, AttributeCase

ARMS = ("fixed", "multi", "multi_order", "multi_prefix", "multi_both")
TRAIN_NAMES = tuple(a + b for a in "JKLM" for b in "JKLM" if a != b)
KNOWN_NAMES = ("JK", "KL", "LM", "MJ")
NOVEL_NAMES = ("JJ", "KK", "LL", "MM")
PREFIXES = ("", "note. ", "this is a record. ", "a note for the record. ")
TRAIN_FRAMES = (
    "{name} has properties: {content}.",
    "Description of {name}: {content}.",
    "the listed traits of {name} are {content}.",
    "a report about {name} says {content}.",
    "{name} is described by {content}.",
    "about {name}. features: {content}.",
)
DEV_FRAMES = (
    "{content}; this describes {name}.",
    "here is {name}. its state is {content}.",
    "observed for {name}: {content}. end of record.",
)
SEALED_FRAMES = (
    "{name} is the item whose features are {content}.",
    "the description is {content}; the item is named {name}.",
    "listed for the item named {name} are these traits: {content}.",
)
STATUSES = ("positive", "negated", "absent")


@dataclass(frozen=True)
class FrameCase(AttributeCase):
    name: str
    content: str
    grammar: str
    prefix: str


def content_of(statuses, reverse, negators):
    chunks = [
        word if s == "positive" else ("not", "without")[negators[j]] + " " + word
        for j, (word, s) in enumerate(zip(TRAITS, statuses, strict=True))
        if s != "absent"
    ]
    if reverse:
        chunks.reverse()
    return "; ".join(chunks) if chunks else "present"


def make_case(
    name,
    statuses,
    frame,
    *,
    reverse=False,
    negators=(0, 0),
    prefix="",
    partition="train",
    case_id="fixture",
):
    if len(statuses) != 2 or any(s not in STATUSES for s in statuses):
        raise ValueError("Invalid annotations")
    frames = {"train": TRAIN_FRAMES, "dev": DEV_FRAMES, "sealed": SEALED_FRAMES}[partition]
    content = content_of(statuses, reverse, negators)
    family = f"{partition}_{frame}"
    return FrameCase(
        case_id,
        prefix + frames[frame].format(name=name, content=content),
        (statuses[0], statuses[1]),
        (int(statuses[0] == "positive"), int(statuses[1] == "positive")),
        family,
        name,
        content,
        family,
        prefix,
    )


def draw_factors(rng):
    # Always consume eight independent draws, including inactive surface factors.
    return (
        rng.randrange(3),
        rng.randrange(3),
        rng.randrange(len(TRAIN_NAMES)),
        rng.randrange(6),
        rng.randrange(2),
        rng.randrange(2),
        rng.randrange(2),
        rng.randrange(4),
    )


def training_case(factors, arm, case_id="train"):
    if arm not in ARMS:
        raise ValueError("Unknown curriculum arm")
    a, b, name, frame, reverse, neg1, neg2, prefix = factors
    return make_case(
        TRAIN_NAMES[name],
        (STATUSES[a], STATUSES[b]),
        frame % 2 if arm == "fixed" else frame,
        reverse=bool(reverse) if arm in ("multi_order", "multi_both") else False,
        negators=(neg1, neg2),
        prefix=PREFIXES[prefix] if arm in ("multi_prefix", "multi_both") else "",
        case_id=case_id,
    )


def evaluation_sets(*, sealed=False, admitted=False):
    if sealed and not admitted:
        raise RuntimeError("Sealed cases are not generated before development admission")
    splits = {}
    namesets = {"known": KNOWN_NAMES, "novel": NOVEL_NAMES}
    if sealed:
        namesets = {"known": KNOWN_NAMES, "novel": ("JKL", "KLM", "LMJ", "MJK")}
    for ntype, names in namesets.items():
        for partition, frames in (
            (("train", TRAIN_FRAMES[:2]), ("sealed", SEALED_FRAMES))
            if sealed
            else (("train", TRAIN_FRAMES[:2]), ("dev", DEV_FRAMES))
        ):
            unique = {}
            for name, statuses, frame, reverse, negators in itertools.product(
                names,
                itertools.product(STATUSES, repeat=2),
                range(len(frames)),
                (False, True),
                itertools.product((0, 1), repeat=2),
            ):
                row = make_case(
                    name,
                    statuses,
                    frame,
                    reverse=bool(reverse),
                    negators=negators,
                    partition=partition,
                    case_id="eval",
                )
                unique.setdefault(row.question, row)
            split = f"{ntype}_{partition}"
            splits[split] = [
                FrameCase(
                    f"{split}-{i}",
                    r.question,
                    r.statuses,
                    r.labels,
                    r.family,
                    r.name,
                    r.content,
                    r.grammar,
                    r.prefix,
                )
                for i, r in enumerate(unique.values())
            ]
    return splits
