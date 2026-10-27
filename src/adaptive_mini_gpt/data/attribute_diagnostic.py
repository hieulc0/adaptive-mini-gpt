"""Development-only name/template factorial; annotations never enter student inference."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from adaptive_mini_gpt.data.attribute_learning import TRAITS, AttributeCase

CELLS = ("familiar_familiar", "novel_familiar", "familiar_novel", "novel_novel")


@dataclass(frozen=True)
class DiagnosticCase(AttributeCase):
    content: str
    name: str
    template_id: int
    matched_id: int


def render(name, content, template):
    if template == 0:
        return f"{name} has properties: {content}."
    if template == 1:
        return f"Description of {name}: {content}."
    if template == 2:
        return f"properties for {name} include {content}."
    if template == 3:
        return f"regarding {name}: its listed properties are {content}."
    raise ValueError("Unknown development template")


def factorial():
    cells = {name: [] for name in CELLS}
    seen = set()
    for statuses, reverse, negation, pair, name_index in itertools.product(
        itertools.product(("positive", "negated", "absent"), repeat=2),
        (False, True),
        ("not", "without"),
        range(2),
        range(2),
    ):
        phrases = [
            word if status == "positive" else f"{negation} {word}"
            for word, status in zip(TRAITS, statuses, strict=True)
            if status != "absent"
        ]
        if reverse:
            phrases.reverse()
        content = "; ".join(phrases) if phrases else "present"
        key = (content, pair, name_index)
        if key in seen:
            continue
        seen.add(key)
        matched = len(cells[CELLS[0]])
        for cell in CELLS:
            novel_name, novel_template = cell.split("_")
            name = (("Y", "Z") if novel_name == "novel" else ("J", "K"))[name_index]
            template = pair + (2 if novel_template == "novel" else 0)
            cells[cell].append(
                DiagnosticCase(
                    f"{cell}-{matched}",
                    render(name, content, template),
                    (statuses[0], statuses[1]),
                    (int(statuses[0] == "positive"), int(statuses[1] == "positive")),
                    "+".join(statuses),
                    content,
                    name,
                    template,
                    matched,
                )
            )
    train = cells["familiar_familiar"]
    tiny = [next(r for r in train if r.family == f) for f in sorted({r.family for r in train})]
    return {"train": train, "tiny": tiny, **cells}
