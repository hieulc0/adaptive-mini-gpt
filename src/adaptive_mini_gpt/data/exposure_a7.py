"""A7 exposure controls; pure index/string operations, never student inference."""

from __future__ import annotations


def repeated_plan(block, steps):
    if not block or steps <= 0 or steps % len(block):
        raise ValueError("Whole preregistered blocks required")
    return [list(ix) for _ in range(steps // len(block)) for ix in block]


def factor_subset(rows, new_count):
    small = [r for r in rows[:new_count] if r.name == "Rax"]
    by_question = {r.question: i for i, r in enumerate(small)}
    mapping = []
    for row in rows[:new_count]:
        if row.name not in ("Rax", "Vim") or len(row.name.encode()) != 3:
            raise ValueError("Only declared equal-byte names permitted")
        question = row.question.replace(row.name, "Rax")
        target = small[by_question[question]]
        if (target.statuses, target.content, target.frame, target.prefix) != (
            row.statuses,
            row.content,
            row.frame,
            row.prefix,
        ) or len(target.question.encode()) != len(row.question.encode()):
            raise ValueError("Changed non-name factors or token positions")
        mapping.append(by_question[question])
    synthetic = rows[new_count:]
    mapping += [len(small) + i for i in range(len(synthetic))]
    return small + synthetic, len(small), mapping


def window_pass(curves, endpoint, families):
    from adaptive_mini_gpt.data.generalization_a6 import stable_acquisition
    from adaptive_mini_gpt.data.rehearsal_a5 import stable_old

    selected = [
        dict(c, updates=c["updates"] - endpoint + 1200)
        for c in curves
        if c["updates"] in (endpoint - 200, endpoint - 100, endpoint)
    ]
    return dict(
        acquisition=stable_acquisition(selected, families), preservation=stable_old(selected)
    )


def probe_training_indices(rows):
    """Exact balanced status/frame/prefix cells; canonical observable TRAINING variants."""
    chosen = {}
    for i, row in enumerate(rows):
        if row.name != "Rax":
            continue
        parts = [
            w if s == "positive" else "not " + w
            for w, s in zip(("flammable", "conductive"), row.statuses, strict=True)
            if s != "absent"
        ]
        content = "; ".join(parts) if parts else "present"
        if row.content == content:
            key = (tuple(row.statuses), row.frame, row.prefix)
            if key in chosen:
                raise ValueError("Duplicate probe-training cell")
            chosen[key] = i
    if len(chosen) != 162:
        raise ValueError("Incomplete balanced probe training support")
    return list(chosen.values())
