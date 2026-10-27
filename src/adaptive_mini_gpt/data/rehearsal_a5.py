"""Historical replay and separately labelled synthetic development rehearsal."""

from __future__ import annotations

import random
from dataclasses import replace

from adaptive_mini_gpt.data.joint_acquisition_a5 import GRAMMARS, STATUSES, verify_observable

SYNTHETIC_NAMES = ("N", "P", "Q", "R")
ARMS = ("none", "historical_6", "synthetic_6", "balanced_9")


def synthetic_old(joint_rows):
    source_name = next(r.name for r in joint_rows if r.group == "old")
    result = []
    for row in joint_rows:
        if row.group != "old" or row.name != source_name:
            continue
        for name in SYNTHETIC_NAMES:
            question = GRAMMARS[row.frame].format(name=name, content=row.content)
            generated = replace(row, name=name, question=question, case_id=question)
            verify_observable(generated)
            result.append(generated)
    return result


def plans(new, historical, synthetic, seed, steps):
    """Identical unreplaced slots and exactly matched per-step annotation pairs."""
    ni = {(r.frame, r.statuses, r.name, r.content): i for i, r in enumerate(new)}
    hi = {tuple(r.statuses): len(new) + i for i, r in enumerate(historical)}
    si = {
        (r.frame, r.statuses, r.name, r.content): len(new) + len(historical) + i
        for i, r in enumerate(synthetic)
    }
    choices = {s: sorted({(r.name, r.content) for r in new if r.statuses == s}) for s in STATUSES}
    rng = random.Random(seed)
    result = {a: [] for a in ARMS}
    for step in range(steps):
        batches = {a: [] for a in ARMS}
        for j in rng.sample(range(9), 9):
            status = STATUSES[j]
            name, content = rng.choice(choices[status])
            old_name = rng.choice(SYNTHETIC_NAMES)
            # Rotate both within and between three-step replacement cycles.
            # A flat step%3 rotation aliases grammar to the second status label.
            frame = (step // 3 + step % 3 + j) % 3
            left = ni[(2 + frame, status, name, content)]
            right = ni[(2 + (frame + 1) % 3, status, name, content)]
            syn = si[((step + j) % 2, status, old_name, content)]
            selected = j % 3 != step % 3  # six slots; each status replaced twice per3steps
            for arm in ARMS:
                replacement = right
                if arm == "balanced_9" or (arm == "synthetic_6" and selected):
                    replacement = syn
                elif arm == "historical_6" and selected:
                    replacement = hi[status]
                batches[arm].extend((left, replacement))
        for arm in ARMS:
            result[arm].append(batches[arm])
    return result


def stable_new(curves):
    selected = [c for c in curves if c["updates"] in (1000, 1100, 1200)]
    if [c["updates"] for c in selected] != [1000, 1100, 1200]:
        return False
    for c in selected:
        m = c["training"]
        if set(m["family_exact"]) != {"frame_2", "frame_3", "frame_4"}:
            return False
        if set(m["status_accuracy"]) != {"positive", "negated", "absent"}:
            return False
        if not all(
            0 <= v <= 1
            for v in (m["exact"], *m["family_exact"].values(), *m["status_accuracy"].values())
        ):
            return False
        if not (
            m["exact"] >= 0.99
            and min(m["family_exact"].values()) >= 0.99
            and min(m["status_accuracy"].values()) >= 0.98
        ):
            return False
    return True


def stable_old(curves):
    selected = [c for c in curves if c["updates"] in (1000, 1100, 1200)]
    return [c["updates"] for c in selected] == [1000, 1100, 1200] and all(
        set(c["retention"]) == {"old_known", "old_renamed"}
        and all(0.95 <= m["exact"] <= 1 for m in c["retention"].values())
        for c in selected
    )
