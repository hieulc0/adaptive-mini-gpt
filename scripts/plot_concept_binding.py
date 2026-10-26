"""LOCAL-SAFE: plot completed JSON metrics; optional matplotlib, no model loading."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

root = Path(__file__).resolve().parents[1]
r = json.loads(
    (root / "runs/kaggle-concept-binding/output-v1/concept-binding/results.json").read_text()
)
fig, ax = plt.subplots(figsize=(8, 4.8))
series = [
    ("neural", "frozen_backbone", "Frozen student", "o-"),
    ("baselines", "bow_linear", "BoW linear / MLP", "--"),
    ("baselines", "ngram_linear", "Word 1–3-gram linear", "s-"),
    ("baselines", "structured_oracle_mlp", "Oracle features: diagnostic only", "^-."),
]
records = []
for group, name, label, style in series:
    c = r[group][name]["curves"]
    ax.plot(
        [v["updates"] for v in c],
        [v["validation_accuracy"] for v in c],
        style,
        label=label,
        linewidth=1.6,
        markersize=4,
    )
    for v in c:
        records.append(
            {
                "arm": name,
                "optimizer_updates": v["updates"],
                "examples": v["examples"],
                "validation_accuracy": v["validation_accuracy"],
            }
        )
ax.axhline(
    0.8,
    color="grey",
    linestyle=":",
    linewidth=1,
    label="80% family gate (also requires other gates)",
)
ax.set(
    xlabel="Total optimizer updates (200 tiny-fit updates included)",
    ylabel="Validation state accuracy",
    ylim=(0.4, 1),
    title="Concept binding: no independent concept qualification",
)
ax.yaxis.set_major_formatter(PercentFormatter(1))
ax.grid(alpha=0.2)
ax.legend(fontsize=8, loc="upper left")
fig.text(
    0.12,
    0.015,
    "Seed 42 only. Selective/full arms stopped at the tiny attribute gate; no validation curves.\nValidation monitoring used no checkpoint selection; held-out results reported separately.",
    fontsize=8,
)
fig.tight_layout(rect=(0, 0.075, 1, 1))
p = root / "docs/figures"
p.mkdir(exist_ok=True)
fig.savefig(p / "concept_binding_v1_curves.svg")
with (root / "data/manifests/concept_binding_v1_curves.csv").open("w") as f:
    w = csv.DictWriter(
        f, fieldnames=["arm", "optimizer_updates", "examples", "validation_accuracy"]
    )
    w.writeheader()
    w.writerows(records)
