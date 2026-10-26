"""LOCAL-SAFE completed-metric analysis only; optional matplotlib, no model loading."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.results.read_text())
    root = Path(__file__).resolve().parents[1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    records = []
    for ax, group in zip(axes, ("neural", "baselines"), strict=True):
        for name, arm in result[group].items():
            curves = arm.get("curves", [])
            if not curves:
                continue
            label = name.replace("_", " ")
            ax.plot(
                [v["examples"] for v in curves],
                [v["validation_exact"] for v in curves],
                "o-",
                label=label,
                markersize=3,
                linewidth=2 if name.startswith("continuing") else 1.2,
            )
            records.extend({"arm": name, **v} for v in curves)
        ax.axhline(0.9, color="grey", linestyle=":", linewidth=1)
        ax.set(xlabel="New-stage examples observed", ylim=(0, 1.04), title=group.capitalize())
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("Validation exact property-set accuracy")
    fig.suptitle("Attribute-only learning: seed 42, fixed budgets")
    fig.text(
        0.05,
        0.02,
        "Tiny-failed arms stop early. Validation did not select checkpoints. Dotted line: basic held-out target only.\nContinuing student has 200 prior updates; these are not budget-matched transfer curves. Held-out and baseline-advantage gates reported separately.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(root / "docs/figures/attribute_learning_v1_curves.svg")
    with (root / "data/manifests/attribute_learning_v1_curves.csv").open("w") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "arm",
                "updates",
                "examples",
                "loss",
                "validation_exact",
                "gpu_wall_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
