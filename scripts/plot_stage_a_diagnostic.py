"""LOCAL-SAFE: static plots of completed diagnostic JSON, never model loading."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True)
    args = p.parse_args()
    result = json.loads(args.results.read_text())
    root = Path(__file__).resolve().parents[1]
    arms = list(result["neural"])
    cells = list(result["baselines"]["ngram_linear"])
    labels = [
        "Inherited W / inherited Adam",
        "Inherited W / fresh Adam",
        "Fresh W / inherited Adam",
        "Fresh W / fresh Adam",
    ]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1.2, 1]})
    for name, label in zip(arms, labels, strict=True):
        points = result["neural"][name]["curves"]
        ax[0].plot(
            [v["updates"] for v in points],
            [v["training_exact"] for v in points],
            label=label,
            marker="o",
            markersize=3,
        )
    ax[0].axvline(200, color="grey", linestyle=":", linewidth=1)
    ax[0].set(
        title="Training acquisition (tiny → main)",
        xlabel="New optimizer updates",
        ylabel="Exact property-set accuracy",
        ylim=(0, 1.04),
    )
    ax[0].yaxis.set_major_formatter(PercentFormatter(1))
    ax[0].grid(alpha=0.2)
    ax[0].legend(fontsize=7, loc="lower right")
    values = [[result["neural"][a]["cells"][c]["exact"] for c in cells] for a in arms]
    values.append([result["baselines"]["ngram_linear"][c]["exact"] for c in cells])
    ax[1].imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax[1].set(
        xticks=range(4),
        xticklabels=[
            "Known name\nknown frame",
            "New name\nknown frame",
            "Known name\nnew frame",
            "New name\nnew frame",
        ],
        yticks=range(5),
        yticklabels=[*labels, "Word 1–3-gram linear"],
        title="Matched development descriptions",
    )
    ax[1].tick_params(axis="both", labelsize=7)
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax[1].text(
                j,
                i,
                f"{value:.1%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 0.65 else "black",
            )
    losses = list(result["loss_controls"])
    ax[2].bar(range(4), [result["loss_controls"][n]["tiny"]["exact"] for n in losses])
    ax[2].set(
        xticks=range(4),
        xticklabels=["Active 2", "Active 2\n÷ 13", "All 26", "All 26\n× 13"],
        ylim=(0, 1.04),
        title="Loss controls: 200 tiny updates",
        ylabel="Exact property-set accuracy",
    )
    ax[2].yaxis.set_major_formatter(PercentFormatter(1))
    ax[2].grid(axis="y", alpha=0.2)
    fig.text(
        0.04,
        0.02,
        "Seed 42, development only. Same new-stage budgets; lifetime histories differ. Known-name/known-frame cells reuse training descriptions.\nNo sealed qualification, transfer estimate, checkpoint selection or promotion. Span-pooling ablations are excluded from these student scores.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    svg = root / "docs/figures/stage_a_diagnostic_v1.svg"
    fig.savefig(svg)
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    with (root / "data/manifests/stage_a_diagnostic_v1_metrics.csv").open("w") as f:
        w = csv.DictWriter(
            f,
            lineterminator="\n",
            fieldnames=[
                "arm",
                "cell",
                "exact",
                "micro_f1",
                "positive_accuracy",
                "negated_accuracy",
                "absent_accuracy",
            ],
        )
        w.writeheader()
        for name in [*arms, "ngram_linear"]:
            scores = (
                result["baselines"]["ngram_linear"]
                if name == "ngram_linear"
                else result["neural"][name]["cells"]
            )
            for cell, m in scores.items():
                w.writerow(
                    {
                        "arm": name,
                        "cell": cell,
                        "exact": m["exact"],
                        "micro_f1": m["micro_f1"],
                        **{s + "_accuracy": v for s, v in m["status_accuracy"].items()},
                    }
                )


if __name__ == "__main__":
    main()
