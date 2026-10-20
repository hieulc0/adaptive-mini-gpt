"""Prepare a private code dataset and a standalone Kaggle entry script; no API calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument(
        "--mode",
        choices=["smoke", "train", "m19_adaptation"],
        default="smoke",
    )
    parser.add_argument("--config", default="configs/tiny.yaml")
    parser.add_argument("--data", default="synthetic:arithmetic")
    parser.add_argument(
        "--steps", type=int, default=1000, help="Total target step, including resume"
    )
    parser.add_argument("--resume", help="Checkpoint path under /kaggle/input")
    parser.add_argument(
        "--input", action="append", default=[], help="Additional owner/dataset-slug"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/kaggle")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.resume and args.mode != "train":
        parser.error("--resume requires --mode train")
    if args.resume and not args.resume.startswith("/kaggle/input/"):
        parser.error("--resume must refer to an attached /kaggle/input checkpoint")
    code = args.output_dir / "code"
    job = args.output_dir / "job"
    code.mkdir(parents=True, exist_ok=True)
    job.mkdir(parents=True, exist_ok=True)

    # Explicit allowlist: no credentials, git history, datasets, weights, or local env.
    files = [
        ROOT / "pyproject.toml",
        ROOT / "README.md",
        ROOT / "scripts/qualify_gpu.py",
        ROOT / "scripts/run_m19_controlled_adapter_experiment.py",
        ROOT / "scripts/run_m18_t4_planning_benchmark.py",
    ]
    files += sorted((ROOT / "src").rglob("*.py"))
    files += sorted((ROOT / "configs").glob("*.yaml"))

    archive = code / "amg-code.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            if path.exists():
                bundle.write(path, path.relative_to(ROOT))

    metadata = {
        "id": f"{args.username}/amg-code",
        "title": "AMG Code",
        "licenses": [{"name": "other"}],
    }
    (code / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    settings = {
        "mode": args.mode,
        "config": args.config,
        "data": args.data,
        "steps": args.steps,
        "resume": args.resume,
        "code_dataset": "amg-code",
    }
    launcher = (
        (ROOT / "kaggle/launch.py").read_text().replace("JOB = {}", "JOB = " + repr(settings), 1)
    )
    (job / "launch.py").write_text(launcher)
    metadata = json.loads((ROOT / "kaggle/kernel-metadata.json").read_text())
    kernel_slug = f"amg-t4-{args.mode.replace('_', '-')}"
    metadata.update(
        id=f"{args.username}/{kernel_slug}",
        title=f"AMG T4 {args.mode.replace('_', ' ').title()}",
        dataset_sources=[f"{args.username}/amg-code", *args.input],
    )
    (job / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Code SHA256: {hashlib.sha256(archive.read_bytes()).hexdigest()}")
    print(f"Dataset: {code}\nJob: {job}")


if __name__ == "__main__":
    main()
