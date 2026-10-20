"""Kaggle entry script. Only this file is uploaded by kernels push.

prepare.py fills JOB; repository files arrive through an attached private dataset.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

JOB = {}  # Filled by prepare.py; do not put credentials here.


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    if not JOB:
        raise RuntimeError("Run kaggle/prepare.py locally before submitting this script")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU job requires CUDA; check accelerator and account quota")
    print(
        f"Preinstalled torch={torch.__version__}, GPU={torch.cuda.get_device_name(0)}", flush=True
    )
    working = Path("/kaggle/working")
    working.mkdir(parents=True, exist_ok=True)
    repo = working / "amg-code"

    archive = Path("/kaggle/input") / JOB["code_dataset"] / "amg-code.zip"
    if archive.is_file():
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.namelist():
                if not (repo / member).resolve().is_relative_to(repo.resolve()):
                    raise ValueError(f"Invalid archive path: {member}")
            bundle.extractall(repo)
    else:
        # Check if Kaggle automatically unzipped the dataset
        pyproject_candidates = list(Path("/kaggle/input").rglob("pyproject.toml"))
        if pyproject_candidates:
            source_dir = pyproject_candidates[0].parent
            if repo.exists():
                shutil.rmtree(repo)
            shutil.copytree(source_dir, repo)
        else:
            zip_candidates = list(Path("/kaggle/input").rglob("amg-code.zip"))
            if len(zip_candidates) == 1:
                with zipfile.ZipFile(zip_candidates[0]) as bundle:
                    bundle.extractall(repo)
            else:
                input_tree = list(Path("/kaggle/input").rglob("*"))
                raise FileNotFoundError(
                    f"Neither amg-code.zip nor unzipped repo found in /kaggle/input: {input_tree[:20]}"
                )

    # Reuse the runtime's CUDA torch. pip only resolves missing/incompatible dependencies.
    run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", str(repo)], repo)
    os.environ["PYTHONUNBUFFERED"] = "1"
    output = working / "amg"
    output.mkdir(exist_ok=True)
    (output / "job.json").write_text(json.dumps(JOB, indent=2))
    run(
        [sys.executable, "scripts/qualify_gpu.py", "--runtime-only", "--output-dir", str(output)],
        repo,
    )
    if JOB["mode"] == "smoke":
        run([sys.executable, "scripts/qualify_gpu.py", "--output-dir", str(output)], repo)
    elif JOB["mode"] == "m19_adaptation":
        run(
            [
                sys.executable,
                "scripts/run_m19_controlled_adapter_experiment.py",
                "--device",
                "cuda",
                "--seeds",
                "42",
                "43",
                "44",
                "--epochs",
                "15",
                "--output-dir",
                str(output / "m19_adaptation"),
            ],
            repo,
        )
    else:
        command = [
            "amg",
            "train",
            "--config",
            JOB["config"],
            "--data",
            JOB["data"],
            "--device",
            "cuda",
            "--precision",
            "fp16",
            "--max-steps",
            str(JOB["steps"]),
            "--save-dir",
            str(output / "train"),
            "--json-output",
        ]
        if JOB["resume"]:
            command += ["--resume", JOB["resume"]]
        run(command, repo)


if __name__ == "__main__":
    main()
