"""One bounded historical test-order reproduction; no seed sweep."""

from __future__ import annotations

import importlib.util
import json
import os
import random
import signal
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

PINNED_SHA = "__PINNED_SHA__"
HISTORICAL_SHA = "e8ab5760dbae452bd48de4478e19ee717c9afb42"
REPO = Path("/tmp/amg-m90-reproduction")
OUT = Path("/kaggle/working/m90-reproduction")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Missing historical diagnostic module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    if (
        not Path("/kaggle/working").is_dir()
        or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or not torch.cuda.is_available()
    ):
        raise RuntimeError("Kaggle CUDA required; local/CPU execution refused")
    if len(PINNED_SHA) != 40:
        raise ValueError("Render pushed source SHA")
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            "m104-e-and-m105-a",
            "https://github.com/hieulc0/adaptive-mini-gpt.git",
            str(REPO),
        ],
        check=True,
    )
    subprocess.run(["git", "checkout", "--detach", PINNED_SHA], cwd=REPO, check=True)
    helpers = REPO / "src/adaptive_mini_gpt/training"
    for filename in ("optimizer_contract.py", "reproducibility.py"):
        (OUT / filename).write_text((helpers / filename).read_text())
    (OUT / "repaired_capacity.py").write_text(
        (REPO / "src/adaptive_mini_gpt/concept_instruction/capacity_feasibility.py").read_text()
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(REPO)], check=True
    )
    subprocess.run(["git", "checkout", "--detach", HISTORICAL_SHA], cwd=REPO, check=True)
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "src"))
    torch.set_num_threads(2)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    import pytest

    records = {
        "historical_sha": HISTORICAL_SHA,
        "repair_source_sha": PINNED_SHA,
        "initial_process_seed": 42,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "scope": "one prespecified historical collection-order prefix through M90; no seed search",
        "cases": {},
    }

    class Capture:
        def pytest_collection_modifyitems(self, items):
            endpoint = next(
                i
                for i, item in enumerate(items)
                if item.nodeid.endswith(
                    "test_m90_concept_instruction.py::test_stage_a_capacity_sweep_and_gate_a"
                )
            )
            del items[endpoint + 1 :]
            records["collected_prefix_tests"] = len(items)

        def pytest_runtest_setup(self, item):
            if not item.nodeid.endswith("test_stage_a_capacity_sweep_and_gate_a"):
                return
            from adaptive_mini_gpt.concept_instruction.capacity_feasibility import (
                StageACapacityBenchmark,
            )

            original = StageACapacityBenchmark.evaluate_capacity_sweep

            def captured(*args, **kwargs):
                model = kwargs["model"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "torch_rng": torch.get_rng_state(),
                        "numpy_rng": np.random.get_state(),
                        "python_rng": random.getstate(),
                    },
                    OUT / "historical_m90_initial_state.pt",
                )
                result = original(*args, **kwargs)
                records["cases"]["historical_prefix"] = asdict(result)
                for name in ("optimizer_contract", "reproducibility"):
                    load(f"adaptive_mini_gpt.training.{name}", OUT / f"{name}.py")
                repaired = load("repaired_capacity", OUT / "repaired_capacity.py")
                seed = sys.modules["adaptive_mini_gpt.training.reproducibility"].seed_all
                for ambient in (111, 222):
                    seed(ambient)
                    records["cases"][f"repaired_same_backbone_{ambient}"] = asdict(
                        repaired.StageACapacityBenchmark.evaluate_capacity_sweep(*args, **kwargs)
                    )
                (OUT / "diagnosis.json").write_text(json.dumps(records, indent=2))
                return result

            StageACapacityBenchmark.evaluate_capacity_sweep = captured

    # Preserve assertion failures and the full prefix trace; later tests are not run.
    with (OUT / "prefix.log").open("w") as log:
        from contextlib import redirect_stderr, redirect_stdout

        with redirect_stdout(log), redirect_stderr(log):
            signal.alarm(900)
            code = pytest.main(["-q", f"--junitxml={OUT / 'prefix.xml'}"], plugins=[Capture()])
    signal.alarm(0)
    records["pytest_exit_code"] = int(code)
    (OUT / "diagnosis.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(records, indent=2), flush=True)
    if not records["cases"]:
        raise RuntimeError("M90 reproduction did not execute; inspect prefix.log")


if __name__ == "__main__":
    main()
