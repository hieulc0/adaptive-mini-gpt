"""CPU checks for packaging/bootstrap, plus an explicitly enabled GPU qualification."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_kaggle_payload(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "kaggle/prepare.py"),
            "--username",
            "tester",
            "--mode",
            "train",
            "--input",
            "tester/amg-resume",
            "--resume",
            "/kaggle/input/amg-resume/latest.pt",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
    )
    metadata = json.loads((tmp_path / "job/kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["dataset_sources"] == ["tester/amg-code", "tester/amg-resume"]
    launcher = load_script("prepared_launcher", str(tmp_path / "job/launch.py"))
    assert launcher.JOB["resume"] == "/kaggle/input/amg-resume/latest.pt"
    with zipfile.ZipFile(tmp_path / "code/amg-code.zip") as bundle:
        names = bundle.namelist()
        assert "src/adaptive_mini_gpt/cli.py" in names
        assert "scripts/qualify_gpu.py" in names
        assert "configs/gpu_smoke.yaml" in names
        assert not any(part in names for part in (".git", ".env", "runs"))
        assert all(n.endswith((".py", ".yaml", ".toml", ".md")) for n in names)


def test_missing_cuda_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    qualification = load_script("qualification_no_cuda", "scripts/qualify_gpu.py")
    monkeypatch.setattr(qualification.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is required"):
        qualification.runtime_info()


def test_qualification_cli_resume_on_cpu(tmp_path: Path) -> None:
    qualification = load_script("qualification_cpu", "scripts/qualify_gpu.py")
    assert qualification.qualify_cli(tmp_path, device="cpu")["resumed_step"] == 4


@pytest.mark.skipif(
    os.environ.get("AMG_RUN_GPU_TESTS") != "1",
    reason="Opt-in: set AMG_RUN_GPU_TESTS=1 on a CUDA machine",
)
def test_gpu_qualification(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/qualify_gpu.py"), "--output-dir", str(tmp_path)],
        check=True,
    )
    assert json.loads((tmp_path / "qualification.json").read_text())["status"] == "passed"


@pytest.mark.parametrize("saved_count,available_count", [(2, 1), (1, 2), (1, 1)])
def test_checkpoint_rng_across_gpu_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved_count: int, available_count: int
) -> None:
    import torch

    from adaptive_mini_gpt.training.checkpoint import CheckpointManager

    saved = [torch.tensor([index], dtype=torch.uint8) for index in range(saved_count)]
    restored = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: available_count)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: restored.extend(states))
    CheckpointManager(tmp_path)._restore_rng_states({"torch_cuda": saved})
    assert len(restored) == min(saved_count, available_count)
    assert all(torch.equal(a, b) for a, b in zip(restored, saved, strict=False))


@pytest.mark.parametrize("mode", ["smoke", "train"])
def test_bootstrap_reuses_cli_and_installed_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    import torch

    inputs = tmp_path / "input/datasets/tester/amg-code"
    inputs.mkdir(parents=True)
    with zipfile.ZipFile(inputs / "amg-code.zip", "w") as bundle:
        bundle.writestr("pyproject.toml", "# example")
    source = (ROOT / "kaggle/launch.py").read_text()
    source = source.replace("/kaggle/working", str(tmp_path / "working"))
    source = source.replace("/kaggle/input", str(tmp_path / "input"))
    launcher_path = tmp_path / "launch.py"
    launcher_path.write_text(source)
    launcher = load_script("launcher_test", str(launcher_path))
    launcher.JOB = {
        "mode": mode,
        "code_dataset": "amg-code",
        "config": "configs/tiny.yaml",
        "data": "synthetic:arithmetic",
        "steps": 10,
        "resume": "/kaggle/input/amg-resume/latest.pt",
    }
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "Tesla T4")
    monkeypatch.setattr(launcher, "run", lambda cmd, cwd: calls.append((cmd, cwd)))
    launcher.main()
    assert (tmp_path / "working/amg-code/pyproject.toml").is_file()
    install = calls[0][0]
    assert install[:4] == [sys.executable, "-m", "pip", "install"]
    assert "--upgrade" not in install and "torch" not in install
    command = calls[-1][0]
    if mode == "train":
        assert command[:2] == ["amg", "train"]
        assert command[command.index("--precision") + 1] == "fp16"
        assert command[command.index("--device") + 1] == "cuda"
        assert command[command.index("--resume") + 1] == launcher.JOB["resume"]
        assert str(tmp_path / "working/amg/train") in command
    else:
        assert command[1] == "scripts/qualify_gpu.py"
        assert "--runtime-only" not in command
