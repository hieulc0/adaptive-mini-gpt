"""End-to-end tests for CLI commands: inspect, train, evaluate, generate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cli_inspect_config() -> None:
    res = subprocess.run(
        [sys.executable, "-m", "adaptive_mini_gpt.cli", "inspect", "--config", "configs/tiny.yaml"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "decoder_only_transformer" in res.stdout
    assert "total_params" in res.stdout


def test_cli_train_eval_generate_workflow(tmp_path: Path) -> None:
    # 1. Train for 5 steps on dummy data
    data_file = tmp_path / "sample.txt"
    data_file.write_text("All the world's a stage, and all the men and women merely players. " * 20)
    save_dir = tmp_path / "run_cli"

    train_cmd = [
        sys.executable,
        "-m",
        "adaptive_mini_gpt.cli",
        "train",
        "--config",
        "configs/tiny.yaml",
        "--data",
        str(data_file),
        "--save-dir",
        str(save_dir),
        "--max-steps",
        "5",
        "--device",
        "cpu",
    ]
    res_train = subprocess.run(train_cmd, capture_output=True, text=True)
    assert res_train.returncode == 0, f"Train failed: {res_train.stderr}"

    latest_ckpt = save_dir / "latest.pt"
    assert latest_ckpt.is_file()

    # 2. Inspect saved checkpoint
    inspect_cmd = [
        sys.executable,
        "-m",
        "adaptive_mini_gpt.cli",
        "inspect",
        "--checkpoint",
        str(latest_ckpt),
    ]
    res_inspect = subprocess.run(inspect_cmd, capture_output=True, text=True)
    assert res_inspect.returncode == 0
    assert "Checkpoint Inspection" in res_inspect.stdout
    assert "Step: 5" in res_inspect.stdout

    # 3. Evaluate checkpoint
    eval_cmd = [
        sys.executable,
        "-m",
        "adaptive_mini_gpt.cli",
        "evaluate",
        "--checkpoint",
        str(latest_ckpt),
        "--data",
        str(data_file),
        "--eval-steps",
        "2",
        "--device",
        "cpu",
    ]
    res_eval = subprocess.run(eval_cmd, capture_output=True, text=True)
    assert res_eval.returncode == 0
    assert "Evaluation Results" in res_eval.stdout
    assert "Loss:" in res_eval.stdout

    # 4. Generate from checkpoint
    gen_cmd = [
        sys.executable,
        "-m",
        "adaptive_mini_gpt.cli",
        "generate",
        "--checkpoint",
        str(latest_ckpt),
        "--prompt",
        "All the ",
        "--max-tokens",
        "20",
        "--device",
        "cpu",
    ]
    res_gen = subprocess.run(gen_cmd, capture_output=True, text=True)
    assert res_gen.returncode == 0
    assert "Generated Output" in res_gen.stdout
