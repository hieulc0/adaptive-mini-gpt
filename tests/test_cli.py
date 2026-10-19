"""End-to-end tests for CLI commands: inspect, train, evaluate, generate, grow."""

from __future__ import annotations

import json
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
    info = json.loads(res_inspect.stdout)
    assert info["architecture"] == "decoder_only_transformer"
    assert info["total_params"] > 0

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
        "--max-batches",
        "2",
        "--device",
        "cpu",
    ]
    res_eval = subprocess.run(eval_cmd, capture_output=True, text=True)
    assert res_eval.returncode == 0
    eval_data = json.loads(res_eval.stdout)
    assert "loss" in eval_data
    assert "perplexity" in eval_data

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
        "--max-new-tokens",
        "20",
        "--device",
        "cpu",
    ]
    res_gen = subprocess.run(gen_cmd, capture_output=True, text=True)
    assert res_gen.returncode == 0
    gen_data = json.loads(res_gen.stdout)
    assert "generated_text" in gen_data

    # 5. Grow checkpoint with residual experts (M4)
    grown_ckpt = save_dir / "grown.pt"
    grow_cmd = [
        sys.executable,
        "-m",
        "adaptive_mini_gpt.cli",
        "grow",
        "--checkpoint",
        str(latest_ckpt),
        "--output",
        str(grown_ckpt),
        "--add-experts",
        "2",
        "--freeze-backbone",
    ]
    res_grow = subprocess.run(grow_cmd, capture_output=True, text=True)
    assert res_grow.returncode == 0, f"Grow failed: {res_grow.stderr}"
    assert grown_ckpt.is_file()
    grow_report = json.loads(res_grow.stdout)
    assert grow_report["status"] == "growth_verified"
    assert grow_report["n_experts"] == 2

    # 6. Inspect grown checkpoint
    inspect_grown_cmd = [
        sys.executable,
        "-m",
        "adaptive_mini_gpt.cli",
        "inspect",
        "--checkpoint",
        str(grown_ckpt),
    ]
    res_inspect_grown = subprocess.run(inspect_grown_cmd, capture_output=True, text=True)
    assert res_inspect_grown.returncode == 0
    grown_info = json.loads(res_inspect_grown.stdout)
    assert grown_info["adaptive_enabled"] is True
    assert grown_info["n_experts"] == 2

    # 7. Grow checkpoint with Mixture-of-Depths (M5)
    mod_ckpt = save_dir / "mod_grown.pt"
    grow_mod_cmd = [
        sys.executable,
        "-m",
        "adaptive_mini_gpt.cli",
        "grow",
        "--checkpoint",
        str(latest_ckpt),
        "--output",
        str(mod_ckpt),
        "--enable-mod",
        "--mod-capacity",
        "0.5",
    ]
    res_mod_grow = subprocess.run(grow_mod_cmd, capture_output=True, text=True)
    assert res_mod_grow.returncode == 0, f"Grow MoD failed: {res_mod_grow.stderr}"
    assert mod_ckpt.is_file()
    mod_report = json.loads(res_mod_grow.stdout)
    assert mod_report["status"] == "growth_verified"
    assert mod_report["mod_enabled"] is True
    assert mod_report["mod_capacity_factor"] == 0.5

    # 8. Continual learning stage 2 from grown checkpoint onto arithmetic
    stage2_dir = tmp_path / "run_cli_stage2"
    train_s2_cmd = [
        sys.executable,
        "-m",
        "adaptive_mini_gpt.cli",
        "train",
        "--config",
        "configs/tiny.yaml",
        "--resume",
        str(grown_ckpt),
        "--data",
        "synthetic:arithmetic",
        "--save-dir",
        str(stage2_dir),
        "--allow-new-dataset",
        "--freeze-backbone",
        "--max-steps",
        "3",
        "--device",
        "cpu",
    ]
    res_s2 = subprocess.run(train_s2_cmd, capture_output=True, text=True)
    assert res_s2.returncode == 0, f"Stage 2 continual train failed: {res_s2.stderr}"
    assert (stage2_dir / "latest.pt").is_file()
