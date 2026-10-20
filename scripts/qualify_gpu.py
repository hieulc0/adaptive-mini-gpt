"""Opt-in CUDA qualification shared by Kaggle, Colab, and local GPU machines."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import torch

from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

ROOT = Path(__file__).resolve().parents[1]


def runtime_info() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU qualification; no CUDA GPU is available")
    free, total = torch.cuda.mem_get_info(0)
    info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "capability": list(torch.cuda.get_device_capability(0)),
        "free_memory_bytes": free,
        "total_memory_bytes": total,
        "device": "cuda:0",
        "precision": "fp16",
        "bf16": "Skipped: qualification uses FP16; T4 has no native BF16 support",
    }
    print(json.dumps(info), flush=True)
    return info


def qualify_fp16() -> None:
    model = MiniGPT(
        ModelConfig(
            vocab_size=128,
            d_model=32,
            n_layers=1,
            n_heads=2,
            d_ff=64,
            context_length=32,
            dropout=0.0,
        )
    ).cuda()
    optimizer = model.configure_optimizers(learning_rate=0.001, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", init_scale=128)
    before = model.lm_head.weight.detach().clone()
    tokens = torch.randint(0, 128, (2, 32), device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.float16):
        logits, loss = model(tokens, tokens)
    assert logits.dtype == torch.float16, "FP16 autocast was not active"
    assert loss is not None and torch.isfinite(loss).item(), "Nonfinite loss"
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all().item() for g in gradients)
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    assert not torch.equal(before, model.lm_head.weight), "Optimizer did not update weights"
    assert all(torch.isfinite(p).all().item() for p in model.parameters())


def qualify_cli(output: Path, device: str = "cuda") -> dict:
    """Exercise the production CLI, including restoring optimizer/scaler and step state."""
    precision = "fp16" if device == "cuda" else "fp32"
    first, resumed = output / "first", output / "resumed"
    command = [
        "amg",
        "train",
        "--config",
        str(ROOT / "configs/gpu_smoke.yaml"),
        "--data",
        "synthetic:arithmetic",
        "--device",
        device,
        "--precision",
        precision,
        "--json-output",
    ]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    for directory, target, extra in [
        (first, 2, []),
        (resumed, 4, ["--resume", str(first / "latest.pt")]),
    ]:
        with (output / f"{directory.name}.log").open("w") as log:
            process = subprocess.Popen(
                command + ["--save-dir", str(directory), "--max-steps", str(target), *extra],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            if process.wait():
                raise RuntimeError(f"amg train failed; see {log.name}")
    a = CheckpointManager.load_raw_checkpoint(first / "latest.pt")
    b = CheckpointManager.load_raw_checkpoint(resumed / "latest.pt")
    assert (a["step"], b["step"]) == (2, 4)
    assert b["tokens_seen"] == 2 * a["tokens_seen"]
    assert b["data_position"] > a["data_position"]
    assert a["optimizer_state_dict"]["state"] and b["optimizer_state_dict"]["state"]
    for state in b["optimizer_state_dict"]["state"].values():
        assert int(state["step"].item()) == 4, "Optimizer state did not resume"
    if device == "cuda":
        assert a["scaler_state_dict"] and b["scaler_state_dict"]
    records = [json.loads(line) for line in (resumed / "metrics.jsonl").read_text().splitlines()]
    train = [r for r in records if r["type"] == "train"]
    assert [r["step"] for r in train] == [3, 4]
    assert all(math.isfinite(r["loss"]) for r in train)
    starts = [
        json.loads(line)
        for line in (output / "resumed.log").read_text().splitlines()
        if line.startswith("{")
    ]
    assert any(r.get("resumed_step") == 2 for r in starts)
    assert all(torch.isfinite(p).all().item() for p in b["model_state_dict"].values())
    assert any(
        not torch.equal(a["model_state_dict"][k], v) for k, v in b["model_state_dict"].items()
    ), "Resume made no parameter updates"
    return {"initial_step": 2, "resumed_step": 4, "cli_resume": "passed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/gpu-qualification"))
    parser.add_argument("--runtime-only", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    info = runtime_info()
    (args.output_dir / "runtime.json").write_text(json.dumps(info, indent=2))
    if args.runtime_only:
        return
    if any((args.output_dir / name).exists() for name in ("first", "resumed")):
        raise RuntimeError("Use a fresh output directory for each qualification")
    qualify_fp16()
    info.update(qualify_cli(args.output_dir))
    info.update(fp16_forward_backward_optimizer="passed", status="passed")
    (args.output_dir / "qualification.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
