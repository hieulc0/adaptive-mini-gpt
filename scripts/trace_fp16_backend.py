"""REMOTE-REQUIRED diagnostic instrumentation; can run against historical checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from adaptive_mini_gpt import cli
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.training.checkpoint import CheckpointManager
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda


def tensor_hash(state):
    h = hashlib.sha256()
    for name, t in sorted(state.items()):
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", required=True)
    p.add_argument("--ambient-seed", type=int, required=True)
    p.add_argument("--resume")
    p.add_argument("--target", type=int, required=True)
    p.add_argument("--force-overflow", action="store_true")
    a = p.parse_args()
    require_remote_cuda()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trace = out / "trace.jsonl"

    def record(event, **kwargs):
        with trace.open("a") as f:
            f.write(json.dumps(dict(event=event, **kwargs)) + "\n")

    torch.manual_seed(a.ambient_seed)
    torch.cuda.manual_seed_all(a.ambient_seed)
    init = MiniGPT.__init__

    def model_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        record(
            "initialization",
            ambient_seed=a.ambient_seed,
            tensor_hash=tensor_hash(self.state_dict()),
        )

    MiniGPT.__init__ = model_init
    adam = torch.optim.AdamW.step
    calls = 0

    def adam_step(self, *args, **kwargs):
        nonlocal calls
        before = [int(s["step"]) for s in self.state.values() if "step" in s]
        result = adam(self, *args, **kwargs)
        calls += 1
        record(
            "adam_applied",
            call=calls,
            before=before,
            after=[int(s["step"]) for s in self.state.values() if "step" in s],
        )
        return result

    torch.optim.AdamW.step = adam_step
    scaler_step = torch.amp.GradScaler.step
    scaler_update = torch.amp.GradScaler.update
    attempts = 0

    def amp_step(self, optimizer, *args, **kwargs):
        nonlocal attempts
        attempts += 1
        record("amp_attempt", attempt=attempts, scale=self.get_scale(), applied_so_far=calls)
        return scaler_step(self, optimizer, *args, **kwargs)

    def amp_update(self, *args, **kwargs):
        before = self.get_scale()
        result = scaler_update(self, *args, **kwargs)
        record("amp_scale", before=before, after=self.get_scale(), applied_so_far=calls)
        return result

    torch.amp.GradScaler.step = amp_step
    torch.amp.GradScaler.update = amp_update
    save = CheckpointManager.save_checkpoint

    def checkpoint(self, *args, **kwargs):
        path = save(self, *args, **kwargs)
        raw = CheckpointManager.load_raw_checkpoint(path)
        record(
            "checkpoint",
            name=Path(path).name,
            step=raw["step"],
            tokens=raw["tokens_seen"],
            data_position=raw["data_position"],
            progress=raw.get("update_progress"),
            optimizer_steps=[
                int(v["step"]) for v in raw["optimizer_state_dict"]["state"].values() if "step" in v
            ],
            scaler=raw["scaler_state_dict"],
            scheduler=raw["scheduler_state_dict"],
            tensor_hash=tensor_hash(raw["model_state_dict"]),
        )
        return path

    CheckpointManager.save_checkpoint = checkpoint
    if a.force_overflow:
        # Inject one deterministic overflow at the loss gradient, before unscale.
        forward = MiniGPT.forward
        injected = False

        def overflowing(self, *args, **kwargs):
            nonlocal injected
            result = forward(self, *args, **kwargs)
            if self.training and not injected and result[1] is not None:
                injected = True
                result[1].register_hook(lambda grad: torch.full_like(grad, float("inf")))
            return result

        MiniGPT.forward = overflowing
    sys.argv = [
        "amg",
        "train",
        "--config",
        "configs/gpu_smoke.yaml",
        "--data",
        "synthetic:arithmetic",
        "--device",
        "cuda",
        "--precision",
        "fp16",
        "--save-dir",
        str(out),
        "--max-steps",
        str(a.target),
        "--json-output",
    ]
    if a.resume:
        sys.argv += ["--resume", a.resume]
    # A non-constant scheduler makes incorrect skipped-step advancement observable.
    load = cli._load_yaml_config

    def config(path):
        c = load(path)
        c["training"].update(lr_schedule="cosine", warmup_steps=0)
        return c

    cli._load_yaml_config = config
    cli.main()


if __name__ == "__main__":
    main()
