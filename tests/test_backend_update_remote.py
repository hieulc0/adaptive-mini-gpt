"""REMOTE-REQUIRED: actual FP16 overflow, applied updates and exact continuation."""

from pathlib import Path

import pytest
import torch

from adaptive_mini_gpt.data.dataset import TokenDataset
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.training.config import TrainingConfig
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda
from adaptive_mini_gpt.training.reproducibility import seed_all
from adaptive_mini_gpt.training.trainer import Trainer


def build(path):
    seed_all(42)
    config = ModelConfig(
        vocab_size=32, d_model=32, n_layers=1, n_heads=2, d_ff=64, context_length=16, dropout=0
    )
    dataset = TokenDataset(torch.arange(320) % 32, context_length=16)
    training = TrainingConfig(
        device="cuda",
        precision="fp16",
        batch_size=2,
        gradient_accumulation=1,
        max_steps=4,
        lr_schedule="cosine",
        warmup_steps=0,
        eval_every_steps=2,
        eval_steps=1,
        checkpoint_every_steps=2,
        save_dir=str(path),
    )
    return Trainer(MiniGPT(config), config, training, dataset, dataset, None, "overflow-fixture")


def inject_once(trainer):
    first = True

    def overflow(grad):
        nonlocal first
        if first:
            first = False
            return torch.full_like(grad, float("inf"))
        return grad

    return next(trainer.model.parameters()).register_hook(overflow)


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            same(x, y)
    else:
        assert a == b


def test_fp16_skipped_update_and_exact_checkpoint_resume(tmp_path: Path):
    require_remote_cuda()
    full = build(tmp_path / "full")
    handle = inject_once(full)
    full.train()
    handle.remove()
    assert full.step == 4 and full.skipped_updates >= 1
    assert full.attempted_updates == full.step + full.skipped_updates
    assert full.scheduler is not None
    assert full.scheduler.last_epoch == 4
    assert full.tokens_seen == full.attempted_updates * 32
    assert {int(v["step"]) for v in full.optimizer.state.values()} == {4}
    partial = build(tmp_path / "partial")
    handle = inject_once(partial)
    save = partial._save_checkpoint_now

    class Interrupted(Exception):
        pass

    saved = None

    def stop(checkpoint_name, is_best=False):
        nonlocal saved
        result = save(checkpoint_name, is_best)
        if partial.step == 2:
            saved = Path(partial.config.save_dir) / checkpoint_name
            raise Interrupted
        return result

    partial._save_checkpoint_now = stop
    with pytest.raises(Interrupted):
        partial.train()
    handle.remove()
    assert saved is not None
    resumed = build(tmp_path / "resumed")
    resumed.resume_from_checkpoint(saved)
    assert resumed.scheduler is not None and partial.scheduler is not None
    assert resumed.step == 2 and resumed.scheduler.last_epoch == 2
    assert resumed.attempted_updates == partial.attempted_updates
    same(resumed.scaler.state_dict(), partial.scaler.state_dict())
    resumed.train()
    same(full.model.state_dict(), resumed.model.state_dict())
    same(full.optimizer.state_dict(), resumed.optimizer.state_dict())
    same(full.scheduler.state_dict(), resumed.scheduler.state_dict())
    same(full.scaler.state_dict(), resumed.scaler.state_dict())
    assert resumed.attempted_updates == full.attempted_updates
    assert resumed.skipped_updates == full.skipped_updates
