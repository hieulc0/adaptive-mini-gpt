"""LOCAL-SAFE: fake CLI, optimizer/scaler/scheduler; no tensor/model operations."""

from argparse import Namespace
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest


def test_cli_configured_seed_precedes_model_construction():
    from adaptive_mini_gpt import cli

    events = []
    args = Namespace(
        config="fake",
        max_steps=None,
        save_dir=None,
        backup_dir=None,
        device=None,
        precision=None,
        resume=None,
        data="fixture",
        freeze_backbone=False,
    )
    tokenizer = SimpleNamespace(vocab_size=128, stoi={}, encode=lambda text: [1] * 100)
    fake_trainer = SimpleNamespace(train=lambda: {"step": 0})
    with (
        patch.object(
            cli,
            "_load_yaml_config",
            return_value={"model": {"context_length": 8}, "training": {"seed": 73}},
        ),
        patch.object(cli, "_load_task_corpus", return_value=("fixture", "hash")),
        patch.object(
            cli, "CharacterTokenizer", return_value=SimpleNamespace(fit=lambda text: tokenizer)
        ),
        patch.object(cli.torch, "tensor", return_value=[1] * 100),
        patch.object(cli, "split_data", return_value=([1] * 50, [1] * 25, [1] * 25)),
        patch.object(cli, "TokenDataset"),
        patch.object(
            cli, "seed_all", side_effect=lambda seed: events.append(("seed", seed)), create=True
        ),
        patch.object(cli, "MiniGPT", side_effect=lambda cfg: events.append(("model", None))),
        patch.object(cli, "Trainer", return_value=fake_trainer),
    ):
        cli.cmd_train(args)
    assert events == [("seed", 73), ("model", None)]


def fake_owner(applied):
    from adaptive_mini_gpt.training.trainer import Trainer

    owner: Any = SimpleNamespace(_apply_update=lambda: Trainer._apply_update(owner))
    callbacks = []

    def step():
        for callback in list(callbacks):
            callback(None, (), {})

    optimizer = SimpleNamespace(
        step=step,
        register_step_post_hook=lambda hook: (
            callbacks.append(hook) or SimpleNamespace(remove=lambda: callbacks.remove(hook))
        ),
    )
    owner.optimizer = optimizer
    owner.scaler = SimpleNamespace(
        is_enabled=lambda: True,
        get_scale=lambda: 128.0,
        step=lambda opt: opt.step() if applied else None,
        update=lambda: None,
    )
    owner.model = SimpleNamespace(parameters=lambda: [])
    owner.config = SimpleNamespace(grad_clip=0)
    owner.step = 2
    owner.attempted_updates = 3
    owner.skipped_updates = 1
    owner.consecutive_skipped_updates = 0
    owner.scheduler = SimpleNamespace(
        last_epoch=2,
        step=lambda: setattr(owner.scheduler, "last_epoch", owner.scheduler.last_epoch + 1),
    )
    return owner, callbacks


@pytest.mark.parametrize("applied", [False, True])
def test_only_applied_update_advances_optimizer_progress_and_scheduler(applied):
    owner, callbacks = fake_owner(applied)
    metrics = owner._apply_update()
    assert metrics["update_applied"] is applied
    assert owner.step == 2 + int(applied)
    assert owner.scheduler.last_epoch == owner.step
    assert owner.attempted_updates == 4
    assert owner.skipped_updates == 1 + int(not applied)
    assert owner.consecutive_skipped_updates == int(not applied)
    assert callbacks == []
