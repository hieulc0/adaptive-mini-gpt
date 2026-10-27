"""REMOTE-REQUIRED: full replay split validation and GPU continuation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda


def test_historical_replay_excludes_protected_descriptions(monkeypatch):
    require_remote_cuda()
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from run_a4_replay import prepare

    new, old, dev, previous, plans, audit = prepare()
    assert len(old) == 9 and audit["historical_old_training"] and int(audit["blocked_count"]) > 1000
    assert len(new) == 156 and len(dev) > 0
    for count in (0, 3, 6):
        assert all(sum(i >= len(new) for i in ix) == count for ix in plans[str(count)])
    assert (
        len({x.question for x in old} & {r.question for rows in previous.values() for r in rows})
        == 0
    )


def test_replay_continuation_preserves_owned_state_across_process(tmp_path, monkeypatch):
    require_remote_cuda()
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from annotate_frame_a2_diagnostics import signature
    from run_a4_replay import advance, prepare
    from run_attribute_learning import encode
    from test_foundation_structure import make_owner

    from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
        ByteTextTokenizerM103,
    )
    from adaptive_mini_gpt.training.learning_state import deterministic_numerical_policy

    old_determinism = torch.are_deterministic_algorithms_enabled()
    old_benchmark = torch.backends.cudnn.benchmark
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    try:
        new, old, _, _, plans, audit = prepare()
        batch, _ = encode(new + old, ByteTextTokenizerM103(), 256)
        owner, _ = make_owner()
        owner.scheduler = None
        owner.metadata["numerical_policy"] = deterministic_numerical_policy()
        owner.external.update(
            a4_indices=plans["6"][:2],
            a4_cursor=0,
            a4_replay_training=[asdict(v) for v in old],
            a4_replay_audit=audit,
            batch=batch,
        )
        before_count = sum(p.numel() for p in owner.model.parameters())
        advance(owner, batch)
        path = tmp_path / "replay.pt"
        owner.save(path)
        expected_loss, expected_ix = advance(owner, batch)
        expected = dict(
            loss=expected_loss,
            indices=expected_ix,
            signature=signature(owner),
            steps=owner.metadata["steps"],
        )
        code = """
import json,sys,torch
from adaptive_mini_gpt.training.learning_state import LearningState
from annotate_frame_a2_diagnostics import signature
from run_a4_replay import advance
torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
owner=LearningState.load(sys.argv[1]);loss,ix=advance(owner,owner.external['batch'])
print(json.dumps(dict(loss=loss,indices=ix,signature=signature(owner),steps=owner.metadata['steps'])))
"""
        result = subprocess.check_output(
            [sys.executable, "-c", code, str(path)],
            text=True,
            env={
                **os.environ,
                "PYTHONPATH": f"{Path('src').resolve()}:{Path('scripts').resolve()}",
            },
        )
        assert json.loads(result) == expected
        assert sum(p.numel() for p in owner.model.parameters()) == before_count
    finally:
        torch.use_deterministic_algorithms(old_determinism)
        torch.backends.cudnn.benchmark = old_benchmark
