"""REMOTE-REQUIRED: real position hooks and Adam tensor-state interventions."""

import json
from pathlib import Path

import pytest
import torch

from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda


def test_position_inspector_is_identity_at_zero_and_removes_hooks(monkeypatch):
    require_remote_cuda()
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from run_attribute_learning import encode, evaluate
    from run_foundational_audit import digest
    from run_frame_a2_diagnosis import position_evaluation
    from test_foundation_structure import make_owner

    from adaptive_mini_gpt.data.attribute_learning import AttributeCase
    from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
        ByteTextTokenizerM103,
    )

    owner, _ = make_owner()
    rows = [
        AttributeCase(**r)
        for r in json.loads(Path("data/fixtures/stage_a_original_tiny.json").read_text())["cases"]
    ]
    batch, _ = encode(rows, ByteTextTokenizerM103(), 256)
    old = digest(owner.model.state_dict())
    table = owner.model.student.base_model.wpe
    assert table is not None
    plain = evaluate(owner.model, batch, rows)
    zero = position_evaluation(owner.model, batch, rows, 0)
    assert plain["logits"] == zero["logits"]
    with pytest.raises(ValueError, match="capacity"):
        position_evaluation(owner.model, batch, rows, 10000)
    assert not table._forward_pre_hooks and digest(owner.model.state_dict()) == old


def test_adam_interventions_preserve_nonselected_state_and_global_clock(monkeypatch):
    require_remote_cuda()
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from frame_a2_common import intervene_backbone
    from test_foundation_structure import make_owner

    owner, batch = make_owner()
    owner.scheduler = None
    owner.step(*batch)
    snapshot = owner.snapshot()
    for strategy in ("moments", "age", "complete"):
        owner.restore(snapshot)
        before = {
            n: {
                k: v.clone()
                for k, v in owner.optimizer.state[p].items()
                if isinstance(v, torch.Tensor)
            }
            for n, p in owner.model.named_parameters()
            if p in owner.optimizer.state
        }
        names = intervene_backbone(owner, strategy)
        assert names and owner.metadata["steps"] == 1
        for n, p in owner.model.named_parameters():
            if n not in before:
                continue
            if n in names:
                if strategy == "complete":
                    assert p not in owner.optimizer.state
                elif strategy == "moments":
                    assert torch.count_nonzero(owner.optimizer.state[p]["exp_avg"]) == 0
                    assert torch.equal(owner.optimizer.state[p]["step"], before[n]["step"])
                else:
                    assert int(owner.optimizer.state[p]["step"]) == 0
                    assert torch.equal(owner.optimizer.state[p]["exp_avg"], before[n]["exp_avg"])
            else:
                for key, value in before[n].items():
                    assert torch.equal(value, owner.optimizer.state[p][key])
