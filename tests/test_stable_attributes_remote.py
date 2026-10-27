"""REMOTE-REQUIRED: real gradients, updates, complete stream and process restart."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F

from adaptive_mini_gpt.training.learning_state import paired_concept_loss
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda


def test_paired_loss_has_symmetric_gradients_and_no_parameterized_module():
    require_remote_cuda()
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device="cuda", requires_grad=True)
    loss = paired_concept_loss(z, ((0, 1),))
    assert loss.item() == 1
    loss.backward()
    assert z.grad is not None and torch.all(z.grad.abs().sum(-1) > 0)
    assert paired_concept_loss(z, ((1, 0),)).item() == loss.item()


def test_default_owner_update_matches_original_bce_contract(monkeypatch):
    require_remote_cuda()
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from frame_a2_common import state_digest
    from test_foundation_structure import make_owner

    owner, batch = make_owner()
    owner.scheduler = None
    before = owner.snapshot()
    got = owner.step(*batch[:3], attribute_ids=(10, 15))
    expected = state_digest((owner.model.state_dict(), owner.optimizer.state_dict()))
    owner.restore(before)
    owner.model.train()
    owner.optimizer.zero_grad(set_to_none=True)
    logits = owner.model(batch[0], batch[1])[1][:, (10, 15)]
    loss = F.binary_cross_entropy_with_logits(logits, batch[2][:, (10, 15)])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(owner.model.parameters(), 1)
    owner.optimizer.step()
    assert got == float(loss.detach())
    assert state_digest((owner.model.state_dict(), owner.optimizer.state_dict())) == expected
    batch[2][1, 10] = 1
    with pytest.raises(ValueError, match="matching observed targets"):
        owner.step(
            *batch[:3], attribute_ids=(10, 15), consistency_pairs=((0, 1),), consistency_weight=0.1
        )
    assert state_digest((owner.model.state_dict(), owner.optimizer.state_dict())) == expected


def test_tiny_streams_balance_support_and_match_semantics():
    require_remote_cuda()
    from adaptive_mini_gpt.data.stable_attributes import corpus, streams

    rows = corpus()
    stream = streams(rows, 42, 24)
    assert len(rows) <= 216
    assert set(stream["augmented"][s][j] for s in range(24) for j in range(0, 18, 2)) == set(
        range(len(rows))
    )
    for x, y in zip(stream["fixed"], stream["augmented"], strict=True):
        for j in range(0, 18, 2):
            assert rows[y[j]].statuses == rows[y[j + 1]].statuses
            assert rows[y[j]].family != rows[y[j + 1]].family
        assert [rows[i].labels for i in x] == [rows[i].labels for i in y]


def test_paired_objective_exact_fresh_process_continuation(tmp_path, monkeypatch):
    require_remote_cuda()
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from annotate_frame_a2_diagnostics import signature
    from test_foundation_structure import make_owner

    owner, batch = make_owner()
    owner.scheduler = None
    owner.external["batch"] = batch[:3]
    kw: dict[str, Any] = dict(
        attribute_ids=(10, 15), consistency_pairs=((0, 1),), consistency_weight=0.1
    )
    owner.step(*batch[:3], **kw)
    path = tmp_path / "paired.pt"
    owner.save(path)
    expected_loss = owner.step(*batch[:3], **kw)
    expected = signature(owner)
    code = """
import json,sys
from adaptive_mini_gpt.training.learning_state import LearningState
from annotate_frame_a2_diagnostics import signature
owner=LearningState.load(sys.argv[1])
loss=owner.step(*owner.external['batch'],attribute_ids=(10,15),consistency_pairs=((0,1),),consistency_weight=.1)
print(json.dumps(dict(loss=loss,signature=signature(owner),steps=owner.metadata['steps'])))
"""
    result = subprocess.check_output(
        [sys.executable, "-c", code, str(path)],
        text=True,
        env={**os.environ, "PYTHONPATH": f"{Path('src').resolve()}:{Path('scripts').resolve()}"},
    )
    actual = json.loads(result)
    assert actual == dict(loss=expected_loss, signature=expected, steps=2)


def test_declared_policy_repeats_full_shape_gradients_and_updates(monkeypatch):
    require_remote_cuda()
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from frame_a2_common import fork_owner, state_digest
    from run_attribute_learning import encode
    from test_foundation_structure import make_owner

    from adaptive_mini_gpt.data.stable_attributes import corpus, streams
    from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
        ByteTextTokenizerM103,
    )
    from adaptive_mini_gpt.training.learning_state import deterministic_numerical_policy

    old = torch.are_deterministic_algorithms_enabled()
    benchmark = torch.backends.cudnn.benchmark
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    try:
        parent, _ = make_owner()
        parent.scheduler = None
        rows = corpus()
        batch, _ = encode(rows, ByteTextTokenizerM103(), 256)
        plan = streams(rows, 42, 24)["augmented"]
        owners = [fork_owner(parent, 42), fork_owner(parent, 42)]
        for owner in owners:
            owner.metadata["numerical_policy"] = deterministic_numerical_policy()
        for ix in plan:
            for owner in owners:
                owner.step(*[v[ix] for v in batch], attribute_ids=(10, 15))
            assert state_digest(
                (owners[0].model.state_dict(), owners[0].optimizer.state_dict())
            ) == state_digest((owners[1].model.state_dict(), owners[1].optimizer.state_dict()))
            for x, y in zip(
                owners[0].model.parameters(), owners[1].model.parameters(), strict=True
            ):
                if x.grad is not None:
                    assert torch.equal(x.grad, y.grad)
    finally:
        torch.use_deterministic_algorithms(old)
        torch.backends.cudnn.benchmark = benchmark
