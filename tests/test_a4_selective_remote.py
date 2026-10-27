"""REMOTE-REQUIRED: trainability transition retains optimizer state/behavior."""

from __future__ import annotations

from pathlib import Path

import torch

from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda


def test_freezing_preserves_named_adam_states_and_backbone(monkeypatch):
    require_remote_cuda()
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from run_a4_selective import state_by_name
    from run_foundational_audit import digest
    from test_foundation_structure import make_owner

    from adaptive_mini_gpt.training.optimizer_contract import optimizer_membership

    old_det = torch.are_deterministic_algorithms_enabled()
    old_bench = torch.backends.cudnn.benchmark
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    try:
        owner, batch = make_owner()
        owner.scheduler = None
        owner.step(*batch)
        state = state_by_name(owner)
        weights = digest(owner.model.state_dict())
        backbone = digest(owner.model.student.state_dict())
        steps = owner.metadata["steps"]
        owner.model.eval()
        with torch.no_grad():
            before = owner.model(batch[0], batch[1])[1].clone()
        owner.set_mode("frozen_backbone")
        owner.model.eval()
        with torch.no_grad():
            after = owner.model(batch[0], batch[1])[1]
        assert torch.equal(before, after) and digest(owner.model.state_dict()) == weights
        assert state_by_name(owner) == state and owner.metadata["steps"] == steps
        assert not any(optimizer_membership(owner.model, owner.optimizer).values())
        assert all(not p.requires_grad for p in owner.model.student.parameters())
        owner.step(*batch)
        assert digest(owner.model.student.state_dict()) == backbone
        after_states = state_by_name(owner)
        assert all(after_states.get(n) == v for n, v in state.items() if n.startswith("student."))
    finally:
        torch.use_deterministic_algorithms(old_det)
        torch.backends.cudnn.benchmark = old_bench
