"""LOCAL-SAFE: fake optimizer state only, no neural model/tensor operations."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    spec = importlib.util.spec_from_file_location(
        "frame_a2_common", Path("scripts/frame_a2_common.py")
    )
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Value:
    def __init__(self, v):
        self.value = v

    def zero_(self):
        self.value = 0


def fake():
    a, b = object(), object()
    state = {p: {"step": Value(200), "exp_avg": Value(2), "exp_avg_sq": Value(3)} for p in (a, b)}
    owner = SimpleNamespace(
        model=SimpleNamespace(
            named_parameters=lambda: [("student.wte", a), ("bottleneck.weight", b)]
        ),
        optimizer=SimpleNamespace(state=state),
        scheduler=None,
        metadata={"steps": 200},
    )
    return owner, a, b


@pytest.mark.parametrize("strategy", ["moments", "age", "complete"])
def test_state_interventions_change_only_declared_backbone_components(strategy):
    owner, a, b = fake()
    m = module()
    assert m.intervene_backbone(owner, strategy) == ["student.wte"]
    assert owner.metadata["steps"] == 200
    assert owner.optimizer.state[b]["step"].value == 200
    assert owner.optimizer.state[b]["exp_avg"].value == 2
    if strategy == "complete":
        assert a not in owner.optimizer.state
    else:
        s = owner.optimizer.state[a]
        assert s["step"].value == (0 if strategy == "age" else 200)
        assert s["exp_avg"].value == (0 if strategy == "moments" else 2)
        assert s["exp_avg_sq"].value == (0 if strategy == "moments" else 3)


def test_rebase_preserves_effective_adam_age_and_rejects_undefined_scheduler():
    owner, a, _ = fake()
    m = module()
    assert m.intervene_backbone(owner, "rebase") == []
    assert owner.optimizer.state[a]["step"].value == 200
    owner.scheduler = object()
    with pytest.raises(ValueError, match="scheduler"):
        m.intervene_backbone(owner, "age")
