"""LOCAL-SAFE: scalar competence-gate checks only."""

import importlib.util
import sys
from pathlib import Path


def test_modular_t0_gate_fails_closed(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec = importlib.util.spec_from_file_location(
        "run_a4_selective", Path("scripts/run_a4_selective.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.modular_t0_admitted(0.95, 1.0)
    assert not module.modular_t0_admitted(0.949, 1.0)
    assert not module.modular_t0_admitted(float("nan"), 1.0)
    assert not module.modular_t0_admitted(1.1, 1.0)
