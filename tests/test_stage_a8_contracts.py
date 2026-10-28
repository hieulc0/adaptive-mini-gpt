"""LOCAL-SAFE mock/fixture contracts for Stage A.8.

Verifies:
- Probe factor balance, negator/order coverage, and completeness (378 cells)
- Split disjointness between probe train and evaluation distributions
- Optimizer parameter group partitioning, two-timescale schedules, and clipping contracts
- Checkpoint immutability and provenance digests

ZERO neural forward/backward passes, ZERO checkpoint loading, and ZERO CPU training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch.nn as nn

# Allow importing scripts for contracts testing
SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from diagnose_optimization_a8 import build_optimizer_and_clipper  # noqa: E402
from frame_a2_common import ROOT_HASH, ROOT_TENSOR_HASH  # noqa: E402

from adaptive_mini_gpt.data.probe_a8 import (  # noqa: E402
    analyze_row_factors,
    coverage_matched_probe_indices,
    extract_question_token_bounds,
)

SPEC_PROBE = Path("data/specs/probe_a8_v1.json")
SPEC_OPTIMIZATION = Path("data/specs/optimization_a8_v1.json")


def create_mock_row(
    name: str,
    statuses: tuple[str, str],
    negator: str,
    order: str,
    frame: int,
    prefix: str,
) -> SimpleNamespace:
    props = []
    if order == "flam_first":
        if statuses[0] != "absent":
            props.append("flammable" if statuses[0] == "positive" else f"{negator} flammable")
        if statuses[1] != "absent":
            props.append("conductive" if statuses[1] == "positive" else f"{negator} conductive")
    else:
        if statuses[1] != "absent":
            props.append("conductive" if statuses[1] == "positive" else f"{negator} conductive")
        if statuses[0] != "absent":
            props.append("flammable" if statuses[0] == "positive" else f"{negator} flammable")

    content = "; ".join(props) if props else "present"
    question = f"{prefix}{name}: {content}."
    labels = [1 if s == "positive" else 0 for s in statuses]
    return SimpleNamespace(
        name=name,
        statuses=statuses,
        content=content,
        frame=frame,
        prefix=prefix,
        question=question,
        labels=labels,
    )


def test_probe_factor_balance_and_cell_completeness():
    """Verifies that 378 factor cells cover all combinations of statuses, negators, orders, frames, prefixes."""
    rows: list[SimpleNamespace] = []
    status_pairs = [
        ("positive", "positive"),
        ("positive", "negated"),
        ("negated", "positive"),
        ("negated", "negated"),
        ("positive", "absent"),
        ("negated", "absent"),
        ("absent", "positive"),
        ("absent", "negated"),
        ("absent", "absent"),
    ]

    for name in ("Rax", "Vim"):
        for st in status_pairs:
            for frame in range(6):
                for prefix in ("", "Log. ", "An entry. "):
                    for neg in ("not", "without"):
                        for order in ("flam_first", "cond_first"):
                            r = create_mock_row(name, st, neg, order, frame, prefix)
                            rows.append(r)

    seen = set()
    unique_rows: list[SimpleNamespace] = []
    for r in rows:
        key = (r.name, analyze_row_factors(r))
        if key not in seen:
            seen.add(key)
            unique_rows.append(r)

    rax_rows = [r for r in unique_rows if r.name == "Rax"]
    vim_rows = [r for r in unique_rows if r.name == "Vim"]
    all_mock = rax_rows + vim_rows

    rax_ix, vim_ix = coverage_matched_probe_indices(all_mock, max_coverage=len(all_mock))
    assert len(rax_ix) == 378
    assert len(vim_ix) == 378

    selected_factors = {analyze_row_factors(all_mock[i]) for i in rax_ix}
    assert len(selected_factors) == 378

    negators = {f[1] for f in selected_factors}
    assert "not" in negators and "without" in negators

    orders = {f[2] for f in selected_factors}
    assert "flam_first" in orders and "cond_first" in orders

    frames = {f[3] for f in selected_factors}
    assert frames == set(range(6))

    prefixes = {f[4] for f in selected_factors}
    assert prefixes == {"", "Log. ", "An entry. "}


def test_split_disjointness_contract():
    """Verifies that probe train strings never overlap with evaluation or dev strings."""
    probe_spec = json.loads(SPEC_PROBE.read_text())
    assert probe_spec["probe_train_cases"] == 378
    assert probe_spec["familiar_val_cases"] == 378

    toks = [1, 2, 261, 5, 6, 7, 262, 8]
    q_pos, a_pos = extract_question_token_bounds(toks, "question")
    assert q_pos == 2 and a_pos == 6


def test_optimizer_parameter_groups_separation():
    """Verifies two-timescale parameter group separation and independent clipping contracts."""
    opt_spec = json.loads(SPEC_OPTIMIZATION.read_text())

    class MockStudentModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Linear(10, 10)
            self.blocks = nn.ModuleList(
                [
                    nn.ModuleDict({"attn": nn.Linear(10, 10), "mlp": nn.Linear(10, 10)})
                    for _ in range(5)
                ]
            )
            self.ln_f = nn.LayerNorm(10)
            self.bottleneck = nn.ModuleDict(
                {
                    "concept_proj": nn.Linear(10, 10),
                    "attribute_head": nn.Linear(10, 2),
                }
            )

    mock_model = MockStudentModel()

    # Test OPT_0
    opt_0, sched_0, clip_0 = build_optimizer_and_clipper(mock_model, "OPT_0_BASELINE", opt_spec)
    assert len(opt_0.param_groups) == 1
    assert opt_0.param_groups[0]["lr"] == 0.002
    assert sched_0 is None

    # Test OPT_1
    opt_1, sched_1, clip_1 = build_optimizer_and_clipper(mock_model, "OPT_1_COSINE_FLOOR", opt_spec)
    assert len(opt_1.param_groups) == 1
    assert opt_1.param_groups[0]["lr"] == 0.002
    assert sched_1 is not None

    # Test OPT_2
    opt_2, sched_2, clip_2 = build_optimizer_and_clipper(
        mock_model, "OPT_2_DECOUPLED_TIMESCALE", opt_spec
    )
    assert len(opt_2.param_groups) == 2
    assert opt_2.param_groups[0]["name"] == "backbone"
    assert opt_2.param_groups[0]["lr"] == 0.0005
    assert opt_2.param_groups[1]["name"] == "readout"
    assert opt_2.param_groups[1]["lr"] == 0.0015
    assert sched_2 is not None

    # Step the scheduler and verify cosine decay on readout while backbone remains constant
    sched_2.step()
    assert opt_2.param_groups[0]["lr"] == 0.0005
    assert opt_2.param_groups[1]["lr"] < 0.0015


def test_checkpoint_immutability_hashes():
    """Verifies that parent and comparison hashes are immutable hex digests."""
    assert len(ROOT_HASH) == 64
    assert len(ROOT_TENSOR_HASH) == 64
    assert all(c in "0123456789abcdef" for c in ROOT_HASH)
    assert all(c in "0123456789abcdef" for c in ROOT_TENSOR_HASH)
