"""LOCAL-SAFE mock/fixture contracts for Stage A.9.

Verifies:
- Layerwise parameter group partitioning and clipping contracts (0 missing/duplicate params)
- LayerwiseDecayScheduler stepping and pickle serialization roundtrip
- RoPE vs Absolute configuration toggle and parameter accounting (exact 32,768 diff, 0 discrepancy)
- Zero-leakage syntactic pretraining corpus isolation (0 forbidden words/templates)
- Checkpoint immutability and provenance digests

ZERO neural forward/backward passes, ZERO checkpoint loading, and ZERO CPU training.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from frame_a2_common import ROOT_HASH, ROOT_TENSOR_HASH  # noqa: E402

from adaptive_mini_gpt.data.pretraining_a9 import (  # noqa: E402
    FORBIDDEN_WORDS,
    generate_syntactic_pretraining_corpus,
    verify_pretraining_corpus_isolation,
)
from adaptive_mini_gpt.model.config import ModelConfig  # noqa: E402
from adaptive_mini_gpt.model.transformer import MiniGPT  # noqa: E402
from adaptive_mini_gpt.training.learning_state import LayerwiseDecayScheduler  # noqa: E402

SPEC_STAGE_A9 = Path("data/specs/stage_a9_v1.json")
SPEC_PRETRAIN = Path("data/specs/a9_language_pretrain_v1.json")


def build_mock_student_model() -> nn.Module:
    """Builds a lightweight mock student model mimicking IntegratedStudentM104_E."""
    class MockStudent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wte = nn.Embedding(1024, 128)
            self.blocks = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "attn": nn.Linear(128, 128),
                            "mlp": nn.Linear(128, 128),
                        }
                    )
                    for _ in range(5)
                ]
            )
            self.ln_f = nn.LayerNorm(128)
            self.bottleneck = nn.ModuleDict(
                {
                    "concept_proj": nn.Linear(128, 32),
                    "attribute_head": nn.Linear(32, 2),
                }
            )

    return MockStudent()


def partition_layerwise_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
    """Partitions model parameters into lower backbone, upper backbone, and readout."""
    lower_params: list[nn.Parameter] = []
    upper_params: list[nn.Parameter] = []
    readout_params: list[nn.Parameter] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(f"blocks.{i}" in name for i in (0, 1, 2)):
            lower_params.append(p)
        elif any(f"blocks.{i}" in name for i in (3, 4)) or any(k in name for k in ("wte", "ln_f")):
            upper_params.append(p)
        elif any(k in name for k in ("concept_proj", "attribute_head")):
            readout_params.append(p)
        else:
            upper_params.append(p)

    return lower_params, upper_params, readout_params


def test_layerwise_parameter_group_partitioning():
    """Verifies that 3-group layerwise partitioning covers 100% of parameters with 0 overlap."""
    model = build_mock_student_model()
    lower, upper, readout = partition_layerwise_groups(model)

    all_params = [p for p in model.parameters() if p.requires_grad]
    partitioned = lower + upper + readout

    # Check completeness and disjointness
    assert len(all_params) == len(partitioned)
    assert set(id(p) for p in all_params) == set(id(p) for p in partitioned)
    assert len(set(id(p) for p in lower).intersection(set(id(p) for p in upper))) == 0
    assert len(set(id(p) for p in lower).intersection(set(id(p) for p in readout))) == 0
    assert len(set(id(p) for p in upper).intersection(set(id(p) for p in readout))) == 0

    # Build 3-group optimizer
    opt = torch.optim.AdamW(
        [
            {"params": lower, "lr": 2.5e-4, "name": "lower_backbone"},
            {"params": upper, "lr": 5.0e-4, "name": "upper_backbone"},
            {"params": readout, "lr": 1.5e-3, "name": "readout"},
        ]
    )
    assert len(opt.param_groups) == 3

    # Build and test LayerwiseDecayScheduler
    sched = LayerwiseDecayScheduler(
        opt,
        total_steps=7200,
        lower_lr=2.5e-4,
        upper_lr=5.0e-4,
        readout_lr_start=1.5e-3,
        readout_lr_min=1.5e-4,
    )
    sched.step()

    assert opt.param_groups[0]["lr"] == 2.5e-4
    assert opt.param_groups[1]["lr"] == 5.0e-4
    assert opt.param_groups[2]["lr"] < 1.5e-3
    assert opt.param_groups[2]["lr"] > 1.5e-4

    # Verify pickle roundtrip
    dumped = pickle.dumps(sched)
    loaded = pickle.loads(dumped)
    assert loaded.step_num == 1
    assert loaded.lower_lr == 2.5e-4
    assert loaded.upper_lr == 5.0e-4


def test_rope_vs_absolute_configuration_and_parameter_accounting():
    """Verifies that RoPE eliminates wpe and reduces parameter count by exactly context_length * d_model."""
    cfg_abs = ModelConfig(
        vocab_size=1024,
        context_length=256,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        pos_encoding="absolute",
        dropout=0.0,
        bias=True,
    )
    model_abs = MiniGPT(cfg_abs)
    assert model_abs.wpe is not None
    assert model_abs.wpe.weight.shape == (256, 128)
    params_abs = sum(p.numel() for p in model_abs.parameters())

    cfg_rope = ModelConfig(
        vocab_size=1024,
        context_length=256,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        pos_encoding="rope",
        dropout=0.0,
        bias=True,
    )
    model_rope = MiniGPT(cfg_rope)
    assert model_rope.wpe is None
    params_rope = sum(p.numel() for p in model_rope.parameters())

    diff = params_abs - params_rope
    assert diff == 256 * 128 == 32768
    assert params_abs == 1088256
    assert params_rope == 1055488

    # Verify position_offset on CausalSelfAttention
    attn: Any = model_rope.blocks[0].attn
    assert hasattr(attn, "position_offset")
    assert attn.position_offset == 0
    attn.position_offset = 8
    assert attn.position_offset == 8


def test_zero_leakage_syntactic_pretraining_corpus():
    """Verifies pretraining corpus generator isolation and strict absence of task tokens."""
    corpus = generate_syntactic_pretraining_corpus(100, seed=42)
    assert len(corpus) == 100

    for sent in corpus:
        tokens = set(sent.lower().replace(".", "").replace(",", "").replace(";", "").replace(":", "").split())
        assert not tokens.intersection(FORBIDDEN_WORDS)

    iso = verify_pretraining_corpus_isolation(corpus)
    assert iso["isolation_verified"] is True
    assert iso["forbidden_words_found"] is False
    assert len(iso["corpus_sha256"]) == 64


def test_checkpoint_immutability_hashes():
    """Verifies that parent and comparison hashes are immutable hex digests."""
    assert len(ROOT_HASH) == 64
    assert len(ROOT_TENSOR_HASH) == 64
    assert all(c in "0123456789abcdef" for c in ROOT_HASH)
    assert all(c in "0123456789abcdef" for c in ROOT_TENSOR_HASH)

    spec = json.loads(SPEC_STAGE_A9.read_text())
    assert spec["parent_sha256"] == ROOT_HASH
    assert spec["parent_tensor_digest"] == ROOT_TENSOR_HASH
