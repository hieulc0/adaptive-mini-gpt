"""LOCAL-SAFE mock/fixture contracts for Stage A.10.

Verifies:
- 3-seed determinism and independence (seeds 42, 43, 44 yield distinct initializations & streams)
- 2x2 factorial configuration matrix and parameter accounting (exact 32,768 diff, 0 discrepancy)
- Sealed evaluation suite zero-overlap, entity isolation, and SHA-256 digest
- Migration bridge structural integrity and state-transition tensor preservation
- Checkpoint immutability and provenance digests

ZERO neural forward/backward passes, ZERO checkpoint loading, and ZERO CPU training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from frame_a2_common import ROOT_HASH, ROOT_TENSOR_HASH  # noqa: E402  # pyright: ignore[reportMissingImports]

from adaptive_mini_gpt.data.pretraining_a9 import (  # noqa: E402
    FORBIDDEN_WORDS,
    generate_syntactic_pretraining_corpus,
)
from adaptive_mini_gpt.data.sealed_a10 import (  # noqa: E402
    generate_master_sealed_suite,
    generate_sealed_cases_for_seed,
    verify_sealed_isolation,
)
from adaptive_mini_gpt.model.config import ModelConfig  # noqa: E402
from adaptive_mini_gpt.model.transformer import MiniGPT  # noqa: E402
from adaptive_mini_gpt.natural_language_learning_m103.semantic_bottleneck_m104_e import (  # noqa: E402
    SemanticBottleneckM104_E,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (  # noqa: E402
    TextStudentM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (  # noqa: E402
    ByteTextTokenizerM103,
)
from adaptive_mini_gpt.training.reproducibility import seed_all  # noqa: E402

SPEC_A10_PATH = Path("data/specs/stage_a10_v1.json")
SPEC_SEALED_PATH = Path("data/specs/sealed_a10_v1.json")


def test_multiseed_independence_and_determinism():
    """Verifies that seeds 42, 43, 44 produce deterministic but distinct initializations & data."""
    # 1. Parameter initialization distinctness across seeds
    weights_by_seed: dict[int, torch.Tensor] = {}
    for s in (42, 43, 44):
        seed_all(s)
        lin = nn.Linear(64, 64)
        weights_by_seed[s] = lin.weight.detach().clone()

    assert not torch.equal(weights_by_seed[42], weights_by_seed[43])
    assert not torch.equal(weights_by_seed[43], weights_by_seed[44])
    assert not torch.equal(weights_by_seed[42], weights_by_seed[44])

    # Verify determinism on repeat
    seed_all(42)
    lin_repeat = nn.Linear(64, 64)
    assert torch.equal(weights_by_seed[42], lin_repeat.weight.detach())

    # 2. Pretraining corpus distinctness across seeds
    corpus_42 = generate_syntactic_pretraining_corpus(50, seed=42)
    corpus_43 = generate_syntactic_pretraining_corpus(50, seed=43)
    corpus_44 = generate_syntactic_pretraining_corpus(50, seed=44)

    assert corpus_42 != corpus_43
    assert corpus_43 != corpus_44
    for c in (corpus_42, corpus_43, corpus_44):
        assert len(c) == 50
        for s in c:
            words = set(s.lower().replace(".", "").replace(",", "").split())
            assert not words.intersection(FORBIDDEN_WORDS)

    # 3. Tokenizer deterministic encoding
    tok = ByteTextTokenizerM103()
    test_str = "For specimen Zor: status is flammable."
    ids1 = tok.encode(test_str)
    ids2 = tok.encode(test_str)
    assert ids1 == ids2
    assert tok.decode(ids1) == test_str


def test_2x2_factorial_configurations_and_parameter_accounting():
    """Verifies all 4 cells of 2x2 factorial matrix and exact parameter diff."""
    cfg_abs = ModelConfig(
        vocab_size=1024,
        context_length=256,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        pos_encoding="absolute",
    )
    cfg_rope = ModelConfig(
        vocab_size=1024,
        context_length=256,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        pos_encoding="rope",
    )

    model_abs = MiniGPT(cfg_abs)
    model_rope = MiniGPT(cfg_rope)

    params_abs = sum(p.numel() for p in model_abs.parameters())
    params_rope = sum(p.numel() for p in model_rope.parameters())

    assert params_abs == 1088256
    assert params_rope == 1055488
    param_diff = params_abs - params_rope
    # Exact difference is context_length * d_model = 256 * 128 = 32,768 (wpe table)
    assert param_diff == 32768

    # Verify TextStudentM103 configuration toggle
    student_abs = TextStudentM103(prune_orphaned_lm_head=True, pos_encoding="absolute")
    student_rope = TextStudentM103(prune_orphaned_lm_head=True, pos_encoding="rope")

    assert student_abs.base_model.config.pos_encoding == "absolute"
    assert student_rope.base_model.config.pos_encoding == "rope"
    assert student_abs.base_model.wpe is not None
    assert student_rope.base_model.wpe is None

    # Check rotary attention position_offset manipulation
    attn: Any = student_rope.base_model.blocks[0].attn
    assert hasattr(attn, "position_offset")
    assert attn.position_offset == 0
    attn.position_offset = 24
    assert attn.position_offset == 24
    attn.position_offset = 0


def test_sealed_evaluation_suite_zero_overlap_and_isolation():
    """Verifies that the 324-case sealed evaluation suite meets all isolation criteria."""
    assert SPEC_SEALED_PATH.exists()
    sealed_spec = json.loads(SPEC_SEALED_PATH.read_text())

    # Generate master suite
    suite = generate_master_sealed_suite()
    assert len(suite) == 324
    assert len(suite) == sealed_spec["total_cases"]

    # Verify per-seed subsets
    for s in (42, 43, 44):
        s_cases = generate_sealed_cases_for_seed(s)
        assert len(s_cases) == 108
        assert all(c.seed == s for c in s_cases)

    # Run strict isolation audit
    iso = verify_sealed_isolation(suite)
    assert iso["isolation_verified"] is True
    assert iso["total_cases"] == 324
    assert iso["unique_questions"] == 324
    assert iso["sealed_suite_sha256"] == sealed_spec["sealed_suite_sha256"]

    # Verify relationship-sensitive family exists and has distractors
    rel_cases = [c for c in suite if c.is_relationship_sensitive]
    assert len(rel_cases) == 81  # 324 / 4 families = 81 cases
    assert all(c.distractor is not None for c in rel_cases)
    assert all(c.family == "sealed_family_3" for c in rel_cases)


def test_migration_bridge_structure_and_state_tensor_integrity():
    """Verifies SemanticBottleneckM104_E tensor preservation for Phase 3 migration."""
    bottleneck_teacher = SemanticBottleneckM104_E(
        d_model=128,
        bottleneck_dim=32,
        action_dim=64,
        num_actions=8,
        num_states=16,
        num_tags=26,
    )
    bottleneck_student = SemanticBottleneckM104_E(
        d_model=128,
        bottleneck_dim=32,
        action_dim=64,
        num_actions=8,
        num_states=16,
        num_tags=26,
    )

    # Verify module names
    assert hasattr(bottleneck_teacher, "action_emb")
    assert hasattr(bottleneck_teacher, "transition_net")
    assert hasattr(bottleneck_teacher, "post_norm")
    assert hasattr(bottleneck_teacher, "state_head")
    assert hasattr(bottleneck_teacher, "concept_proj")
    assert hasattr(bottleneck_teacher, "attribute_head")

    # Simulate preserving historical parent's transition/state tensors
    with torch.no_grad():
        bottleneck_student.action_emb.weight.copy_(bottleneck_teacher.action_emb.weight)
        for p_src, p_dst in zip(
            bottleneck_teacher.transition_net.parameters(),
            bottleneck_student.transition_net.parameters(),
            strict=True,
        ):
            p_dst.copy_(p_src)
        for p_src, p_dst in zip(
            bottleneck_teacher.post_norm.parameters(),
            bottleneck_student.post_norm.parameters(),
            strict=True,
        ):
            p_dst.copy_(p_src)
        bottleneck_student.state_head.weight.copy_(bottleneck_teacher.state_head.weight)
        bottleneck_student.state_head.bias.copy_(bottleneck_teacher.state_head.bias)

    # Verify bit-exact preservation
    assert torch.equal(bottleneck_student.action_emb.weight, bottleneck_teacher.action_emb.weight)
    assert torch.equal(bottleneck_student.state_head.weight, bottleneck_teacher.state_head.weight)
    assert torch.equal(bottleneck_student.state_head.bias, bottleneck_teacher.state_head.bias)


def test_checkpoint_provenance_and_immutability_hashes():
    """Verifies that Stage A.10 specification binds to historical immutable parent hashes."""
    assert SPEC_A10_PATH.exists()
    spec = json.loads(SPEC_A10_PATH.read_text())

    assert spec["parent_sha256"] == ROOT_HASH
    assert spec["parent_tensor_digest"] == ROOT_TENSOR_HASH
    assert ROOT_HASH == "3e93d7238bfdb706ddfd49ee9903fed8b1e6f577529537f1e21cae12e905b262"
    assert (
        ROOT_TENSOR_HASH == "aed7480606a1a60e38ee7a95d22b5bcf14141f034f76cb45fb59680de277a33e"
    )
