"""Unit tests for M34: Modular Continual Learning and Verified Checkpoint Growth."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from adaptive_mini_gpt.learning.modular_cumulative_learner import (
    ModularReasoner,
    evaluate_multi_domain_system,
)
from adaptive_mini_gpt.learning.modular_lineage import (
    ModularLineageManager,
    compute_adapter_hash,
)
from adaptive_mini_gpt.model.modular_adapter import (
    MultiAdapterLinear,
    add_domain_adapter,
    compute_base_model_hash,
    freeze_for_adapter_training,
    get_adapter_state_dict,
    inject_multi_domain_adapters,
    load_adapter_state_dict,
    set_active_domain_adapter,
)
from adaptive_mini_gpt.model.transformer import MiniGPT, ModelConfig


def test_multi_adapter_linear_mechanics() -> None:
    """Test MultiAdapterLinear freezing, zero-init identity, and active adapter switching."""
    torch.manual_seed(42)
    base_linear = nn.Linear(16, 16)
    orig_weight = base_linear.weight.clone()

    wrapped = MultiAdapterLinear(base_linear)
    assert not wrapped.base_linear.weight.requires_grad

    # Test pure base forward
    x = torch.randn(2, 16)
    out_base = wrapped(x)
    assert torch.allclose(out_base, base_linear(x))

    # Add adapter "arithmetic"
    wrapped.add_adapter("arithmetic", r=4, lora_alpha=8.0)
    wrapped.set_active_adapter("arithmetic")

    # Due to B zero-initialization, output at init is exactly equal to base
    out_adapt = wrapped(x)
    assert torch.allclose(out_adapt, out_base)

    # Add adapter "logic"
    wrapped.add_adapter("logic", r=4, lora_alpha=8.0)
    wrapped.set_active_adapter("logic")
    assert torch.allclose(wrapped(x), out_base)


def test_mathematical_base_invariance_during_adapter_training() -> None:
    """Verify base parameter hash remains 100% mathematically invariant when training adapters."""
    torch.manual_seed(42)
    cfg = ModelConfig(
        vocab_size=64,
        context_length=32,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
    )
    model = MiniGPT(cfg)
    inject_multi_domain_adapters(model)

    base_hash_init = compute_base_model_hash(model)

    add_domain_adapter(model, "arithmetic", r=4)
    trainable, frozen = freeze_for_adapter_training(model, "arithmetic")
    assert trainable > 0
    assert frozen > 0

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)

    # Run dummy training steps on adapter
    for _ in range(5):
        optimizer.zero_grad()
        dummy_in = torch.randint(0, 64, (2, 8))
        logits, _ = model(dummy_in)
        loss = logits.sum()
        loss.backward()
        optimizer.step()

    # Base hash must remain identical
    base_hash_after = compute_base_model_hash(model)
    assert base_hash_init == base_hash_after, "Base parameter weights were modified!"


def test_adapter_state_dict_save_and_load() -> None:
    """Test extracting, resetting, and loading adapter state dicts."""
    torch.manual_seed(42)
    cfg = ModelConfig(
        vocab_size=64,
        context_length=32,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
    )
    model = MiniGPT(cfg)
    inject_multi_domain_adapters(model)
    add_domain_adapter(model, "logic", r=4)

    # Mutate adapter parameters
    for p in model.parameters():
        if p.requires_grad:
            p.data.add_(torch.randn_like(p))

    state = get_adapter_state_dict(model, "logic")
    h1 = compute_adapter_hash(state)

    # Reset parameters
    for p in model.parameters():
        if p.requires_grad:
            p.data.zero_()

    state_zero = get_adapter_state_dict(model, "logic")
    assert compute_adapter_hash(state_zero) != h1

    # Load back saved state
    load_adapter_state_dict(model, "logic", state)
    state_restored = get_adapter_state_dict(model, "logic")
    assert compute_adapter_hash(state_restored) == h1


def test_modular_lineage_manager() -> None:
    """Test ModularLineageManager foundation registration, candidate promotion, and rollback."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = Path(tmp_dir)
        mgr = ModularLineageManager(seed=42, storage_dir=storage)

        cfg = ModelConfig(
            vocab_size=64,
            context_length=32,
            d_model=32,
            n_layers=2,
            n_heads=2,
            d_ff=64,
        )
        model = MiniGPT(cfg)
        inject_multi_domain_adapters(model)

        rec_root = mgr.register_foundation(model, foundation_id="V1", description="Test Foundation")
        assert rec_root.version_id == "V1"
        assert mgr.active_version_id == "V1"

        # Candidate 1: Arithmetic Adapter (Promoted)
        add_domain_adapter(model, "arithmetic", r=4)
        adapt_state = get_adapter_state_dict(model, "arithmetic")
        rec_cand = mgr.create_candidate(
            candidate_id="candidate_V2",
            episode_id="ep_arith",
            domain="arithmetic",
            adapter_state=adapt_state,
            base_model=model,
        )
        assert rec_cand.status == "CANDIDATE"

        rec_qual = mgr.promote_candidate(
            candidate_id="candidate_V2",
            qualified_id="qualified_V2",
            metrics={"arithmetic_exact": 0.9},
            rationale="Passed arithmetic gate",
        )
        assert rec_qual.status == "QUALIFIED"
        assert mgr.active_version_id == "qualified_V2"

        # Candidate 2: Logic Adapter (Rolled back)
        add_domain_adapter(model, "logic", r=4)
        adapt_state_logic = get_adapter_state_dict(model, "logic")
        mgr.create_candidate(
            candidate_id="candidate_V3",
            episode_id="ep_logic",
            domain="logic",
            adapter_state=adapt_state_logic,
            base_model=model,
        )
        rec_rb = mgr.rollback_candidate(
            candidate_id="candidate_V3",
            metrics={"logic_exact": 0.2},
            rationale="Failed logic gate",
        )
        assert rec_rb.status == "ROLLED_BACK"
        assert mgr.active_version_id == "qualified_V2"

        manifest_file = storage / "modular_manifest.json"
        mgr.save_manifest_json(manifest_file)
        assert manifest_file.exists()
