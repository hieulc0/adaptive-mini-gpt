"""Unit and regression tests for Milestone M74: Verified Neural Weight Learning and High-Information Generalization."""

from __future__ import annotations

import pytest
import torch

from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.neural_weight_learning import (
    CausalWeightAblationBenchmark,
    M73ScientificAuditor,
    M74SafeguardsAuditor,
    NeuralPathwayInstrumenter,
    ProceduralRuleFamily,
    ProceduralTaskGenerator,
    Rank16SequenceAdapter,
    apply_procedural_rule,
)


@pytest.fixture
def tiny_model() -> MiniGPT:
    """Create deterministic tiny MiniGPT model for unit testing."""
    torch.manual_seed(42)
    cfg = ModelConfig(
        vocab_size=128,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        context_length=64,
        bias=True,
    )
    return MiniGPT(cfg)


def test_m73_audit_and_parameter_lineage() -> None:
    """Verify Phase 1: Tensor-level parameter derivation (928,512 -> 974,658) and M73 findings."""
    lineage = M73ScientificAuditor.audit_parameter_lineage()
    assert lineage["verified"] is True
    assert lineage["total_orig_parameters"] == 928512
    assert lineage["total_expanded_parameters"] == 974658
    assert lineage["total_delta"] == 46146
    assert lineage["wte_delta"] == 22784
    assert lineage["lm_head_weight_delta"] == 22784
    assert lineage["lm_head_bias_delta"] == 578

    audit = M73ScientificAuditor.audit_m73_results()
    assert audit["audit_passed"] is True
    assert len(audit["records"]) == 3
    assert audit["t_critical"] > 4.0


def test_procedural_task_generation_and_split_isolation() -> None:
    """Verify Phase 4 & 5: Procedural task rules and strict disjoint split isolation."""
    gen = ProceduralTaskGenerator(vocab_size=16)

    # Test rule correctness
    assert apply_procedural_rule(
        ProceduralRuleFamily.CAESAR_SHIFT_MOD3, (1, 5, 14), vocab_size=16
    ) == (4, 8, 1)
    assert apply_procedural_rule(
        ProceduralRuleFamily.CAESAR_SHIFT_MOD5, (1, 5, 14), vocab_size=16
    ) == (6, 10, 3)
    assert apply_procedural_rule(
        ProceduralRuleFamily.AFFINE_TRANSFORM, (0, 1, 2), vocab_size=16
    ) == (1, 4, 7)
    assert apply_procedural_rule(
        ProceduralRuleFamily.COMPLEMENT_REFLECT, (0, 15), vocab_size=16
    ) == (15, 0)
    assert apply_procedural_rule(ProceduralRuleFamily.NONLINEAR_SBOX, (0, 1, 2), vocab_size=16) == (
        7,
        12,
        14,
    )

    # Generate task and verify disjoint splits
    task = gen.generate_task(
        task_id="test_shift",
        rule_family=ProceduralRuleFamily.CAESAR_SHIFT_MOD3,
        seed=42,
        seq_len=3,
        n_train=20,
        n_dev=10,
        n_qual=10,
        n_test=20,
    )

    train_inputs = {ex.input_tokens for ex in task.train_examples}
    dev_inputs = {ex.input_tokens for ex in task.dev_examples}
    qual_inputs = {ex.input_tokens for ex in task.qualification_examples}
    test_inputs = {ex.input_tokens for ex in task.sealed_test_examples}

    assert len(train_inputs & dev_inputs) == 0
    assert len(train_inputs & qual_inputs) == 0
    assert len(train_inputs & test_inputs) == 0
    assert len(dev_inputs & qual_inputs) == 0
    assert len(dev_inputs & test_inputs) == 0
    assert len(qual_inputs & test_inputs) == 0


def test_sequence_adapter_and_pathway_instrumentation(tiny_model: MiniGPT) -> None:
    """Verify Phase 2: Forward hook execution, gradient norms, and weight update tracking."""
    adapter = Rank16SequenceAdapter(d_model=128, bottleneck_dim=16, num_symbols=16)
    assert adapter.count_parameters() == 2592

    gen = ProceduralTaskGenerator(vocab_size=16)
    task = gen.generate_task(
        task_id="test_instrument",
        rule_family=ProceduralRuleFamily.CAESAR_SHIFT_MOD3,
        seed=42,
        n_train=16,
        n_dev=8,
        n_qual=8,
        n_test=8,
    )

    trained_ad, audit = NeuralPathwayInstrumenter.train_and_instrument(
        model=tiny_model,
        adapter=adapter,
        task=task,
        adapter_id="test_ad",
        device=torch.device("cpu"),
        steps=10,
    )

    assert audit.verified_executed is True
    assert audit.trainable_params_count == 2592
    assert audit.weight_delta_norm > 0.0
    assert audit.total_grad_norm > 0.0
    assert audit.evaluation_verified is True


def test_causal_weight_ablation_8_arms(tiny_model: MiniGPT) -> None:
    """Verify Phase 3: 8-arm weight ablation showing TRAINED decisively beats SHUFFLED and RESET."""
    gen = ProceduralTaskGenerator(vocab_size=16)
    task = gen.generate_task(
        task_id="target_shift",
        rule_family=ProceduralRuleFamily.CAESAR_SHIFT_MOD3,
        seed=42,
        n_train=48,
        n_dev=16,
        n_qual=16,
        n_test=16,
    )
    unrelated_task = gen.generate_task(
        task_id="unrelated_affine",
        rule_family=ProceduralRuleFamily.AFFINE_TRANSFORM,
        seed=99,
        n_train=48,
        n_dev=16,
        n_qual=16,
        n_test=16,
    )

    benchmark_res = CausalWeightAblationBenchmark.run_benchmark(
        model=tiny_model,
        task=task,
        unrelated_task=unrelated_task,
        device=torch.device("cpu"),
        seed=42,
    )

    assert benchmark_res["attribution_verified"] is True
    assert benchmark_res["trained_exact_match"] >= 0.80
    assert benchmark_res["shuffled_exact_match"] <= 0.10
    assert benchmark_res["exact_match_delta_vs_shuffled"] >= 0.70
    assert len(benchmark_res["results"]) == 8


def test_safeguards_carry_logic_and_rollback(tiny_model: MiniGPT) -> None:
    """Verify Phase 8: 100% carry arithmetic, 100% logic probes, double hashes, and bit-exact rollback."""
    # Carry arithmetic
    carry_acc = M74SafeguardsAuditor.verify_arithmetic_carry_retention()
    assert carry_acc == 1.0

    # Chained operational logic
    logic_acc = M74SafeguardsAuditor.verify_chained_operational_logic()
    assert logic_acc == 1.0

    # Double hashes
    h1 = M74SafeguardsAuditor.compute_unnamed_param_hash(tiny_model)
    h2 = M74SafeguardsAuditor.compute_named_param_hash(tiny_model)
    assert M74SafeguardsAuditor.verify_base_invariance(tiny_model, h1, h2) is True

    # Bit-exact rollback
    adapter = Rank16SequenceAdapter()
    assert M74SafeguardsAuditor.verify_bit_exact_rollback(adapter) is True

    # Capacity comparison
    cap = M74SafeguardsAuditor.compare_capacity_variants(d_model=128, num_symbols=16)
    assert cap["rank16_parameters"] == 2592
    assert cap["rank32_parameters"] == 4912
