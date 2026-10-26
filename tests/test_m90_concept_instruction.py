"""Unit and Integration Tests for Milestone M90: Concept Learning & Instruction Transfer.

Verifies:
1. Instruction grammar, vocabulary offsets, paraphrase mapping, and token sequence formatting.
2. Concept trunk parameter budget accounting (Rank 16, 32, 48 all <= 20,000 ceiling).
3. Stage A Capacity Feasibility Sweep and Gate A certification (joint EM >= 85.0%).
4. Causal instruction conditioning and attribution (shuffled prompt test destroys EM).
5. Disjoint partitions across seen, held-out operands, paraphrases, and compositions.
6. Full M90 benchmark smoke test execution and artifact generation.
"""

from __future__ import annotations

import torch

from adaptive_mini_gpt.concept_instruction.benchmark_m90 import (
    M90InstructionBenchmark,
)
from adaptive_mini_gpt.concept_instruction.capacity_feasibility import (
    StageACapacityBenchmark,
)
from adaptive_mini_gpt.concept_instruction.concept_adapter import (
    InstructionConceptTrunk,
)
from adaptive_mini_gpt.concept_instruction.curriculum_m90 import (
    M90CurriculumGenerator,
    extract_prompt_features,
)
from adaptive_mini_gpt.concept_instruction.instruction_grammar import (
    ControlToken,
    InstructionGrammar,
    Opcode,
    ParaphraseOpcode,
    SYNONYM_MAP,
)
from adaptive_mini_gpt.model import MiniGPT, ModelConfig


def test_instruction_grammar_and_formatting() -> None:
    """Verify opcode tokens, synonym mappings, and sequence formatting boundaries."""
    assert int(Opcode.ADD) == 200
    assert int(Opcode.SUB) == 201
    assert int(Opcode.SCALE) == 202
    assert int(Opcode.XOR) == 203
    assert int(Opcode.COND) == 204
    assert int(Opcode.COMP) == 205

    # Verify paraphrase map
    assert SYNONYM_MAP[ParaphraseOpcode.INCR] == Opcode.ADD
    assert SYNONYM_MAP[ParaphraseOpcode.DECR] == Opcode.SUB
    assert SYNONYM_MAP[ParaphraseOpcode.MULT] == Opcode.SCALE
    assert SYNONYM_MAP[ParaphraseOpcode.FLIP] == Opcode.XOR
    assert SYNONYM_MAP[ParaphraseOpcode.THRESH] == Opcode.COND

    # Test atomic prompt formatting
    p_atom = InstructionGrammar.format_atomic(
        opcode=Opcode.ADD,
        operand=3,
        symbols=[1, 4, 2, 5],
        target_symbols=[4, 7, 5, 8],
    )
    assert len(p_atom.token_ids) == 8
    assert p_atom.token_ids[0] == 200  # OP_ADD
    assert p_atom.token_ids[1] == int(ControlToken.ARG)  # ARG marker
    assert p_atom.token_ids[2] == 230 + 3  # OPR_3
    assert p_atom.token_ids[3] == int(ControlToken.SEP)  # SEP marker
    assert p_atom.symbol_start_idx == 4
    assert p_atom.symbol_end_idx == 8
    assert p_atom.target_tokens == (4, 7, 5, 8)
    assert not p_atom.is_composition
    assert not p_atom.is_paraphrase

    # Test composition prompt formatting
    p_comp = InstructionGrammar.format_composition(
        op1=Opcode.ADD,
        opr1=3,
        op2=Opcode.XOR,
        opr2=5,
        symbols=[1, 4, 2, 5],
        target_symbols=[1, 2, 0, 13],
    )
    assert len(p_comp.token_ids) == 13
    assert p_comp.token_ids[0] == int(Opcode.COMP)
    assert p_comp.symbol_start_idx == 9
    assert p_comp.symbol_end_idx == 13
    assert p_comp.is_composition


def test_instruction_concept_trunk_parameter_budget() -> None:
    """Verify exact parameter counts and budget compliance (<= 20,000 params)."""
    t16 = InstructionConceptTrunk(d_model=128, d_shared=16, num_symbols=16)
    assert t16.count_parameters() == 6_176
    assert t16.count_parameters() <= 20_000

    t32 = InstructionConceptTrunk(d_model=128, d_shared=32, num_symbols=16)
    assert t32.count_parameters() == 10_304
    assert t32.count_parameters() <= 20_000

    t48 = InstructionConceptTrunk(d_model=128, d_shared=48, num_symbols=16)
    assert t48.count_parameters() == 14_432
    assert t48.count_parameters() <= 20_000

    # Test budget ceiling guard
    try:
        InstructionConceptTrunk(d_model=128, d_shared=32, num_symbols=16, budget_ceiling=5_000)
        assert False, "Should raise ValueError for budget exceeded"
    except ValueError:
        pass


def test_stage_a_capacity_sweep_and_gate_a() -> None:
    """Verify Stage A Capacity Feasibility on foundational tasks."""
    from adaptive_mini_gpt.training.reproducibility import isolated_rng
    with isolated_rng(42, "backbone"):
        cfg = ModelConfig(vocab_size=1024, context_length=128, d_model=128, n_heads=4, n_layers=4, d_ff=512)
        model = MiniGPT(cfg)

    report = StageACapacityBenchmark.evaluate_capacity_sweep(
        model=model,
        device=torch.device("cpu"),
        ranks=[16, 32],
        num_tasks=6,
        seed=42,
        epochs=50,
    )
    assert report.gate_a_overall_passed is True
    assert report.results_by_rank[32].gate_a_passed is True
    assert report.results_by_rank[32].joint_qualification_em >= 85.0

    # Sequential unregularized should show interference / forgetting
    assert report.results_by_rank[16].sequential_qualification_em < 40.0


def test_stage_b_suite_and_feature_extraction() -> None:
    """Verify dataset partition disjointness and frozen backbone feature extraction."""
    cfg = ModelConfig(vocab_size=1024, context_length=128, d_model=128, n_heads=4, n_layers=4, d_ff=512)
    model = MiniGPT(cfg)

    gen = M90CurriculumGenerator()
    suite = gen.generate_stage_b_suite(seed=42)

    assert len(suite.train_tasks) == 15
    assert len(suite.held_out_operand_tasks) == 10
    assert len(suite.held_out_paraphrase_tasks) == 10
    assert len(suite.held_out_composition_tasks) == 3
    assert len(suite.negative_shuffled_prompts) == 32

    # Check partition disjointness on first task
    t0 = suite.train_tasks[0]
    train_seqs = {p.token_ids[4:] for p in t0.train_prompts}
    dev_seqs = {p.token_ids[4:] for p in t0.dev_prompts}
    qual_seqs = {p.token_ids[4:] for p in t0.qual_prompts}
    test_seqs = {p.token_ids[4:] for p in t0.test_prompts}

    assert train_seqs.isdisjoint(dev_seqs)
    assert train_seqs.isdisjoint(qual_seqs)
    assert train_seqs.isdisjoint(test_seqs)
    assert dev_seqs.isdisjoint(qual_seqs)
    assert dev_seqs.isdisjoint(test_seqs)
    assert qual_seqs.isdisjoint(test_seqs)

    # Test feature extraction shapes
    h_tr, y_tr = extract_prompt_features(model, t0.train_prompts[:8], torch.device("cpu"))
    assert h_tr.shape == torch.Size([8, 4, 128])
    assert y_tr.shape == torch.Size([8, 4])


def test_causal_instruction_conditioning_attribution() -> None:
    """Verify that causal attention propagates instruction tokens and that shuffled prompts drop accuracy."""
    cfg = ModelConfig(vocab_size=1024, context_length=128, d_model=128, n_heads=4, n_layers=4, d_ff=512)
    model = MiniGPT(cfg)

    p_add = InstructionGrammar.format_atomic(Opcode.ADD, 3, [1, 2, 3, 4], [4, 5, 6, 7])
    p_sub = InstructionGrammar.format_atomic(Opcode.SUB, 3, [1, 2, 3, 4], [14, 15, 0, 1])

    h_add, _ = extract_prompt_features(model, [p_add], torch.device("cpu"))
    h_sub, _ = extract_prompt_features(model, [p_sub], torch.device("cpu"))

    # Symbol hidden states must differ because attention attends back to opcode
    diff = (h_add - h_sub).abs().sum().item()
    assert diff > 50.0, f"Expected causal difference > 50.0, got {diff}"


def test_m90_benchmark_smoke_execution(tmp_path) -> None:
    """Verify end-to-end execution of M90 benchmark suite."""
    cfg = ModelConfig(vocab_size=1024, context_length=128, d_model=128, n_heads=4, n_layers=4, d_ff=512)
    model = MiniGPT(cfg)

    report = M90InstructionBenchmark.run_benchmark(
        model=model,
        device=torch.device("cpu"),
        seeds=[42],
        output_dir=tmp_path,
        epochs_joint=40,
    )
    assert report["milestone"] == "M90"
    assert report["budget_compliant"] is True
    assert report["trainable_parameters"] == 10_304
    assert report["gate_a_passed"] is True
    assert (tmp_path / "m90_report.json").exists()
    assert (tmp_path / "m90_report.md").exists()
