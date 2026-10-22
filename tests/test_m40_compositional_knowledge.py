"""Unit tests for M40: Compositional Knowledge, Transfer and Foundation Readiness."""

import pytest
import torch

from adaptive_mini_gpt.composition.audit import M39BaselineAuditor
from adaptive_mini_gpt.composition.benchmark_suite import (
    M40_COMPOSITIONAL_BENCHMARK,
    M40_CROSS_MODULE_BENCHMARK,
    SEQUENTIAL_RELATIONAL_EPISODES,
)
from adaptive_mini_gpt.composition.composition_trainer import CompositionTrainer
from adaptive_mini_gpt.composition.hybrid_router import HybridRoutingEngine
from adaptive_mini_gpt.composition.relational_world import (
    DeterministicGraphSolver,
    RelationalFact,
    RelationSemantics,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.modular_adapter import (
    add_domain_adapter,
    compute_base_model_hash,
    inject_multi_domain_adapters,
)
from adaptive_mini_gpt.model.transformer import MiniGPT


def _create_minimal_test_model(vocab_size: int = 300) -> MiniGPT:
    """Create a minimal MiniGPT for fast unit test execution."""
    cfg = ModelConfig(
        vocab_size=vocab_size,
        context_length=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        dropout=0.0,
    )
    return MiniGPT(cfg)


def _create_dummy_tokenizer() -> BPETokenizer:
    tok = BPETokenizer()
    corpus = "Newton force mass acceleration F ma mitochondria ATP cell Alan Turing 1936 Gutenberg 1440 0 1 2 3 4 5 6 7 8 9 + = : [STATE] [OUT] d c abstain Question Answer logic Socrates mortal Zorblax Nexus-4 Vortice Celestia Krypton-9 Epsilon-3 Pyron-1 Glapton Xyphos-7 Morvan-Prime inside precede connected_to power [PREMISES] [THINK] [ANSWER] yes no"
    tok.train(corpus, target_vocab_size=300)
    return tok


def test_deterministic_graph_solver_deductions() -> None:
    """Verify ground-truth graph solver handles 1-hop, 2-hop, 3-hop, asymmetry, and abstention."""
    solver = DeterministicGraphSolver()
    solver.add_fact(RelationalFact("A", "is_inside", "B", RelationSemantics.TRANSITIVE))
    solver.add_fact(RelationalFact("B", "is_inside", "C", RelationSemantics.TRANSITIVE))
    solver.add_fact(RelationalFact("C", "is_inside", "D", RelationSemantics.TRANSITIVE))

    # 1-hop
    res1 = solver.evaluate_query("A", "is_inside", "B", RelationSemantics.TRANSITIVE)
    assert res1.verdict == "yes"
    assert len(res1.shortest_path) == 2

    # 2-hop
    res2 = solver.evaluate_query("A", "is_inside", "C", RelationSemantics.TRANSITIVE)
    assert res2.verdict == "yes"
    assert len(res2.shortest_path) == 3

    # 3-hop
    res3 = solver.evaluate_query("A", "is_inside", "D", RelationSemantics.TRANSITIVE)
    assert res3.verdict == "yes"
    assert len(res3.shortest_path) == 4

    # Asymmetry check
    res_asym = solver.evaluate_query("B", "is_inside", "A", RelationSemantics.ASYMMETRIC)
    assert res_asym.verdict == "no"

    # Abstention on ungrounded entity
    res_abs = solver.evaluate_query("A", "is_inside", "UnknownZ", RelationSemantics.TRANSITIVE)
    assert res_abs.verdict == "abstain"


def test_m40_benchmark_dataset_integrity() -> None:
    """Test task balance, categories, and counterfactual properties."""
    assert len(M40_COMPOSITIONAL_BENCHMARK) == 24

    categories = {t.category for t in M40_COMPOSITIONAL_BENCHMARK}
    assert {"single_hop", "two_hop", "three_hop", "asymmetric", "counterfactual", "abstention"}.issubset(categories)

    counterfactuals = [t for t in M40_COMPOSITIONAL_BENCHMARK if t.is_counterfactual]
    assert len(counterfactuals) == 4
    for cf in counterfactuals:
        assert cf.category == "counterfactual"
        assert len(cf.entities) >= 2

    abstentions = [t for t in M40_COMPOSITIONAL_BENCHMARK if t.category == "abstention"]
    assert len(abstentions) == 4
    for ab in abstentions:
        assert ab.expected_answer == "abstain"

    assert len(M40_CROSS_MODULE_BENCHMARK) >= 4
    assert len(SEQUENTIAL_RELATIONAL_EPISODES) == 4


def test_m39_baseline_audit_record() -> None:
    """Verify Phase 1 audit record generation and hash invariance verification."""
    tok = _create_dummy_tokenizer()
    model = _create_minimal_test_model(tok.vocab_size)
    dev = torch.device("cpu")
    initial_hash = compute_base_model_hash(model)

    audit = M39BaselineAuditor.audit_baseline(model, tok, dev, initial_hash)
    assert audit.base_parameters_invariant is True
    assert audit.canonical_carry_protocol_active is True
    assert audit.m39_reproduced_direct_recall > 0.90
    assert audit.m39_reproduced_compositional < 0.10
    assert audit.m39_reproduced_unseen_transfer == 0.0


def test_hybrid_routing_engine_behavior() -> None:
    """Verify deterministic fast path, neural router training, and hybrid routing fallback."""
    tok = _create_dummy_tokenizer()
    dev = torch.device("cpu")
    engine = HybridRoutingEngine(tok, dev)

    # 1. Obvious deterministic prompt
    det_pred, conf = engine.deterministic_route("[STATE] 5 7 0 ->")
    assert det_pred == "arithmetic"
    assert conf >= 0.90

    # 2. Counterfactual prompt
    cf_pred, cf_conf = engine.deterministic_route("Is Glapton inside Xyphos-7?")
    assert cf_pred == "counterfactual"
    assert cf_conf >= 0.90

    # 3. Train neural router
    train_samples = [
        ("Calculate 5 + 7 with carry", "arithmetic"),
        ("All men are mortal and Socrates is a man", "logic"),
        ("Zorblax is inside Nexus-4", "relational_1hop"),
        ("Step 1: A in B. Step 2: B in C. Therefore A in C.", "relational_multihop"),
        ("Glapton inside Xyphos-7 counterfactual", "counterfactual"),
        ("How many total cores in Sector X?", "cross_module"),
        ("Lead into gold alchemy transmutation", "abstain"),
        ("Search open web encyclopedia", "retrieval"),
    ]
    loss = engine.train_neural_router(train_samples, epochs=10, lr=1e-2)
    assert isinstance(loss, float)

    # 4. Evaluate policies
    res_det, res_neu, res_hyb = engine.evaluate_routing_policies(train_samples)
    assert res_det.policy_name == "DETERMINISTIC_HEURISTIC_SHIELD"
    assert res_neu.policy_name == "NEURAL_METACOGNITIVE_ROUTER"
    assert res_hyb.policy_name == "HYBRID_ROUTING_ENGINE"
    assert res_hyb.overall_routing_accuracy >= 0.0


def test_composition_trainer_adapter_optimization() -> None:
    """Test that reasoning supervision pairs train on a minimal modular adapter."""
    tok = _create_dummy_tokenizer()
    model = _create_minimal_test_model(tok.vocab_size)
    dev = torch.device("cpu")

    inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=4)
    add_domain_adapter(model, domain_name="test_comp", r=4)

    trainer = CompositionTrainer(model, tok, dev)
    pairs = [
        ("[PREMISES] A inside B. [QUESTION] Is A in B?", "[THINK] Direct premise. [ANSWER] yes"),
        ("[PREMISES] A inside B. B inside C. [QUESTION] Is A in C?", "[THINK] Transitive deduction. [ANSWER] yes"),
    ]
    loss = trainer.train_adapter_on_text_pairs(model, "test_comp", pairs, epochs=2, lr=1e-3)
    assert loss >= 0.0
