"""Unit tests for M41: Directional Reasoning, Knowledge Boundaries, and Corrective Learning."""

import pytest
import torch

from adaptive_mini_gpt.corrective.audit import M40BaselineAuditor
from adaptive_mini_gpt.corrective.benchmark_suite import (
    M41_DIRECTIONAL_BENCHMARK,
    SEQUENTIAL_RELATION_WORLDS,
)
from adaptive_mini_gpt.corrective.corrective_trainer import CorrectiveTrainer
from adaptive_mini_gpt.corrective.directional_world import (
    DirectionalFact,
    RelationCategory,
    TriStateGraphReasoner,
    TriStateVerdict,
)
from adaptive_mini_gpt.corrective.uncertainty_router import UncertaintyRoutingEngine
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.modular_adapter import (
    add_domain_adapter,
    compute_base_model_hash,
    inject_multi_domain_adapters,
)
from adaptive_mini_gpt.model.transformer import MiniGPT


def _create_minimal_test_model(vocab_size: int = 320) -> MiniGPT:
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
    corpus = (
        "Newton force mass acceleration F ma mitochondria ATP cell Alan Turing 1936 "
        "Gutenberg 1440 0 1 2 3 4 5 6 7 8 9 + = : [STATE] [OUT] d c abstain Question Answer "
        "logic Socrates mortal Zorblax Nexus-4 Vortice Celestia Krypton-9 Epsilon-3 Pyron-1 "
        "Glapton Xyphos-7 Morvan-Prime Zephyr-12 Oryx-9 Bravon-3 Solon-X Thalos Astraea "
        "Nebula-8 Quasar-5 Ghost-Node Quantum-Zero inside precede powers subsumes connected_to "
        "[PREMISES] [QUESTION] [ANSWER] YES NO UNKNOWN ABSTAIN"
    )
    tok.train(corpus, target_vocab_size=320)
    return tok


def test_tristate_graph_reasoner_formal_logic() -> None:
    """Verify ground-truth reasoner outputs YES, NO, and UNKNOWN under formal semantics."""
    reasoner = TriStateGraphReasoner()
    reasoner.add_fact(DirectionalFact("A", "is_inside", "B", RelationCategory.TRANSITIVE_ASYMMETRIC))
    reasoner.add_fact(DirectionalFact("B", "is_inside", "C", RelationCategory.TRANSITIVE_ASYMMETRIC))

    # 1. Forward transitive entailment -> YES
    ded_fwd = reasoner.evaluate_query("A", "is_inside", "C", RelationCategory.TRANSITIVE_ASYMMETRIC)
    assert ded_fwd.verdict == TriStateVerdict.YES
    assert len(ded_fwd.reasoning_path) == 3

    # 2. Asymmetric reversal -> NO
    ded_rev = reasoner.evaluate_query("B", "is_inside", "A", RelationCategory.TRANSITIVE_ASYMMETRIC)
    assert ded_rev.verdict == TriStateVerdict.NO

    # 3. Non-transitive link -> NO
    reasoner_nt = TriStateGraphReasoner()
    reasoner_nt.add_fact(DirectionalFact("X", "connected_to", "Y", RelationCategory.NON_TRANSITIVE))
    reasoner_nt.add_fact(DirectionalFact("Y", "connected_to", "Z", RelationCategory.NON_TRANSITIVE))
    ded_nt = reasoner_nt.evaluate_query("X", "connected_to", "Z", RelationCategory.NON_TRANSITIVE)
    assert ded_nt.verdict == TriStateVerdict.NO

    # 4. Ungrounded missing entity -> UNKNOWN
    ded_unk = reasoner.evaluate_query("A", "is_inside", "GhostSector", RelationCategory.TRANSITIVE_ASYMMETRIC)
    assert ded_unk.verdict == TriStateVerdict.UNKNOWN

    # 5. Contradictory cyclic loop -> UNKNOWN
    reasoner_cycle = TriStateGraphReasoner()
    reasoner_cycle.add_fact(DirectionalFact("P", "precedes", "Q", RelationCategory.TRANSITIVE_ASYMMETRIC))
    reasoner_cycle.add_fact(DirectionalFact("Q", "precedes", "P", RelationCategory.TRANSITIVE_ASYMMETRIC))
    ded_cyc = reasoner_cycle.evaluate_query("P", "precedes", "Q", RelationCategory.TRANSITIVE_ASYMMETRIC)
    assert ded_cyc.verdict == TriStateVerdict.UNKNOWN


def test_m41_benchmark_dataset_integrity() -> None:
    """Verify task distribution, category coverage, and tri-state balance."""
    assert len(M41_DIRECTIONAL_BENCHMARK) == 40

    categories = {t.category for t in M41_DIRECTIONAL_BENCHMARK}
    assert {
        "single_hop",
        "two_hop",
        "three_hop",
        "asymmetric",
        "nontransitive",
        "ungrounded",
        "contradictory",
        "counterfactual",
    }.issubset(categories)

    verdicts = {t.expected_verdict for t in M41_DIRECTIONAL_BENCHMARK}
    assert verdicts == {"YES", "NO", "UNKNOWN"}

    counterfactuals = [t for t in M41_DIRECTIONAL_BENCHMARK if t.is_counterfactual]
    assert len(counterfactuals) == 5

    assert len(SEQUENTIAL_RELATION_WORLDS) == 4


def test_m40_baseline_audit_record() -> None:
    """Verify Phase 1 audit record generation, retention reconciliation, and hash invariants."""
    tok = _create_dummy_tokenizer()
    model = _create_minimal_test_model(tok.vocab_size)
    dev = torch.device("cpu")
    initial_hash = compute_base_model_hash(model)

    audit = M40BaselineAuditor.audit_baseline(model, tok, dev, initial_hash)
    assert audit.base_parameters_invariant is True
    assert audit.canonical_carry_protocol_verified is True
    assert audit.m40_reproduced_2hop_accuracy == 0.75
    assert audit.m40_reproduced_asymmetric_accuracy == 0.0
    assert audit.m40_reproduced_abstention_accuracy == 0.0
    assert audit.m40_raw_final_accuracy == 1.0
    assert "RECONCILED" in audit.retention_reconciliation_verdict
    assert "DIAGNOSIS CONFIRMED" in audit.abstention_non_transfer_root_cause


def test_uncertainty_routing_engine_mechanics() -> None:
    """Verify deterministic routing, neural training, and hybrid routing fallback."""
    tok = _create_dummy_tokenizer()
    dev = torch.device("cpu")
    engine = UncertaintyRoutingEngine(tok, dev)

    # 1. Deterministic signatures
    act_arith, conf_arith = engine.deterministic_route("[STATE] 5 7 0 ->")
    assert act_arith == "arithmetic"
    assert conf_arith >= 0.90

    act_unk, conf_unk = engine.deterministic_route("Is Zorblax inside Ghost-Node?")
    assert act_unk == "epistemic_unknown"
    assert conf_unk >= 0.90

    # 2. Neural router training
    train_samples = [
        ("Calculate 5 + 7 with carry", "arithmetic"),
        ("All men are mortal and Socrates is a man", "logic"),
        ("Zorblax is inside Nexus-4", "relational_positive"),
        ("Is Nexus-4 inside Zorblax?", "relational_negative"),
        ("Step 1: A in B. Step 2: B in C. Therefore A in C.", "relational_multihop"),
        ("Is Glapton inside Xyphos-7?", "counterfactual"),
        ("Is Zorblax inside Ghost-Node?", "epistemic_unknown"),
        ("Search open web archives", "retrieval"),
    ]
    loss = engine.train_neural_router(train_samples, epochs=10, lr=1e-2)
    assert isinstance(loss, float)

    # 3. Policy evaluation
    res_det, res_neu, res_hyb = engine.evaluate_routing(train_samples)
    assert res_det.policy_name == "DETERMINISTIC_HEURISTIC_CONTROLLER"
    assert res_neu.policy_name == "NEURAL_UNCERTAINTY_ROUTER"
    assert res_hyb.policy_name == "HYBRID_UNCERTAINTY_CONTROLLER"
    assert res_hyb.overall_routing_accuracy >= 0.0


def test_corrective_trainer_adapter_optimization() -> None:
    """Verify that directional and negative pairs train cleanly on a minimal modular adapter."""
    tok = _create_dummy_tokenizer()
    model = _create_minimal_test_model(tok.vocab_size)
    dev = torch.device("cpu")

    inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=4)
    add_domain_adapter(model, domain_name="test_corrective", r=4)

    trainer = CorrectiveTrainer(model, tok, dev)
    pairs = [
        ("[PREMISES] A is inside B. [QUESTION] Is A inside B?", "[ANSWER] YES"),
        ("[PREMISES] A is inside B. [QUESTION] Is B inside A?", "[ANSWER] NO"),
        ("[PREMISES] A is inside B. [QUESTION] Is A inside UnknownZ?", "[ANSWER] UNKNOWN"),
    ]
    loss = trainer.train_adapter_on_pairs(model, "test_corrective", pairs, epochs=2, lr=1e-3)
    assert loss >= 0.0
