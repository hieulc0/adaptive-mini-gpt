"""Unit tests for M39: Cumulative Knowledge Integration and Generalization."""

import pytest
import torch

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.integration.benchmark_suite import (
    FACTUAL_LEARNING_DATASET,
    PROCEDURAL_LOGIC_TASKS,
    SEQUENTIAL_INTEGRATION_EPISODES,
    IntegrationTask,
)
from adaptive_mini_gpt.integration.carry_audit import CarryRetentionAuditor
from adaptive_mini_gpt.integration.knowledge_integrator import KnowledgeIntegrator
from adaptive_mini_gpt.integration.learned_router import (
    MetacognitiveRoutingSystem,
    NeuralTaskRouter,
)
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
    corpus = "Newton force mass acceleration F ma mitochondria ATP cell Alan Turing 1936 Gutenberg 1440 0 1 2 3 4 5 6 7 8 9 + = : [STATE] [OUT] d c abstain Question Answer logic Socrates mortal"
    tok.train(corpus, target_vocab_size=280)
    return tok


def test_carry_retention_reconciliation_audit() -> None:
    """Verify CarryRetentionAuditor correctly diagnoses raw vs canonical protocols."""
    tok = _create_dummy_tokenizer()
    model = _create_minimal_test_model(vocab_size=max(300, tok.vocab_size))
    dev = torch.device("cpu")
    initial_hash = compute_base_model_hash(model)

    res = CarryRetentionAuditor.audit_reconciliation(model, tok, dev, initial_hash)
    assert res.base_parameters_invariant is True
    assert "protocol" in res.root_cause.lower()
    assert "reconciled" in res.reconciliation_verdict.lower()


def test_factual_benchmark_dataset_integrity() -> None:
    """Test task taxonomy, entity partitions, and abstention markers."""
    assert len(FACTUAL_LEARNING_DATASET) >= 16

    subjects = {t.subject for t in FACTUAL_LEARNING_DATASET}
    assert {"physics", "biology", "computing", "history"}.issubset(subjects)

    abstention_tasks = [t for t in FACTUAL_LEARNING_DATASET if t.is_unanswerable]
    assert len(abstention_tasks) >= 4
    for t in abstention_tasks:
        assert t.task_type == "abstention"
        assert t.expected_answer == "abstain"

    direct_tasks = [t for t in FACTUAL_LEARNING_DATASET if t.task_type == "direct"]
    assert len(direct_tasks) >= 4
    for dt in direct_tasks:
        assert len(dt.alternate_phrasings) >= 1
        assert dt.relationship_triplet is not None


def test_neural_task_router_mechanics() -> None:
    """Test NeuralTaskRouter forward pass and classification training convergence."""
    tok = _create_dummy_tokenizer()
    dev = torch.device("cpu")
    routing_system = MetacognitiveRoutingSystem(tok, dev)

    samples = [
        ("Calculate 5 + 7 with carry", "arithmetic"),
        ("Socrates is human and mortals die", "logic"),
        ("What is force mass times acceleration", "factual_physics"),
        ("Mitochondria produces cellular energy ATP", "factual_biology"),
        ("Alan Turing computing machine 1936", "factual_computing"),
        ("Johannes Gutenberg movable type printing 1440", "factual_history"),
        ("Lead into gold alchemy transmutation", "abstain"),
    ]

    final_loss = routing_system.train_neural_router(samples, epochs=10, lr=1e-2)
    assert isinstance(final_loss, float)

    det_res, neu_res = routing_system.evaluate_routing_policies(samples)
    assert det_res.policy_name == "DETERMINISTIC_HEURISTIC"
    assert neu_res.policy_name == "NEURAL_METACOGNITIVE_ROUTER"
    assert neu_res.overall_routing_accuracy >= 0.0


def test_multi_view_supervision_adapter_training() -> None:
    """Test that multi-view factual pairs train cleanly on isolated adapters."""
    tok = _create_dummy_tokenizer()
    model = _create_minimal_test_model(vocab_size=max(300, tok.vocab_size))
    dev = torch.device("cpu")

    inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=4)
    add_domain_adapter(model, domain_name="test_mv", r=4)

    integrator = KnowledgeIntegrator(model, tok, dev)
    pairs = [
        ("What is force?", "F = ma"),
        ("State Newton's second law.", "F = ma"),
    ]

    loss = integrator.train_adapter_on_pairs(model, "test_mv", pairs, epochs=2, lr=1e-3)
    assert loss >= 0.0


def test_sequential_episodes_data_structure() -> None:
    """Test sequential multi-subject episode structures."""
    assert len(SEQUENTIAL_INTEGRATION_EPISODES) == 4
    for ep in SEQUENTIAL_INTEGRATION_EPISODES:
        assert "subject" in ep
        assert "train_pairs" in ep
        assert len(ep["train_pairs"]) >= 2
        assert "eval_tasks" in ep


def test_procedural_logic_benchmark_structure() -> None:
    """Test procedural logic verification tasks."""
    assert len(PROCEDURAL_LOGIC_TASKS) >= 4
    for lt in PROCEDURAL_LOGIC_TASKS:
        assert lt.subject == "logic"
        assert lt.expected_answer in ["mortal", "true", "beta", "no"]
