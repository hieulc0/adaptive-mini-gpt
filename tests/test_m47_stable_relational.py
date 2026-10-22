from __future__ import annotations
from typing import Any
"""Unit tests for M47: Stable Relational Representations and Transferable Learning."""

import torch

from adaptive_mini_gpt.data.bpe_scaling import train_scaled_bpe_tokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.stable_relational.audit import M46ForensicAuditor
from adaptive_mini_gpt.stable_relational.continual_relational import ContinualRelationalEvaluator
from adaptive_mini_gpt.stable_relational.hard_negatives import (
    HardNegativeGenerator,
    StructuralContrastiveLoss,
)
from adaptive_mini_gpt.stable_relational.learning_to_learn import RelationalMetaLearningRunner
from adaptive_mini_gpt.stable_relational.relational_arms import M47RelationalArmsRunner
from adaptive_mini_gpt.stable_relational.representation_probes import IndependentRepresentationProber
from adaptive_mini_gpt.stable_relational.stratified_replay import StratifiedRelationalReplayBuffer
from adaptive_mini_gpt.stable_relational.structural_benchmark import (
    EnhancedRelationalDataset,
    get_m47_benchmark_suite,
)


def _make_dummy_model_and_tok() -> tuple[MiniGPT, Any]:
    text = "The court binds the magistrate. Glucose catalyzes pyruvate. Domain subsumes kingdom. Confederacy defends province. ApexConstitutionalCourt AppellateCircuitA DistrictTribunalOne AdministrativeBoardA MunicipalMagistrate."
    tok = train_scaled_bpe_tokenizer(text, target_vocab_size=512)
    cfg = ModelConfig(
        vocab_size=512,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        context_length=128,
    )
    model = MiniGPT(cfg)
    return model, tok


def test_m46_audit_and_seed_42_diagnosis() -> None:
    auditor = M46ForensicAuditor()
    report = auditor.generate_audit_report()
    assert report.m46_reported_accuracy == 0.4722
    assert report.m46_reported_macro_f1 == 0.4373
    assert "FORENSIC SEED 42 DIAGNOSIS" in report.seed_42_diagnosis
    assert len(report.stratification_verifications) == 6
    for v in report.stratification_verifications:
        assert v.is_balanced is True


def test_enhanced_structural_benchmark_topologies() -> None:
    suite = get_m47_benchmark_suite()
    assert len(suite) == 4
    assert set(suite.keys()) == {
        "biological_taxonomy",
        "jurisprudential_precedence",
        "metabolic_pathways",
        "geopolitical_alliances",
    }
    topos = {w.topology for w in suite.values()}
    assert topos == {"CHAIN", "TREE", "DIAMOND", "FORK"}

    for name, w in suite.items():
        assert len(w.held_out_test_pool) == 24
        # Exactly 8 YES, 8 NO, 8 UNK
        yes_c = sum(1 for t in w.held_out_test_pool if t.expected_verdict == "YES")
        no_c = sum(1 for t in w.held_out_test_pool if t.expected_verdict == "NO")
        unk_c = sum(1 for t in w.held_out_test_pool if t.expected_verdict == "UNKNOWN")
        assert yes_c == 8 and no_c == 8 and unk_c == 8


def test_stratified_replay_buffer_balance() -> None:
    suite = get_m47_benchmark_suite()
    w = suite["biological_taxonomy"]
    buf = StratifiedRelationalReplayBuffer(capacity_per_class=50, seed=42)
    buf.add_batch(w.train_practice_pool)

    # Sample batch of 6 -> must have exactly 2 YES, 2 NO, 2 UNK
    batch = buf.sample_stratified_batch(batch_size=6)
    assert len(batch) == 6
    counts = {c: sum(1 for t in batch if t.expected_verdict == c) for c in ["YES", "NO", "UNKNOWN"]}
    assert counts["YES"] == 2 and counts["NO"] == 2 and counts["UNKNOWN"] == 2


def test_hard_negatives_and_structural_contrastive_loss() -> None:
    suite = get_m47_benchmark_suite()
    w = suite["jurisprudential_precedence"]
    gen = HardNegativeGenerator(w, seed=42)
    revs = gen.generate_directional_reversals()
    assert len(revs) > 0
    for r in revs:
        assert r.expected_verdict == "NO"

    loss_fn = StructuralContrastiveLoss(temperature=0.15)
    feats = torch.randn(6, 32)
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    hard_negs = torch.randn(2, 32)
    loss = loss_fn(feats, labels, hard_neg_features=hard_negs)
    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_relational_arms_runner_evaluation() -> None:
    model, tok = _make_dummy_model_and_tok()
    suite = get_m47_benchmark_suite()
    w = suite["biological_taxonomy"]

    runner = M47RelationalArmsRunner(model, tok, torch.device("cpu"), seed=42)
    # Fast evaluation with 2 epochs
    arm_b = runner.evaluate_arm_b_balanced_replay(w, practice_budget=6, epochs=2)
    arm_c = runner.evaluate_arm_c_hard_negative_contrastive(w, practice_budget=6, epochs=2)
    arm_d = runner.evaluate_arm_d_external_reasoner(w)

    assert arm_b.overall_accuracy >= 0.0
    assert arm_c.overall_accuracy >= 0.0
    assert arm_d.overall_accuracy == 1.0  # External verifier is 100%


def test_representation_probes_stratified_split() -> None:
    model, tok = _make_dummy_model_and_tok()
    suite = get_m47_benchmark_suite()

    prober = IndependentRepresentationProber(model, tok, torch.device("cpu"))
    metrics = prober.evaluate_representation_quality(list(suite.values()))
    assert 0.0 <= metrics.topology_classification_accuracy <= 1.0
    assert 0.0 <= metrics.directed_edge_accuracy <= 1.0
    assert metrics.non_neural_heuristic_accuracy > 0.0


def test_continual_relational_evaluation_and_retention() -> None:
    model, tok = _make_dummy_model_and_tok()
    suite = get_m47_benchmark_suite()
    worlds = [suite["biological_taxonomy"], suite["jurisprudential_precedence"]]

    evaluator = ContinualRelationalEvaluator(model, tok, torch.device("cpu"), seed=42)
    res = evaluator.run_continual_experiment(worlds, practice_budget_per_world=6, epochs_per_world=2)
    assert len(res) == 4
    for strat, ep_results in res.items():
        assert len(ep_results) == 2
        for ep in ep_results:
            assert ep.arithmetic_carry_retention == 1.00
            assert ep.chained_logic_retention == 1.00


def test_learning_to_learn_curve_auc() -> None:
    model, tok = _make_dummy_model_and_tok()
    suite = get_m47_benchmark_suite()
    meta_w = suite["geopolitical_alliances"]

    runner = RelationalMetaLearningRunner(model, tok, torch.device("cpu"), seed=42)
    res = runner.evaluate_strategy("M47_META_LEARNER", meta_w, epochs_per_budget=2)
    assert len(res.curve) == 6
    assert 0.0 <= res.area_under_curve <= 1.0
    assert res.arithmetic_carry_retention == 1.00
