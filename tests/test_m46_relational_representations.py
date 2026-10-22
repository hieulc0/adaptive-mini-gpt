"""Unit Test Suite for Milestone 46: Relational Representation Learning and Epistemic Generalization."""
from __future__ import annotations

import pytest
import torch

from adaptive_mini_gpt.data.bpe_scaling import train_scaled_bpe_tokenizer
from adaptive_mini_gpt.data.language_corpus import PRETRAIN_PROSE_CORPUS
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.representation.audit import M45ScientificAuditor
from adaptive_mini_gpt.representation.corrective_efficiency import CorrectiveEfficiencyRunner
from adaptive_mini_gpt.representation.epistemic_evaluator import EpistemicGeneralizationEvaluator
from adaptive_mini_gpt.representation.positive_control import (
    RepresentationPositiveControlRunner,
    TriStateClassificationHead,
)
from adaptive_mini_gpt.representation.relational_arms import (
    GraphContrastiveLoss,
    RelationalArmsBenchmark,
)
from adaptive_mini_gpt.representation.representation_evaluator import (
    IndependentRepresentationEvaluator,
)
from adaptive_mini_gpt.representation.structural_benchmark import (
    build_stratified_relational_world,
    get_m46_benchmark_worlds,
)


@pytest.fixture(scope="module")
def tokenizer():
    corpus = " ".join(PRETRAIN_PROSE_CORPUS)
    return train_scaled_bpe_tokenizer(corpus, target_vocab_size=1024)


@pytest.fixture(scope="module")
def mini_model(tokenizer):
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=4,
        context_length=128,
    )
    return MiniGPT(config)


def test_m45_scientific_audit():
    auditor = M45ScientificAuditor()
    report = auditor.generate_audit_report()

    assert "M45 AUDIT VERDICT" in report["phase1_audit_verdict"]
    assert report["metrics_reproduced"]["reproduction_verified"] is True
    assert report["metrics_reproduced"]["mean_accuracy"] == 0.3333
    assert report["metrics_reproduced"]["mean_unknown_recall"] == 0.0

    # Verify discovery of sampling bug
    flaw = report["sampling_flaw_analysis"]
    assert flaw["flaw_name"] == "SEQUENTIAL_CLASS_SLICING_BIAS"
    assert flaw["counts_at_budget_16"]["UNKNOWN"] == 0
    assert flaw["counts_at_budget_16"]["YES"] == 11

    # Verify M44 confounds documentation
    confounds = report["m44_confounds"]
    assert "oracle_schema_confound" in confounds
    assert "bfs_fallback_confound" in confounds

    # Verify open-world semantics invariants
    sem = report["open_world_semantics"]
    assert sem["neural_isolation_verified"] is True


def test_representation_positive_control(mini_model, tokenizer):
    device = torch.device("cpu")
    world = build_stratified_relational_world(
        world_name="test_world",
        split="train",
        topology="CHAIN",
        rel="subsumes",
        inv_rel="subsumed_by",
        entities=["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"],
    )

    runner = RepresentationPositiveControlRunner(mini_model, tokenizer, device)
    res = runner.run_control(world, use_adapter=True, epochs=50, lr=2e-3)

    assert res.configuration == "ADAPTER_PLUS_HEAD"
    assert len(res.loss_curve) == 50
    assert res.final_loss < res.loss_curve[0]
    assert res.gradient_norm_initial > 0.0
    assert res.max_adapter_weight_delta > 0.0
    assert res.train_accuracy >= 0.66  # Significantly above 0.33 chance
    assert res.held_out_accuracy >= 0.33


def test_stratified_relational_benchmark():
    worlds = get_m46_benchmark_worlds()
    assert len(worlds) == 4
    topologies = {w.topology for w in worlds.values()}
    assert topologies == {"CHAIN", "TREE", "DIAMOND", "FORK"}

    for name, w in worlds.items():
        assert len(w.premises) >= 4
        assert len(w.train_practice_pool) >= 12
        assert len(w.held_out_test_pool) >= 24

        # Verify that practice pool is stratified interleaved: YES, NO, UNK
        first_3 = [t.expected_verdict for t in w.train_practice_pool[:3]]
        assert first_3 == ["YES", "NO", "UNKNOWN"]

        # Verify open-world semantics: NO tasks have explicit reverse reachability
        no_tasks = [t for t in w.held_out_test_pool if t.expected_verdict == "NO"]
        for t in no_tasks:
            assert t.epistemic_category == "REFUTED_ASYMMETRY"


def test_contrastive_loss():
    loss_fn = GraphContrastiveLoss(temperature=0.1)
    feats = torch.randn(6, 128)
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    loss = loss_fn(feats, labels)
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)


def test_relational_arms_benchmark(mini_model, tokenizer):
    device = torch.device("cpu")
    world = build_stratified_relational_world(
        world_name="arm_test_world",
        split="test",
        topology="DIAMOND",
        rel="governs",
        inv_rel="governed_by",
        entities=["node_a", "node_b", "node_c", "node_d", "node_e", "node_f", "node_g"],
    )

    benchmark = RelationalArmsBenchmark(mini_model, tokenizer, device)

    # Arm A: M45 baseline (unstratified slice)
    arm_a = benchmark.evaluate_arm_a_m45_baseline(world, practice_budget=8)
    assert arm_a.total_samples == len(world.held_out_test_pool)

    # Arm B: Balanced supervised training
    arm_b = benchmark.evaluate_arm_b_balanced_supervised(world, practice_budget=9, epochs=10)
    assert arm_b.overall_accuracy >= 0.30

    # Arm C: Contrastive supervised training
    arm_c = benchmark.evaluate_arm_c_contrastive_head(world, practice_budget=9, epochs=10)
    assert arm_c.overall_accuracy >= 0.30

    # Arm D: External verified reasoner
    arm_d = benchmark.evaluate_arm_d_external_reasoner(world)
    assert arm_d.overall_accuracy == 1.0000
    assert arm_d.macro_f1 == 1.0000
    assert arm_d.expected_calibration_error == 0.0


def test_representation_evaluator_and_selective_risk(mini_model, tokenizer):
    device = torch.device("cpu")
    worlds = list(get_m46_benchmark_worlds().values())
    w = worlds[0]

    evaluator = IndependentRepresentationEvaluator(mini_model, tokenizer, device)
    rep_metrics = evaluator.evaluate_representation_quality(worlds[:2])

    assert 0.0 <= rep_metrics.topology_classification_accuracy <= 1.0
    assert 0.0 <= rep_metrics.connectivity_classification_accuracy <= 1.0
    assert -1.0 <= rep_metrics.embedding_separation_margin <= 1.0
    assert 0.0 <= rep_metrics.non_neural_heuristic_accuracy <= 1.0

    tri_head = TriStateClassificationHead(d_model=128, num_classes=3).to(device)
    ep_eval = EpistemicGeneralizationEvaluator(mini_model, tri_head, tokenizer, device)
    thresh = ep_eval.calibrate_threshold(w.validation_pool)
    assert 0.40 <= thresh <= 0.85

    risk = ep_eval.evaluate_selective_risk(w.held_out_test_pool, thresh)
    assert 0.0 <= risk.coverage_100_accuracy <= 1.0
    assert 0.0 <= risk.effective_coverage_at_threshold <= 1.0


def test_corrective_efficiency_runner(mini_model, tokenizer):
    device = torch.device("cpu")
    world = build_stratified_relational_world(
        world_name="eff_world",
        split="test",
        topology="TREE",
        rel="synthesizes",
        inv_rel="synthesized_by",
        entities=["e0", "e1", "e2", "e3", "e4", "e5", "e6"],
    )

    runner = CorrectiveEfficiencyRunner(mini_model, tokenizer, device)
    res = runner.run_strategy_benchmark("FRESH_LEARNER", world)

    assert res.strategy_name == "FRESH_LEARNER"
    assert len(res.curve) == 6  # [0, 2, 4, 8, 16, 32]
    assert res.area_under_curve >= 0.0
    assert res.total_duration_seconds > 0.0
    assert res.arithmetic_carry_retention == 1.0000
    assert res.chained_logic_retention == 1.0000
