"""Unit tests for M42: Learning-to-Learn and Sample Efficiency."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from adaptive_mini_gpt.adaptation.modular_adapter import (
    add_domain_adapter,
    inject_multi_domain_adapters,
    set_active_domain_adapter,
)
from adaptive_mini_gpt.meta.audit import M41Auditor
from adaptive_mini_gpt.meta.benchmark_suite import (
    MetaGraphOracle,
    MetaTask,
    build_meta_family,
    get_benchmark_families,
)
from adaptive_mini_gpt.meta.learning_policy import SelfDirectedLearningPolicy
from adaptive_mini_gpt.meta.meta_learner import MetaLearner
from adaptive_mini_gpt.meta.tri_state_diagnostic import TriStateDiagnostic
from adaptive_mini_gpt.model.minigpt import MiniGPT, MiniGPTConfig
from adaptive_mini_gpt.training.bpe import BPETokenizer


@pytest.fixture
def dummy_setup() -> tuple[MiniGPT, BPETokenizer, torch.device]:
    device = torch.device("cpu")
    config = MiniGPTConfig(
        vocab_size=128,
        context_length=64,
        d_model=64,
        num_layers=2,
        num_heads=2,
        d_ff=128,
        dropout=0.0,
    )
    model = MiniGPT(config).to(device)

    # Train dummy tokenizer with sufficient text to populate vocab
    tok = BPETokenizer()
    corpus = (
        "YES NO UNKNOWN [PREMISES] [QUESTION] [ANSWER] AlphaPhase BetaCycle GammaEra DeltaEpoch "
        "precedes succeeds contains inside_of subsumes charges discharges catalyzes converts "
        "FluxCondenser PlasmaResistor VoltRelay ArcCapacitor Does Is it verified that GhostEntity"
    )
    tok.train(corpus, vocab_size=128)

    return model, tok, device


def test_m41_forensic_audit() -> None:
    report = M41Auditor.audit_m41()
    assert report.audit_status == "AUDIT_VERIFIED_AND_RECONCILED"
    assert "DIAGNOSIS (Directional Tradeoff)" in report.directional_tradeoff_diagnosis
    assert "DIAGNOSIS (0% UNKNOWN Abstention)" in report.unknown_abstention_root_cause
    assert "RECONCILIATION (Cumulative Retention)" in report.retention_reconciliation_verdict
    assert report.backward_transfer_formula_verified is True
    assert report.readiness_for_m42_learning_to_learn is True

    # Validate confusion matrices
    cms = report.audited_confusion_matrices
    assert "ARM_A_M40_BASELINE" in cms
    assert "ARM_B_DIRECTIONAL" in cms
    assert cms["ARM_B_DIRECTIONAL"]["matrix"]["TRUE_NO"]["PRED_NO"] == 10


def test_benchmark_families_separation_and_oracle() -> None:
    families = get_benchmark_families()
    assert "temporal_precedence" in families
    assert "spatial_containment" in families
    assert "hierarchical_subsumption" in families
    assert "dependency_chains" in families
    assert "directed_energy" in families
    assert "chemical_catalysis" in families

    # Test oracle logic
    premises = ["A precedes B", "B precedes C", "C precedes D"]
    # Transitive YES
    assert MetaGraphOracle.solve(premises, "A", "precedes", "C", is_transitive=True, is_symmetric=False) == "YES"
    # Asymmetric reversal NO
    assert MetaGraphOracle.solve(premises, "C", "precedes", "A", is_transitive=True, is_symmetric=False) == "NO"
    # Ungrounded UNKNOWN
    assert MetaGraphOracle.solve(premises, "A", "precedes", "UnknownEntity", is_transitive=True, is_symmetric=False) == "UNKNOWN"
    # Circular self-loop UNKNOWN
    assert MetaGraphOracle.solve(premises, "A", "precedes", "A", is_transitive=True, is_symmetric=False) == "UNKNOWN"

    # Verify disjointness between meta-train and sealed meta-test families
    train_entities = set(families["temporal_precedence"].held_out_test_pool[0].entities_involved)
    test_entities = set(families["directed_energy"].held_out_test_pool[0].entities_involved)
    assert len(train_entities.intersection(test_entities)) == 0


def test_tri_state_diagnostic_mechanisms(dummy_setup: tuple[MiniGPT, BPETokenizer, torch.device]) -> None:
    model, tok, device = dummy_setup
    diag = TriStateDiagnostic(model, tok, device)

    tasks = [
        MetaTask("t1", "temporal", "test", ["A precedes B"], "Does A precedes B?", "YES", "single_hop", ["A", "B"]),
        MetaTask("t2", "temporal", "test", ["A precedes B"], "Does B precedes A?", "NO", "asymmetric", ["B", "A"]),
        MetaTask("t3", "temporal", "test", ["A precedes B"], "Does A precedes X?", "UNKNOWN", "ungrounded", ["A", "X"]),
    ]

    res_a = diag.evaluate_free_form(model, tasks)
    assert res_a.method_name == "METHOD_A_FREE_FORM_GENERATION"

    res_b = diag.evaluate_constrained_decoding(model, tasks)
    assert res_b.method_name == "METHOD_B_CONSTRAINED_DECODING"

    res_c = diag.evaluate_prediction_head(model, train_tasks=tasks, eval_tasks=tasks, epochs=2)
    assert res_c.method_name == "METHOD_C_PREDICTION_HEAD"
    assert res_c.trainable_parameters == 64 * 3 + 3


def test_meta_learner_strategies_and_efficiency_curves(dummy_setup: tuple[MiniGPT, BPETokenizer, torch.device]) -> None:
    model, tok, device = dummy_setup
    learner = MetaLearner(model, tok, device)

    fam = build_meta_family("dummy_fam", "meta_test", "charges", "discharges", ["P1", "P2", "P3", "P4"])

    # Test adaptation curve with small budgets [0, 2]
    inject_multi_domain_adapters(model, target_modules=["c_attn", "c_proj", "c_fc"], r=4)
    add_domain_adapter(model, domain_name="test_adapter", r=4)

    report = learner.evaluate_learning_curve(
        init_model=model,
        adapter_name="test_adapter",
        practice_pool=fam.train_practice_pool,
        test_pool=fam.held_out_test_pool[:4],
        strategy_name="Fresh Learner",
        target_family_name="dummy_fam",
        budgets=[0, 2],
    )

    assert report.strategy_name == "Fresh Learner"
    assert len(report.curve) == 2
    assert 0.0 <= report.area_under_curve <= 1.0


def test_cumulative_retention_and_bwt(dummy_setup: tuple[MiniGPT, BPETokenizer, torch.device]) -> None:
    model, tok, device = dummy_setup
    learner = MetaLearner(model, tok, device)

    fam1 = build_meta_family("fam1", "meta_train", "precedes", "succeeds", ["A1", "A2", "A3", "A4"])
    fam2 = build_meta_family("fam2", "meta_train", "contains", "inside", ["B1", "B2", "B3", "B4"])

    ret_report = learner.evaluate_cumulative_retention(
        strategy="shared_adapter",
        families=[fam1, fam2],
        practice_budget=2,
    )

    assert ret_report.strategy_name == "shared_adapter"
    assert len(ret_report.retention_matrix) == 2
    assert len(ret_report.raw_final_accuracies) == 2
    assert ret_report.arithmetic_carry_intact is True
    assert ret_report.logic_reasoning_intact is True


def test_self_directed_learning_policy(dummy_setup: tuple[MiniGPT, BPETokenizer, torch.device]) -> None:
    model, tok, device = dummy_setup
    policy = SelfDirectedLearningPolicy(model, tok, device)

    t_ans = MetaTask("t1", "temp", "test", ["A precedes B"], "Does A precedes B?", "YES", "single_hop", ["A", "B"])
    t_abs = MetaTask("t2", "temp", "test", ["A precedes B"], "Does A precedes X?", "UNKNOWN", "ungrounded", ["A", "X"])

    dec_ans = policy.decide_deterministic(t_ans)
    assert dec_ans.recommended_action == "ANSWER"

    dec_abs = policy.decide_deterministic(t_abs)
    assert dec_abs.recommended_action == "ABSTAIN"

    comp_report = policy.evaluate_policy_comparison([t_ans, t_abs])
    assert comp_report.total_evaluated_tasks == 2
    assert comp_report.deterministic_accuracy == 1.0
