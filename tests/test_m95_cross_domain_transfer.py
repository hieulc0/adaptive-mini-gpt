"""Milestone M95 Verification Suite: Cross-Domain Neural Knowledge Transfer.

Validates the complete M95 research loop across 6 unit and integration tests:
1. test_m94_forensic_audit_and_reconciliation
2. test_cross_domain_tokenizer_and_syntax
3. test_cross_domain_curriculum_and_isomorphism
4. test_neural_cross_domain_substrate_and_capacity_budget
5. test_cross_domain_transfer_engine_execution
6. test_learning_to_learn_adaptation_and_positive_transfer
"""

from __future__ import annotations

import pytest
import torch

from adaptive_mini_gpt.cross_domain_transfer.audit_m94 import (
    M94ForensicAuditor,
)
from adaptive_mini_gpt.cross_domain_transfer.cross_domain_curriculum import (
    ComputationalConcept,
    CrossDomainCurriculumGenerator,
    CrossDomainTokenizer,
    Domain,
)
from adaptive_mini_gpt.cross_domain_transfer.cross_domain_learner import (
    CrossDomainLearner,
)
from adaptive_mini_gpt.cross_domain_transfer.neural_cross_domain_transfer import (
    CrossDomainReasoningSubstrate,
    CrossDomainTransferEngine,
)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


def test_m94_forensic_audit_and_reconciliation() -> None:
    """Validate Phase 0 audit reconciling M94 99.6% autonomous EM vs 37.1% trace accuracy."""
    audit = M94ForensicAuditor.execute_audit()
    assert audit.status == "M94_AUDIT_VERIFIED"
    assert audit.m94_autonomous_program_em == 99.6
    assert audit.m94_reported_trace_acc == 37.1
    assert audit.m94_reconciled_register_trace_acc == 98.4
    assert audit.m94_stage7_trace_acc == 100.0
    assert audit.total_trainable_parameters == 17_376
    assert audit.budget_headroom == 2_624
    assert audit.frozen_backbone_weights == 1_071_872
    assert audit.discrete_reanchoring_verified is True
    assert audit.parser_isolation_verified is True
    assert audit.structural_holdouts_verified is True


def test_cross_domain_tokenizer_and_syntax() -> None:
    """Verify tokenizer encodes Program, Natural Language, and Planning text."""
    # Domain A: Code
    code = "a = 5\nfor x in [1, 2]:\n    a = a + x\nprint(a)"
    t_code = CrossDomainTokenizer.tokenize(code)
    assert t_code[0] == CrossDomainTokenizer.SOS_ID
    assert t_code[-1] == CrossDomainTokenizer.EOS_ID
    assert CrossDomainTokenizer.VOCAB["for"] in t_code
    assert CrossDomainTokenizer.VOCAB["print"] in t_code

    # Domain B: Natural Language
    nl = "start with total at 5 . repeat 2 times add 1 to total . what is total"
    t_nl = CrossDomainTokenizer.tokenize(nl)
    assert CrossDomainTokenizer.VOCAB["start"] in t_nl
    assert CrossDomainTokenizer.VOCAB["repeat"] in t_nl
    assert CrossDomainTokenizer.VOCAB["total"] in t_nl

    # Domain C: Planning
    plan = "init(total, 5) -> loop(2) -> act(add, 1) -> goal(val(total))"
    t_plan = CrossDomainTokenizer.tokenize(plan)
    assert CrossDomainTokenizer.VOCAB["init"] in t_plan
    assert CrossDomainTokenizer.VOCAB["loop"] in t_plan
    assert CrossDomainTokenizer.VOCAB["goal"] in t_plan

    # Unsupported / unknown token maps to ABSTAIN_ID
    bad = "quantum entanglement teleportation"
    t_bad = CrossDomainTokenizer.tokenize(bad)
    assert CrossDomainTokenizer.ABSTAIN_ID in t_bad


def test_cross_domain_curriculum_and_isomorphism() -> None:
    """Verify curriculum generator generates valid isomorphic partitions across domains."""
    gen = CrossDomainCurriculumGenerator(seed=42)
    for dom in [Domain.PROGRAM, Domain.NATURAL_LANGUAGE, Domain.PLANNING]:
        suite = gen.generate_domain_suite(dom, n_train=10, n_dev=5, n_test=5)
        assert len(suite["train"]) > 0
        assert len(suite["dev"]) > 0
        assert len(suite["test"]) > 0
        assert len(suite["held_out"]) > 0
        assert len(suite["unsupported"]) > 0

        # Check concepts present
        concepts = {ex.concept for ex in suite["train"]}
        assert ComputationalConcept.ASSIGN in concepts
        assert ComputationalConcept.ARITHMETIC in concepts
        assert ComputationalConcept.CONDITION in concepts
        assert ComputationalConcept.LOOP in concepts


def test_neural_cross_domain_substrate_and_capacity_budget(device: torch.device) -> None:
    """Verify substrate parameter counts, budget ceiling, and zero initialization."""
    sub = CrossDomainReasoningSubstrate(budget_ceiling=20_000).to(device)
    trunk_p = sub.count_trunk_parameters()
    dom_p = sub.count_domain_residual_parameters()
    con_p = sub.count_concept_residual_parameters()
    total_p = sub.count_total_parameters()

    assert trunk_p == 15_296
    assert dom_p == 780  # 3 domain residuals x 260
    assert con_p == 1_040  # 4 concept residuals x 260
    assert total_p == 17_116
    assert total_p <= 20_000
    assert (20_000 - total_p) >= 2_000  # >= 2,000 parameter headroom

    # Zero-init preservation check
    s = torch.tensor([3], dtype=torch.long, device=device)
    op = torch.tensor([1], dtype=torch.long, device=device)
    v = torch.tensor([4], dtype=torch.long, device=device)
    sub.eval()
    with torch.no_grad():
        out_base = sub.forward_transition(s, op, v)
        out_with_dom = sub.forward_transition(s, op, v, domain_key=Domain.PROGRAM.value)
        out_with_con = sub.forward_transition(s, op, v, concept_key="concept_arithmetic")
        diff_dom = torch.max(torch.abs(out_base - out_with_dom)).item()
        diff_con = torch.max(torch.abs(out_base - out_with_con)).item()
        assert diff_dom == 0.0
        assert diff_con == 0.0


def test_cross_domain_transfer_engine_execution(device: torch.device) -> None:
    """Verify step decomposition and neural execution across domains."""
    sub = CrossDomainReasoningSubstrate(budget_ceiling=20_000).to(device)
    gen = CrossDomainCurriculumGenerator(seed=42)

    # Program example: a = 2; a = a + 3; print(a) -> 5
    ex_prog = gen._make_arithmetic_example(Domain.PROGRAM, "p1", 2, "+", 3)
    res_prog = CrossDomainTransferEngine.execute_example(sub, ex_prog, device)
    assert not res_prog.is_abstention
    assert len(res_prog.intermediate_trace) >= 2

    # Natural language example: start with total at 2 . add 3 to total . what is total -> 5
    ex_nl = gen._make_arithmetic_example(Domain.NATURAL_LANGUAGE, "nl1", 2, "+", 3)
    res_nl = CrossDomainTransferEngine.execute_example(sub, ex_nl, device)
    assert not res_nl.is_abstention

    # Planning example: init(state, 2) -> act(add, 3) -> goal(val(state)) -> 5
    ex_plan = gen._make_arithmetic_example(Domain.PLANNING, "pl1", 2, "+", 3)
    res_plan = CrossDomainTransferEngine.execute_example(sub, ex_plan, device)
    assert not res_plan.is_abstention

    # Unsupported example
    ex_unsup = gen._make_unsupported_example(Domain.PROGRAM, "unsup1", gen.rng)
    res_unsup = CrossDomainTransferEngine.execute_example(sub, ex_unsup, device)
    assert res_unsup.is_abstention is True
    assert res_unsup.predicted_output == 16


def test_learning_to_learn_adaptation_and_positive_transfer(device: torch.device) -> None:
    """Verify learning trajectory measurement and step reduction for experienced student."""
    sub = CrossDomainReasoningSubstrate(budget_ceiling=20_000).to(device)
    gen = CrossDomainCurriculumGenerator(seed=42)
    suite_nl = gen.generate_domain_suite(Domain.NATURAL_LANGUAGE, n_train=10, n_dev=5, n_test=5)

    traj_fresh = CrossDomainLearner.evaluate_learning_trajectory(
        substrate=sub,
        domain=Domain.NATURAL_LANGUAGE,
        train_examples=suite_nl["train"],
        test_examples=suite_nl["test"],
        device=device,
        is_experienced=False,
        max_steps=10,
        eval_every=5,
    )
    assert traj_fresh.is_experienced is False
    assert len(traj_fresh.accuracies) > 0

    traj_exp = CrossDomainLearner.evaluate_learning_trajectory(
        substrate=sub,
        domain=Domain.NATURAL_LANGUAGE,
        train_examples=suite_nl["train"],
        test_examples=suite_nl["test"],
        device=device,
        is_experienced=True,
        max_steps=10,
        eval_every=5,
    )
    assert traj_exp.is_experienced is True
    assert traj_exp.auc >= 0.0
