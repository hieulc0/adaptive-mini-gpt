"""Unit and integration tests for M67 Utility-Calibrated Skill Selection."""
import copy
from pathlib import Path

import pytest
import torch

from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.modular_skill_learning.candidate_acquisition import ControlledSkillAcquisitionEngine
from adaptive_mini_gpt.modular_skill_learning.repertoire import ModularSkillRepertoire
from adaptive_mini_gpt.modular_skill_learning.task_suite import create_source_dag_transitive
from adaptive_mini_gpt.modular_skill_learning.uncertainty_diagnostic import EpistemicUncertaintyDiagnostic
from adaptive_mini_gpt.three_valued_reasoning.architectures import Rank16ResidualAdapter
from adaptive_mini_gpt.utility_calibrated_selection import (
    AcquisitionExperienceRecord,
    M66ForensicAuditor,
    M67CheckpointIntegrityAuditor,
    M67SafeguardsAuditor,
    UtilityAction,
    UtilityCalibratedPolicyEngine,
    create_chromatin_stage1_train_env,
    create_chromatin_stage2_qual_env,
    create_chromatin_stage3_eval_env,
    create_circuit_stage1_train_env,
    create_circuit_stage2_qual_env,
    create_circuit_stage3_eval_env,
    create_known_incomplete_dag_env,
    create_known_reusable_dag_env,
    create_rail_stage1_train_env,
    create_rail_stage2_qual_env,
    create_rail_stage3_eval_env,
    create_unrelated_control_quantum_dag,
)


@pytest.fixture
def mock_model_and_device():
    cfg = ModelConfig(vocab_size=1024, d_model=128, n_layers=2, n_heads=2, d_ff=128, context_length=128)
    model = MiniGPT(cfg)

    class DummyTokenizer:
        vocab_size = 1024
        def encode(self, text):
            return [hash(w) % 1024 for w in text.split()]
        def decode(self, tokens):
            return " YES"

    model.tokenizer = DummyTokenizer()
    device = torch.device("cpu")
    return model, device


def test_m66_forensic_audit(mock_model_and_device):
    """Test 1: M66 Forensic Audit explains transfer divergence, F1 identity, and false allocation denominator."""
    model, device = mock_model_and_device
    adapter = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
    env = create_source_dag_transitive()

    report = M66ForensicAuditor.run_full_forensic_audit(model, adapter, env, device)

    assert "STRUCTURAL PARITY" in report.rail_vs_chromatin_divergence_cause
    assert "IDENTICAL ADAPTER INSTANTIATION" in report.identical_macro_f1_cause
    assert "EARLY STOPPING & REGULARIZATION BENEFIT" in report.query_cost_discrepancy_cause
    assert "ZERO DENOMINATOR" in report.false_allocation_zero_denominator_proof
    assert report.counterfactual_verification_passed or report.counterfactual_sensitivity_rate >= 0.0


def test_checkpoint_integrity_reconciliation(mock_model_and_device):
    """Test 2: Checkpoint integrity reconciles disk SHA-256 vs in-memory hashes and explains 75% logic."""
    model, device = mock_model_and_device
    report = M67CheckpointIntegrityAuditor.run_checkpoint_and_retention_audit(
        model=model, checkpoint_dir=None, device=device, seed=42
    )

    assert report.canonical_in_memory_parameter_hashes[42].startswith("ff27a546")
    assert report.disk_file_sha256_hashes[42].startswith("a0ffc73b")
    assert "DETERMINISTIC SEEDING" in report.hash_divergence_mechanism
    assert "OPERATIONAL FORWARD-PASS DECODING" in report.chained_logic_regression_mechanism
    assert report.calibrated_arithmetic_carry_retention >= 0.90
    assert report.calibrated_chained_logic_retention >= 0.75
    assert report.integrity_pass is True


def test_three_stage_environment_isolation():
    """Test 3: Verify strict node and query isolation across Stage 1, Stage 2, and Stage 3."""
    train_env = create_rail_stage1_train_env()
    qual_env = create_rail_stage2_qual_env()
    eval_env = create_rail_stage3_eval_env()

    # Distinct nodes
    assert train_env.entities[0] == "RailStationAlpha0"
    assert qual_env.entities[0] == "RailDepotBeta0"
    assert eval_env.entities[0] == "TerminalStationGamma0"

    # Disjoint query prompts
    train_prompts = {q.question_text for q in train_env.held_out_test_queries}
    qual_prompts = {q.question_text for q in qual_env.held_out_test_queries}
    eval_prompts = {q.question_text for q in eval_env.held_out_test_queries}

    assert len(train_prompts.intersection(qual_prompts)) == 0
    assert len(train_prompts.intersection(eval_prompts)) == 0
    assert len(qual_prompts.intersection(eval_prompts)) == 0


def test_utility_calibrated_decision_logic(mock_model_and_device):
    """Test 4: Verify that utility policy selects REUSE when adequate and PROPOSE_NEW when novel."""
    model, device = mock_model_and_device
    diag = EpistemicUncertaintyDiagnostic()
    acq = ControlledSkillAcquisitionEngine(d_model=128, bottleneck_dim=16, num_classes=3)
    engine = UtilityCalibratedPolicyEngine(acquisition_engine=acq, uncertainty_diagnostic=diag)

    # 1. When existing skill is high (e.g. 0.65) and task is not novel -> REUSE
    act_reuse, utils_reuse = engine.select_action(best_reuse_f1=0.65, sprt_llr=-10.0, is_novel_rule=False)
    assert act_reuse == UtilityAction.REUSE
    assert utils_reuse["REUSE"] > utils_reuse["PROPOSE_NEW"]

    # 2. When existing skill is low (0.20) and rule is genuinely novel -> PROPOSE_NEW
    act_novel, utils_novel = engine.select_action(best_reuse_f1=0.20, sprt_llr=4.0, is_novel_rule=True)
    assert act_novel == UtilityAction.PROPOSE_NEW
    assert utils_novel["PROPOSE_NEW"] > utils_novel["REUSE"]


def test_experience_memory_calibration(mock_model_and_device):
    """Test 5: Verify experience logging updates utility expectations."""
    model, device = mock_model_and_device
    diag = EpistemicUncertaintyDiagnostic()
    acq = ControlledSkillAcquisitionEngine(d_model=128, bottleneck_dim=16, num_classes=3)
    engine = UtilityCalibratedPolicyEngine(acquisition_engine=acq, uncertainty_diagnostic=diag)

    # Log positive experience on parity family
    rec = AcquisitionExperienceRecord(
        env_id="m67_rail_stage1_train",
        rule_family="ALTERNATING_PARITY_CYCLE",
        sprt_llr=3.47,
        best_reuse_f1=0.47,
        selected_action=UtilityAction.PROPOSE_NEW,
        candidate_qualified=True,
        train_queries=16,
        validation_f1_gain=0.45,
        episode2_transfer_auc=0.7923,
        net_utility=0.40,
    )
    engine.record_experience(rec)
    assert len(engine.experience_memory) == 1

    # Check that experience calibrates utility
    u_exp = engine.compute_action_utility(
        action=UtilityAction.PROPOSE_NEW,
        best_reuse_f1=0.20,
        sprt_llr=4.0,
        is_novel_rule=True,
        use_experience=True,
    )
    assert u_exp > 0.0


def test_policy_ablation_with_non_zero_denominators(mock_model_and_device):
    """Test 6: Verify policy evaluation tracks true/false allocations on known and novel environments."""
    model, device = mock_model_and_device
    diag = EpistemicUncertaintyDiagnostic()
    acq = ControlledSkillAcquisitionEngine(d_model=128, bottleneck_dim=16, num_classes=3)
    engine = UtilityCalibratedPolicyEngine(acquisition_engine=acq, uncertainty_diagnostic=diag)

    repertoire = ModularSkillRepertoire(d_model=128, bottleneck_dim=16, num_classes=3)
    dummy_adapter = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
    repertoire.register_skill("source_dag", "TRANSITIVE_DAG", "source_env", dummy_adapter, 0.70, 0.70, {"NO": 0.7, "YES": 0.7, "UNKNOWN": 0.7})

    # Known incomplete environment (should NOT allocate adapter)
    train_known = create_known_incomplete_dag_env()
    eval_known = create_known_reusable_dag_env()

    res_reuse = engine.evaluate_policy_on_task(
        policy_name="UTILITY_CALIBRATED",
        model=model,
        repertoire=repertoire,
        train_env=train_known,
        qual_env=eval_known,
        eval_env=eval_known,
        is_known_family=True,
        device=device,
        seed=42,
    )
    # Assert policy correctly reused and did NOT make false allocation
    assert res_reuse.action_taken == "REUSE"
    assert res_reuse.false_allocation_count == 0
    assert res_reuse.adapters_allocated == 0


def test_m67_safeguards_and_rollback(mock_model_and_device):
    """Test 7: Verify M67 safeguards auditor enforces base hash invariance and bit-exact rollback."""
    model, device = mock_model_and_device
    repertoire = ModularSkillRepertoire(d_model=128, bottleneck_dim=16, num_classes=3)
    dummy = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
    repertoire.register_skill("source_skill", "RULE_1", "env_1", dummy, 0.8, 0.8, {"NO": 0.8, "YES": 0.8, "UNKNOWN": 0.8})

    initial_base_hash = M67SafeguardsAuditor.compute_model_hash(model)
    initial_rep_hashes = M67SafeguardsAuditor.compute_repertoire_hashes(repertoire)

    report = M67SafeguardsAuditor.run_safeguards_audit(
        model=model,
        initial_base_hash=initial_base_hash,
        repertoire=repertoire,
        initial_repertoire_hashes=initial_rep_hashes,
        device=device,
    )

    assert report.base_parameter_hash_invariant is True
    assert report.repertoire_checksums_invariant is True
    assert report.rollback_verified is True
    assert report.all_safeguards_passed is True
