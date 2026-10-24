"""Tests for Milestone M68: Autonomous Research Loop and Verified Self-Improvement."""
from __future__ import annotations

import copy
import pytest
import torch

from adaptive_mini_gpt.autonomous_research_loop import (
    AutonomousResearchLoopEngine,
    CausalNeuralAttributionAuditor,
    EpisodeCostLedger,
    ExperimentProposal,
    ExperimentProposalEngine,
    M67QualificationCorrectionAuditor,
    M68SafeguardsAuditor,
    RealLearningCostLedger,
    ResearchExperimentAction,
    VerifiedExperimentHistoryTracker,
)
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.three_valued_reasoning.architectures import Rank16ResidualAdapter
from adaptive_mini_gpt.modular_skill_learning.candidate_acquisition import ControlledSkillAcquisitionEngine
from adaptive_mini_gpt.modular_skill_learning.repertoire import ModularSkillRepertoire
from adaptive_mini_gpt.modular_skill_learning.uncertainty_diagnostic import EpistemicUncertaintyDiagnostic
from adaptive_mini_gpt.utility_calibrated_selection.task_environments import (
    create_known_reusable_dag_env,
    create_rail_stage1_train_env,
    create_rail_stage2_qual_env,
    create_rail_stage3_eval_env,
)


class DummyTokenizer:
    vocab_size = 1024
    def encode(self, text: str) -> list[int]:
        return [abs(hash(w)) % 1024 for w in text.split()]

def _build_test_model() -> MiniGPT:
    torch.manual_seed(42)
    cfg = ModelConfig(vocab_size=1024, d_model=128, n_layers=2, n_heads=2, d_ff=128, context_length=128)
    model = MiniGPT(cfg)
    model.tokenizer = DummyTokenizer()
    return model


def test_m67_qualification_correction_audit() -> None:
    report = M67QualificationCorrectionAuditor.run_full_m67_correction_audit()
    assert report.audit_passed is True
    assert "FAIL" in report.corrected_policy_verdict
    assert "UNAMORTIZED" in report.always_fresh_superiority_cause
    assert "SYMMETRIC INVARIANCE" in report.rail_paired_correctness_cause


def test_real_learning_cost_accounting() -> None:
    ledger = RealLearningCostLedger(
        lambda_query=0.001,
        lambda_step=0.0005,
        lambda_gpu_sec=0.002,
        lambda_storage=0.00002,
        expected_life_cycle_episodes=3,
    )

    # Compute cost for FRESH_ADAPT
    fresh_entry = ledger.compute_episode_utility(
        action_name="FRESH_ADAPT",
        macro_f1=0.80,
        oracle_queries=20,
        optimizer_steps=40,
        gpu_seconds=1.0,
        active_trainable_params=2371,
        stored_adapter_params=0,
        inference_latency_ms=10.0,
    )
    assert fresh_entry.stored_adapter_params == 0
    assert fresh_entry.amortized_life_cycle_utility < 0.80

    # Compute cost for TRAIN_NEW_SKILL
    new_skill_entry = ledger.compute_episode_utility(
        action_name="TRAIN_NEW_SKILL",
        macro_f1=0.80,
        oracle_queries=32,
        optimizer_steps=40,
        gpu_seconds=1.5,
        active_trainable_params=2371,
        stored_adapter_params=2371,
        inference_latency_ms=10.0,
    )
    # Under amortization across K=3 episodes, training cost is divided by 3!
    assert new_skill_entry.stored_adapter_params == 2371
    assert new_skill_entry.amortized_life_cycle_utility > new_skill_entry.single_episode_net_utility


def test_experiment_proposal_engine() -> None:
    engine = ExperimentProposalEngine()

    # Known / high reuse task -> REUSE_EXISTING
    prop_reuse = engine.formulate_proposal(
        target_env_id="env_dag",
        best_reuse_f1=0.60,
        best_skill_id="source_dag",
        sprt_llr=-2.0,
        evidence_gap_ratio=0.0,
        is_known_family=True,
    )
    assert prop_reuse.action == ResearchExperimentAction.REUSE_EXISTING
    assert prop_reuse.max_active_parameters == 0

    # Novel rule with high SPRT LLR -> TRAIN_NEW_SKILL
    prop_novel = engine.formulate_proposal(
        target_env_id="env_novel",
        best_reuse_f1=0.20,
        best_skill_id="source_dag",
        sprt_llr=3.5,
        evidence_gap_ratio=0.0,
        is_known_family=False,
    )
    assert prop_novel.action == ResearchExperimentAction.TRAIN_NEW_SKILL
    assert prop_novel.max_active_parameters == 2371

    # High evidence deficit -> EVIDENCE_REPAIR
    prop_repair = engine.formulate_proposal(
        target_env_id="env_gap",
        best_reuse_f1=0.30,
        best_skill_id="source_dag",
        sprt_llr=0.5,
        evidence_gap_ratio=0.8,
        is_known_family=False,
    )
    assert prop_repair.action == ResearchExperimentAction.EVIDENCE_REPAIR


def test_verified_experiment_history_tracker() -> None:
    tracker = VerifiedExperimentHistoryTracker()
    assert tracker.get_topology_conditioned_prior_bonus("PARITY_CYCLE", mode="NO_HISTORY") == 0.0

    # Record successful rail parity experiment
    tracker.record_completed_experiment(
        env_id="rail_stage1",
        graph_topology="PARITY_CYCLE",
        action_taken="TRAIN_NEW_SKILL",
        validation_f1_gain=0.45,
        candidate_accepted=True,
        life_cycle_utility_gain=0.35,
    )

    # Topology-matched query gets the bonus
    cycle_bonus = tracker.get_topology_conditioned_prior_bonus("PARITY_CYCLE", mode="RELEVANT")
    assert cycle_bonus > 0.0

    # Non-matched topology (e.g. CHROMATIN_LATTICE) gets 0 bonus -> prevents over-spawning!
    lattice_bonus = tracker.get_topology_conditioned_prior_bonus("CHROMATIN_LATTICE", mode="RELEVANT")
    assert lattice_bonus == 0.0


def test_causal_neural_attribution_auditor() -> None:
    device = torch.device("cpu")
    model = _build_test_model()
    env = create_known_reusable_dag_env()

    learned_ad = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
    unrelated_ad = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)

    report = CausalNeuralAttributionAuditor.run_causal_attribution_audit(
        model=model,
        learned_adapter=learned_ad,
        unrelated_adapter=unrelated_ad,
        target_env=env,
        device=device,
        seed=42,
    )
    assert hasattr(report, "causal_neural_gain_over_fresh")
    assert hasattr(report, "causal_neural_gain_over_shuffled")


def test_m68_safeguards_auditor() -> None:
    device = torch.device("cpu")
    model = _build_test_model()
    init_hash = M68SafeguardsAuditor.compute_model_hash(model)

    repertoire = ModularSkillRepertoire(d_model=128)
    repertoire.register_skill(
        skill_id="test_skill",
        rule_family="DAG",
        source_env_id="env1",
        adapter=torch.nn.Linear(128, 3),
        macro_f1=0.8,
        accuracy=0.8,
        classwise_recall={},
    )
    init_rep_hashes = M68SafeguardsAuditor.compute_repertoire_hashes(repertoire)

    report = M68SafeguardsAuditor.run_safeguards_audit(
        model=model,
        initial_base_hash=init_hash,
        repertoire=repertoire,
        initial_repertoire_hashes=init_rep_hashes,
        device=device,
    )
    assert report.base_parameter_hash_invariant is True
    assert report.repertoire_checksums_invariant is True
    assert report.expanded_logic_prompts_tested == 8
    assert report.rollback_verified is True
    assert report.all_safeguards_passed is True
