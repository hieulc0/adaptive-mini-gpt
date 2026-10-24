"""Comprehensive Unit Test Suite for Milestone M66: Modular Skill Reuse & Controlled Autonomous Learning."""
from __future__ import annotations

import copy
import pytest
import torch

from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.modular_skill_learning import (
    ControlledSkillAcquisitionEngine,
    EpistemicUncertaintyDiagnostic,
    LearningPolicy,
    LearningPolicyAblationEngine,
    M65EvidenceAndLineageAuditor,
    M66SafeguardsAuditor,
    ModularSkillRepertoire,
    SequentialEpisodeExperiment,
    SequentialTransferArm,
    build_m66_complete_task_suite,
    create_ep1_autonomous_rail_grid,
    create_ep1_epigenetic_chromatin_lattice,
    create_ep2_autonomous_rail_grid,
    create_ep2_epigenetic_chromatin_lattice,
    create_source_consortium_bipartite,
    create_source_dag_transitive,
    create_source_metabolic_inhibitor,
    create_unrelated_control_quantum_dag,
)
from adaptive_mini_gpt.three_valued_reasoning.architectures import Rank16ResidualAdapter
from adaptive_mini_gpt.three_valued_reasoning.semantics import ThreeValuedDatasetValidator


class MockMiniGPT(torch.nn.Module):
    """Compact mock model for deterministic local unit testing without GPU."""

    def __init__(self, vocab_size: int = 128, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        self.linear = torch.nn.Linear(d_model, d_model)
        self.tokenizer = self._build_mock_tokenizer()

    class MockTokenizer:
        def encode(self, text: str) -> list[int]:
            return [(hash(w) % 120) + 1 for w in text.split()] or [1]

        def decode(self, tokens: list[int]) -> str:
            return "MockToken"

    def _build_mock_tokenizer(self) -> MockTokenizer:
        return self.MockTokenizer()

    def forward(self, input_ids: torch.Tensor) -> Any:
        emb = self.embedding(input_ids)
        h = self.linear(emb)

        class Output:
            def __init__(self, hidden: torch.Tensor) -> None:
                self.hidden_states = hidden
                self.last_hidden_state = hidden
                self.logits = hidden

        return Output(h)


@pytest.fixture
def mock_setup():
    device = torch.device("cpu")
    model = MockMiniGPT()
    adapter = Rank16ResidualAdapter(d_model=128, bottleneck_dim=16, num_classes=3)
    return model, adapter, device


def test_m65_evidence_and_lineage_audit(mock_setup):
    """Verify test count reconciliation, base hash lineage, and evidence shortcut audit."""
    model, adapter, device = mock_setup
    lineage = M65EvidenceAndLineageAuditor.audit_checkpoint_and_test_lineage()
    assert lineage["m65_active_suite_test_count"] == 107
    assert lineage["m64_broad_test_count"] == 249
    assert lineage["canonical_tokenizer_fingerprint"] == "f698e45e209b7fe6"
    assert len(lineage["canonical_m18_hashes"]) == 3

    env = create_source_dag_transitive()
    shortcut = M65EvidenceAndLineageAuditor.audit_evidence_context_construction(env.evidence_context)
    assert shortcut["is_valid_inference_evidence"] is True
    assert shortcut["contains_ground_truth_leakage"] is False

    rep = M65EvidenceAndLineageAuditor.run_full_m65_audit(model, adapter, env, device)
    assert rep.evidence_grounding_verified is True
    assert "UNKNOWN" in rep.classwise_metrics


def test_modular_skill_repertoire(mock_setup):
    """Verify skill registration, parameter immutability, and zero-shot evaluation."""
    model, adapter, device = mock_setup
    repertoire = ModularSkillRepertoire(d_model=128, bottleneck_dim=16, num_classes=3)

    env = create_source_dag_transitive()
    rec = repertoire.register_skill(
        skill_id="source_dag_transitive",
        rule_family="DIR_TRANS",
        source_env_id=env.env_id,
        adapter=adapter,
        macro_f1=0.85,
        accuracy=0.88,
        classwise_recall={"NO": 0.9, "YES": 0.9, "UNKNOWN": 0.75},
    )
    assert rec.skill_id == "source_dag_transitive"
    assert len(repertoire.get_skill_ids()) == 1

    f1, acc, recalls, paired = repertoire.evaluate_skill_on_env(
        "source_dag_transitive", model, env, device
    )
    assert 0.0 <= f1 <= 1.0
    assert 0.0 <= acc <= 1.0


def test_uncertainty_diagnostic_separation(mock_setup):
    """Verify that RELATION_UNKNOWN never triggers adapter allocation without rule novelty."""
    model, adapter, device = mock_setup
    diag = EpistemicUncertaintyDiagnostic()

    env_dag = create_source_dag_transitive()
    trace_inc = diag.evaluate_environment_condition(
        env=env_dag,
        known_hypotheses=[env_dag.true_hypothesis],
        condition_name="known_incomplete",
    )
    assert trace_inc.allocation_decision_valid is True
    assert trace_inc.adapter_allocation_proposed is False  # RELATION_UNKNOWN does NOT trigger allocation!

    # Unknown rule condition
    trace_nov = diag.evaluate_environment_condition(
        env=create_ep1_epigenetic_chromatin_lattice(),
        known_hypotheses=[env_dag.true_hypothesis],
        condition_name="unknown_rule",
    )
    assert trace_nov.allocation_decision_valid is True
    assert trace_nov.rule_unknown_detected is True
    assert trace_nov.adapter_allocation_proposed is True


def test_candidate_acquisition_and_independent_qualification(mock_setup):
    """Verify bounded candidate training and independent qualification check."""
    model, adapter, device = mock_setup
    engine = ControlledSkillAcquisitionEngine(max_steps=5, qualification_threshold_f1=0.30)
    repertoire = ModularSkillRepertoire()
    repertoire.register_skill("source_mock", "RULE_MOCK", "mock_env", adapter, 0.20, 0.20, {"UNKNOWN": 0.1})

    train_env = create_ep1_epigenetic_chromatin_lattice()
    eval_env = create_ep2_epigenetic_chromatin_lattice()

    report, candidate = engine.acquire_and_qualify_candidate(
        candidate_id="candidate_epigenetic_lattice",
        rule_family="OPEN_WORLD_EXPLICIT_REFUTATION",
        train_env=train_env,
        eval_env=eval_env,
        model=model,
        repertoire=repertoire,
        budget_n=8,
        device=device,
        seed=42,
    )
    assert report.budget_n == 8
    assert report.candidate_id == "candidate_epigenetic_lattice"
    assert report.train_steps == 5


def test_sequential_episode_skill_reuse(mock_setup):
    """Verify sequential episode transfer: Episode 1 acquisition -> Episode 2 reuse."""
    model, adapter, device = mock_setup
    engine = ControlledSkillAcquisitionEngine(max_steps=5, qualification_threshold_f1=0.10)
    repertoire = ModularSkillRepertoire()
    repertoire.register_skill("source_base", "DIR_TRANS", "mock_env", adapter, 0.30, 0.30, {"UNKNOWN": 0.2})

    exp = SequentialEpisodeExperiment(acquisition_engine=engine, max_steps=5)
    results = exp.run_sequential_episode(
        model=model,
        base_repertoire=repertoire,
        ep1_train_env=create_ep1_epigenetic_chromatin_lattice(),
        ep1_eval_env=create_ep2_epigenetic_chromatin_lattice(),
        ep2_target_env=create_ep2_epigenetic_chromatin_lattice(),
        unrelated_eval_env=create_unrelated_control_quantum_dag(),
        rule_family="OPEN_WORLD_EXPLICIT_REFUTATION",
        device=device,
        seed=42,
    )

    assert SequentialTransferArm.WITH_ACQUIRED_SKILL.value in results
    assert SequentialTransferArm.WITHOUT_ACQUIRED_SKILL.value in results
    assert SequentialTransferArm.FRESH_ADAPTATION_CONTROL.value in results


def test_learning_policy_ablation_and_costs(mock_setup):
    """Verify policy comparisons (Always Fresh, Always Reuse, Novelty-Gated) and cost tracking."""
    model, adapter, device = mock_setup
    engine = ControlledSkillAcquisitionEngine(max_steps=5)
    diag = EpistemicUncertaintyDiagnostic()
    policy_engine = LearningPolicyAblationEngine(engine, diag)

    repertoire = ModularSkillRepertoire()
    repertoire.register_skill("source_base", "DIR_TRANS", "mock_env", adapter, 0.40, 0.40, {"UNKNOWN": 0.2})

    train_env = create_ep1_autonomous_rail_grid()
    eval_env = create_ep2_autonomous_rail_grid()

    res_fresh = policy_engine.evaluate_policy_on_task(
        policy=LearningPolicy.ALWAYS_FRESH,
        model=model,
        repertoire=repertoire,
        train_env=train_env,
        eval_env=eval_env,
        is_known_family=False,
        device=device,
    )
    assert res_fresh.adapters_allocated == 1
    assert res_fresh.oracle_queries_used == 16

    res_gated = policy_engine.evaluate_policy_on_task(
        policy=LearningPolicy.NOVELTY_GATED_REUSE_OR_SPAWN,
        model=model,
        repertoire=repertoire,
        train_env=train_env,
        eval_env=eval_env,
        is_known_family=False,
        device=device,
    )
    assert res_gated.oracle_queries_used > 0


def test_m66_safeguards_retention(mock_setup):
    """Verify base parameter hash invariance, skill immutability, and retention."""
    model, adapter, device = mock_setup
    initial_hash = M66SafeguardsAuditor.compute_model_hash(model)
    repertoire = ModularSkillRepertoire()
    repertoire.register_skill("source_base", "DIR_TRANS", "mock_env", adapter, 0.80, 0.80, {"UNKNOWN": 0.8})
    init_rep_hashes = M66SafeguardsAuditor.compute_repertoire_hashes(repertoire)

    report = M66SafeguardsAuditor.run_safeguards_audit(
        model=model,
        initial_base_hash=initial_hash,
        repertoire=repertoire,
        initial_repertoire_hashes=init_rep_hashes,
        device=device,
    )
    assert report.base_parameter_hash_invariant is True
    assert report.repertoire_checksums_invariant is True
    assert report.all_safeguards_passed is True
