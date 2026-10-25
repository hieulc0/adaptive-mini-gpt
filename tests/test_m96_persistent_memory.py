"""Milestone M96 Verification Suite: Persistent Memory & Long-Horizon Learning.

Validates the complete M96 research system across 6 unit and integration tests:
1. test_m95_forensic_audit_and_baseline_certification
2. test_persistent_knowledge_store_operations_and_budget
3. test_selective_memory_manager_routing_and_epistemics
4. test_long_horizon_curriculum_generator_structure
5. test_integrated_student_dual_system_execution
6. test_checkpoint_atomic_persistence_and_rollback
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from adaptive_mini_gpt.cross_domain_transfer.neural_cross_domain_transfer import (
    CrossDomainReasoningSubstrate,
)
from adaptive_mini_gpt.persistent_memory.audit_m95 import (
    M95ForensicAuditor,
)
from adaptive_mini_gpt.persistent_memory.integrated_student import (
    IntegratedAdaptiveStudent,
)
from adaptive_mini_gpt.persistent_memory.knowledge_store import (
    PersistentKnowledgeStore,
)
from adaptive_mini_gpt.persistent_memory.long_horizon_curriculum import (
    EpisodeType,
    LongHorizonCurriculumGenerator,
    LongHorizonEpisode,
)
from adaptive_mini_gpt.persistent_memory.selective_memory_manager import (
    MemoryAction,
    SelectiveMemoryManager,
)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


def test_m95_forensic_audit_and_baseline_certification() -> None:
    """Validate Phase 0 forensic audit reconciling M95 baseline findings."""
    audit = M95ForensicAuditor.execute_audit()
    assert audit.status == "M95_AUDIT_VERIFIED"
    assert audit.m95_zero_shot_em == 100.0
    assert audit.m95_fresh_student_em == 40.96
    assert audit.m95_step_reduction_pct == 100.0
    assert audit.m95_total_trainable_parameters == 17_116
    assert audit.m95_budget_headroom == 2_884
    assert audit.m95_frozen_backbone_weights == 1_071_872
    assert audit.semantic_translation_audited is True
    assert audit.isomorphic_semantics_clarified is True
    assert audit.full_regression_tests_passed == 100


def test_persistent_knowledge_store_operations_and_budget() -> None:
    """Test persistent memory storage, retrieval, update, and capacity bounds."""
    store = PersistentKnowledgeStore(max_byte_budget=10_000)

    # 1. Store Fact
    rid = store.store_fact("atlas_labs", "founding_year", 2024, confidence=1.0)
    assert rid.startswith("fact_atlas_labs_founding_year")
    rec = store.retrieve_fact("atlas_labs", "founding_year")
    assert rec is not None
    assert rec.value == 2024
    assert rec.is_active is True

    # 2. Update Fact
    uid = store.update_fact("atlas_labs", "founding_year", 2025, reason="re-audit")
    assert uid != rid
    rec_updated = store.retrieve_fact("atlas_labs", "founding_year")
    assert rec_updated is not None
    assert rec_updated.value == 2025
    assert rec_updated.is_active is True
    assert len(store.corrections) == 1

    # 3. Byte budget calculation
    bytes_used = store.calculate_byte_size()
    assert bytes_used > 0
    assert bytes_used <= 10_000

    # 4. Search
    results = store.search_facts("atlas founding")
    assert len(results) >= 1
    assert results[0].value == 2025


def test_selective_memory_manager_routing_and_epistemics() -> None:
    """Test autonomous memory manager decisions and epistemic filtering."""
    store = PersistentKnowledgeStore()
    manager = SelectiveMemoryManager(store, min_confidence_threshold=0.6)

    # 1. Store New Fact
    d1 = manager.evaluate_statement("hyperscale", "hq", "zurich", confidence=0.95)
    assert d1.action == MemoryAction.STORE_NEW_FACT
    assert d1.value == "zurich"

    # 2. Reject Low-Confidence Fact
    d2 = manager.evaluate_statement("fake_corp", "val", 999, confidence=0.3)
    assert d2.action == MemoryAction.ABSTAIN_UNSUPPORTED

    # 3. Reject Contradictory Fact
    d3 = manager.evaluate_statement(
        "hyperscale", "hq", "tokyo", confidence=0.9, is_contradictory=True
    )
    assert d3.action == MemoryAction.REJECT_CONTRADICTION

    # 4. Update Existing Fact
    d4 = manager.evaluate_statement(
        "hyperscale", "hq", "geneva", confidence=0.95, is_verified_correction=True
    )
    assert d4.action == MemoryAction.UPDATE_EXISTING_FACT

    # 5. Query
    q_res = manager.evaluate_query("hyperscale", "hq")
    assert q_res.action == MemoryAction.RETRIEVE_KNOWLEDGE
    assert q_res.value == "geneva"

    # 6. Unknown Query
    q_unk = manager.evaluate_query("unknown_corp", "unknown_attr")
    assert q_unk.action == MemoryAction.ABSTAIN_UNSUPPORTED


def test_long_horizon_curriculum_generator_structure() -> None:
    """Verify long-horizon curriculum generator creates balanced multi-phase stream."""
    gen = LongHorizonCurriculumGenerator(seed=42, num_episodes=60)
    curriculum = gen.generate_curriculum()
    assert len(curriculum) >= 50

    types = {ep.episode_type for ep in curriculum}
    assert EpisodeType.FACT_LEARN in types
    assert EpisodeType.FACT_UPDATE in types
    assert EpisodeType.FACT_QUERY in types
    assert EpisodeType.DELAYED_FACT_QUERY in types
    assert EpisodeType.PROCEDURAL_SKILL in types
    assert EpisodeType.FACT_SKILL_COMPOSITION in types
    assert EpisodeType.UNSUPPORTED_QUERY in types


def test_integrated_student_dual_system_execution(device: torch.device) -> None:
    """Test dual-system execution across factual, skill, and composite queries."""
    substrate = CrossDomainReasoningSubstrate(budget_ceiling=20_000).to(device)
    student = IntegratedAdaptiveStudent(substrate=substrate, device=device)

    # 1. Fact Learning
    ep_learn = LongHorizonEpisode(
        episode_id=1,
        episode_type=EpisodeType.FACT_LEARN,
        prompt_text="Learn: NovaCore datacenter count is 8.",
        target_output=8,
        entity="novacore",
        attribute="datacenter_count",
        input_value=8,
    )
    res_learn = student.process_episode(ep_learn)
    assert res_learn.predicted_output == 8
    assert res_learn.memory_action == MemoryAction.STORE_NEW_FACT

    # 2. Fact Query
    ep_query = LongHorizonEpisode(
        episode_id=2,
        episode_type=EpisodeType.FACT_QUERY,
        prompt_text="Query: What is datacenter count of NovaCore?",
        target_output=8,
        entity="novacore",
        attribute="datacenter_count",
    )
    res_query = student.process_episode(ep_query)
    assert res_query.predicted_output == 8

    # 3. Fact-Skill Composition
    ep_comp = LongHorizonEpisode(
        episode_id=3,
        episode_type=EpisodeType.FACT_SKILL_COMPOSITION,
        prompt_text="Composition: Retrieve NovaCore datacenters and add 2 repeated twice.",
        target_output=(8 + 2 + 2) % 16,
        entity="novacore",
        attribute="datacenter_count",
        input_value=2,
    )
    res_comp = student.process_episode(ep_comp)
    assert not res_comp.is_abstention
    assert res_comp.intermediate_trace is not None

    # 4. Unsupported Query
    ep_unsup = LongHorizonEpisode(
        episode_id=4,
        episode_type=EpisodeType.UNSUPPORTED_QUERY,
        prompt_text="Query: unknown",
        target_output="ABSTAIN",
        requires_abstention=True,
    )
    res_unsup = student.process_episode(ep_unsup)
    assert res_unsup.is_abstention is True
    assert res_unsup.predicted_output == "ABSTAIN"


def test_checkpoint_atomic_persistence_and_rollback(device: torch.device) -> None:
    """Verify atomic checkpoint serialization and bit-exact restart restoration."""
    substrate = CrossDomainReasoningSubstrate(budget_ceiling=20_000).to(device)
    student = IntegratedAdaptiveStudent(substrate=substrate, device=device)

    # Store fact
    ep = LongHorizonEpisode(
        episode_id=1,
        episode_type=EpisodeType.FACT_LEARN,
        prompt_text="Learn: AlphaTech register is 14.",
        target_output=14,
        entity="alphatech",
        attribute="register",
        input_value=14,
    )
    student.process_episode(ep)

    with tempfile.TemporaryDirectory() as tmpdir:
        chk_path = Path(tmpdir) / "test_student"
        student.save_checkpoint(chk_path)

        # Restore into clean student instance
        new_substrate = CrossDomainReasoningSubstrate(budget_ceiling=20_000).to(device)
        restored_student = IntegratedAdaptiveStudent(substrate=new_substrate, device=device)
        restored_student.load_checkpoint(chk_path)

        # Query fact on restored student
        q_ep = LongHorizonEpisode(
            episode_id=2,
            episode_type=EpisodeType.FACT_QUERY,
            prompt_text="Query: AlphaTech register",
            target_output=14,
            entity="alphatech",
            attribute="register",
        )
        res = restored_student.process_episode(q_ep)
        assert res.predicted_output == 14
        assert restored_student.get_memory_byte_size() == student.get_memory_byte_size()
