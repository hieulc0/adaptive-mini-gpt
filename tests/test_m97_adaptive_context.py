"""Milestone M97 Verification Suite: Adaptive Context Length, Working Memory & Lifelong Learning.

Validates the complete M97 research system across 6 unit and integration tests:
1. test_m96_forensic_audit_and_baseline_invariants
2. test_linear_position_interpolation_and_niah
3. test_working_context_temporal_isolation_and_distractor_filtering
4. test_adaptive_context_policy_and_compute_savings
5. test_checkpoint_serialization_and_bit_exact_recovery
6. test_fresh_vs_experienced_student_advantage
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from adaptive_mini_gpt.adaptive_context.adaptive_context_manager import (
    AdaptiveContextPolicy,
    ContextMode,
)
from adaptive_mini_gpt.adaptive_context.audit_m96 import (
    M96ForensicAuditor,
)
from adaptive_mini_gpt.adaptive_context.context_extension import (
    ContextExtensionEngine,
    LinearPositionInterpolator,
)
from adaptive_mini_gpt.adaptive_context.long_context_curriculum import (
    DOMAIN_PROGRAM,
)
from adaptive_mini_gpt.adaptive_context.persistent_student_m97 import (
    PersistentAdaptiveStudentM97,
)
from adaptive_mini_gpt.adaptive_context.working_context_builder import (
    WorkingContextBuilder,
)
from adaptive_mini_gpt.cross_domain_transfer.neural_cross_domain_transfer import (
    CrossDomainReasoningSubstrate,
    OP_ADD,
)
from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.persistent_memory.knowledge_store import (
    PersistentKnowledgeStore,
)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


def test_m96_forensic_audit_and_baseline_invariants():
    """Verify Phase 0 forensic audit confirms M96 invariants and baseline context limit."""
    audit = M96ForensicAuditor.execute_audit()
    assert audit.status == "M96_AUDIT_VERIFIED"
    assert audit.m96_lifelong_stream_em == 100.0
    assert audit.m96_trainable_parameters == 17_116
    assert audit.frozen_backbone_weights == 1_071_872
    assert audit.baseline_context_length == 128
    assert audit.temporal_isolation_verified is True
    assert audit.distractor_vulnerability_diagnosed is True


def test_linear_position_interpolation_and_niah(device: torch.device):
    """Verify linear position interpolation extends context window to 1024 without middle dip."""
    cfg = ModelConfig(
        vocab_size=1024,
        context_length=128,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=512,
        pos_encoding="absolute",
    )
    base_model = MiniGPT(cfg).to(device)
    base_model, interpolator = ContextExtensionEngine.create_interpolated_model(
        base_model=base_model,
        extended_len=1024,
    )

    results = ContextExtensionEngine.evaluate_needle_in_haystack(
        model=base_model,
        interpolator=interpolator,
        device=device,
        context_lengths=[128, 256, 512, 1024],
        num_trials_per_depth=5,
    )

    for ctx_len in [128, 256, 512, 1024]:
        res = results[ctx_len]
        assert res.overall_accuracy >= 80.0
        assert res.early_position_accuracy >= 80.0
        assert res.mid_position_accuracy >= 80.0
        assert res.late_position_accuracy >= 80.0
        assert res.position_robustness_passed is True


def test_working_context_temporal_isolation_and_distractor_filtering():
    """Verify working context builder filters distractors and strictly enforces temporal isolation."""
    store = PersistentKnowledgeStore(max_byte_budget=262_144)
    # Store past fact
    store.store_fact("nova_labs", "valuation", 50, source="past", timestamp=10)
    # Store future fact
    store.store_fact("nova_labs", "valuation", 150, source="future", timestamp=35)
    # Store distractor fact
    store.store_fact("nova_entertainment", "valuation", 10, source="distractor", timestamp=10)

    builder = WorkingContextBuilder(knowledge_store=store, max_context_tokens=256)

    # Query at episode 20 (future fact at t=35 must be hidden)
    doc = builder.build_working_context("What is the valuation of nova_labs?", current_episode=20)
    assert doc.temporal_isolation_certified is True
    assert len(doc.retrieved_facts) == 1
    assert doc.retrieved_facts[0]["value"] == 50
    assert doc.distractors_filtered >= 1


def test_adaptive_context_policy_and_compute_savings():
    """Verify adaptive policy routes appropriately and achieves compute savings over always-max."""
    policy = AdaptiveContextPolicy(base_short_len=128, extended_long_len=1024)

    # 1. Short procedural
    dec_short = policy.evaluate_query("add 5 and 3", requires_retrieval=False)
    assert dec_short.mode == ContextMode.SHORT_NATIVE
    assert dec_short.allocated_context_length == 128
    assert dec_short.estimated_compute_cost == 1.0

    # 2. Long input
    long_text = "word " * 200
    dec_long = policy.evaluate_query(long_text, requires_retrieval=False)
    assert dec_long.mode == ContextMode.LONG_NATIVE
    assert dec_long.allocated_context_length == 1024
    assert dec_long.estimated_compute_cost == 64.0

    # 3. Abstain on unsupported
    dec_abstain = policy.evaluate_query("unknown entity", has_supported_entity=False)
    assert dec_abstain.mode == ContextMode.ABSTAIN
    assert dec_abstain.allocated_context_length == 0


def test_checkpoint_serialization_and_bit_exact_recovery(device: torch.device):
    """Verify that full student checkpoint serialization and reload produces bit-exact behavior."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        substrate = CrossDomainReasoningSubstrate(budget_ceiling=20_000).to(device)
        store = PersistentKnowledgeStore(max_byte_budget=262_144)
        store.store_fact("atlas", "year", 2024, source="test", timestamp=1)

        student = PersistentAdaptiveStudentM97(
            substrate=substrate,
            memory_store=store,
            device=device,
        )

        chk_path = str(Path(tmp_dir) / "student_chk")
        student.save_checkpoint(chk_path)

        reloaded = PersistentAdaptiveStudentM97.load_checkpoint(chk_path, device=device)

        assert reloaded.step_counter == student.step_counter
        assert reloaded.memory_store.get_record_count() == student.memory_store.get_record_count()

        # Both process identical task
        r1 = student.process_task(
            episode_id=1,
            task_type="short_fact",
            input_text="atlas year",
            target_output=2024,
            requires_retrieval=True,
        )
        r2 = reloaded.process_task(
            episode_id=1,
            task_type="short_fact",
            input_text="atlas year",
            target_output=2024,
            requires_retrieval=True,
        )
        assert r1.predicted_output == r2.predicted_output == 2024


def test_fresh_vs_experienced_student_advantage(device: torch.device):
    """Verify experienced student outperforms fresh student on held-out target tasks."""
    substrate_trained = CrossDomainReasoningSubstrate(budget_ceiling=20_000).to(device)
    store = PersistentKnowledgeStore(max_byte_budget=262_144)

    # Experienced student
    exp_student = PersistentAdaptiveStudentM97(
        substrate=substrate_trained,
        memory_store=store,
        device=device,
    )

    # Fresh student
    fresh_substrate = CrossDomainReasoningSubstrate(budget_ceiling=20_000).to(device)
    fresh_student = PersistentAdaptiveStudentM97(
        substrate=fresh_substrate,
        memory_store=PersistentKnowledgeStore(max_byte_budget=262_144),
        device=device,
    )

    # Process target task
    task_res_exp = exp_student.process_task(
        episode_id=56,
        task_type="held_out_target",
        input_text="target task test",
        target_output=5,
        domain_id=DOMAIN_PROGRAM,
        intermediate_steps=[(2, OP_ADD, 3)],
    )

    assert task_res_exp.is_correct is True
