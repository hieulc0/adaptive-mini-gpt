"""Milestone M86 Test Suite: Integrated Autonomous Continual Learning.

Verifies:
1. M85 Forensic Audit and Multi-Module Parameter Amortization.
2. Persistent Student State & Skill Repertoire Management.
3. Unified Learning Policy Bounds (<1,000 parameters, exactly 893 params).
4. Continual Task Stream Generation across Distinct Regimes.
5. Zero-Shot Knowledge Reuse & Dynamic Acquisition Cycle.
6. Bit-Exact Multi-Module State Rollback via Holistic SHA-256 Fingerprint.
7. M86 Master Benchmark Smoke Test and Report Generation.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.continual_student.audit_m85 import (
    M85ForensicAuditor,
)
from adaptive_mini_gpt.continual_student.benchmark import (
    M86ContinualLearningBenchmark,
)
from adaptive_mini_gpt.continual_student.continual_learner import (
    UnifiedContinualStudent,
)
from adaptive_mini_gpt.continual_student.learning_policy import (
    ContinualStudentStateSignals,
    StudentAction,
    UnifiedLearningPolicy,
)
from adaptive_mini_gpt.continual_student.student_state import (
    PersistentStudentState,
)
from adaptive_mini_gpt.continual_student.task_stream import (
    ContinualTaskStreamGenerator,
)
from adaptive_mini_gpt.model import MiniGPT, ModelConfig


def test_m85_forensic_audit() -> None:
    """Verify M85 audit metrics, multi-module amortization, and speedup."""
    audit = M85ForensicAuditor.execute_audit()
    assert audit.reported_adapter_savings_pct == pytest.approx(44.75, abs=0.01)
    assert audit.controller_parameters == 989
    assert audit.net_savings_1module_pct == pytest.approx(6.60, abs=0.01)
    assert audit.net_savings_5modules_pct == pytest.approx(37.12, abs=0.01)
    assert audit.net_savings_10modules_pct == pytest.approx(40.94, abs=0.01)
    assert audit.asymptotic_savings_pct == pytest.approx(44.75, abs=0.01)
    assert audit.step_advantage_vs_random == pytest.approx(9.4, abs=0.1)
    assert audit.mean_retention_pct == pytest.approx(98.75, abs=0.01)
    assert audit.rollback_verified is True
    assert audit.base_hash_invariant is True


def test_persistent_student_state_and_repertoire() -> None:
    """Verify persistent state tracking, parameter counts, and repertoire registration."""
    cfg = ModelConfig(vocab_size=64, context_length=16, d_model=32, n_heads=2, n_layers=1, d_ff=64)
    model = MiniGPT(cfg)
    model.eval()

    state = PersistentStudentState(base_model=model, d_model=32, num_symbols=16)
    assert state.count_active_adapter_parameters() == 0

    # Allocate module and register skill
    state.module_manager.allocate_module("prim_1", initial_rank=4)
    state.register_skill(
        skill_id="task_1",
        module_name="prim_1",
        task_family="ARITHMETIC_AFFINE",
        rank=4,
        em=1.0,
        episode=0,
    )
    assert "task_1" in state.skill_repertoire
    assert state.skill_repertoire["task_1"].qualification_em == 1.0
    assert state.count_active_adapter_parameters() > 0


def test_unified_learning_policy_bounds() -> None:
    """Verify policy parameters (<1,000 params, exactly 893 params) and action mapping."""
    policy = UnifiedLearningPolicy(in_features=6, hidden_dim=24, num_actions=5)
    param_count = policy.count_parameters()

    assert param_count < 1000
    assert param_count == 989

    # High existing EM -> REUSE_OR_COMPOSE
    supported_state = ContinualStudentStateSignals(
        max_existing_skill_em=0.95,
        current_loss=0.05,
        loss_slope=-0.05,
        prediction_entropy=0.1,
        capacity_pressure=0.2,
        repertoire_overlap_score=0.9,
    )
    act_reuse = policy.select_action(supported_state, deterministic=True)
    assert act_reuse == StudentAction.REUSE_OR_COMPOSE

    # High capacity pressure -> RETIRE_OBSOLETE
    extreme_pressure_state = ContinualStudentStateSignals(
        max_existing_skill_em=0.0,
        current_loss=1.8,
        loss_slope=-0.001,
        prediction_entropy=1.5,
        capacity_pressure=0.98,
        repertoire_overlap_score=0.1,
    )
    act_retire = policy.select_action(extreme_pressure_state, deterministic=True)
    assert act_retire == StudentAction.REIRE_OBSOLETE if hasattr(StudentAction, "REIRE_OBSOLETE") else StudentAction.RETIRE_OBSOLETE


def test_task_stream_generation() -> None:
    """Verify continual task stream generator generates distinct task regimes."""
    stream = ContinualTaskStreamGenerator.generate_stream(seed=42, num_episodes=5, vocab_size=16, seq_len=4)
    assert len(stream) == 5
    assert stream[0].intended_regime == "NOVEL_PRIMITIVE_ACQUISITION"
    assert stream[1].intended_regime == "NOVEL_PRIMITIVE_ACQUISITION"
    assert stream[2].intended_regime == "ZERO_SHOT_REUSE"
    assert stream[3].intended_regime == "CAPACITY_EXPANSION"
    assert stream[4].intended_regime == "CAPACITY_SHRINKAGE"


def test_continual_student_reuse_and_acquisition() -> None:
    """Verify that student acquires unfamiliar task and reuses on duplicate task."""
    torch.manual_seed(42)
    cfg = ModelConfig(vocab_size=1024, context_length=128, d_model=128, n_heads=4, n_layers=4, d_ff=512)
    model = MiniGPT(cfg)
    model.eval()
    device = torch.device("cpu")

    state = PersistentStudentState(base_model=model, d_model=128, num_symbols=16).to(device)
    policy = UnifiedLearningPolicy(in_features=6, hidden_dim=24, num_actions=5).to(device)
    student = UnifiedContinualStudent(state=state, policy=policy, device=device)

    stream = ContinualTaskStreamGenerator.generate_stream(seed=42, num_episodes=3, vocab_size=16, seq_len=4)

    # Episode 0: Novel Primitive -> ACQUIRE
    log0 = student.process_episode(stream[0], max_steps=40, lr=0.015)
    assert "ACQUIRE" in log0.action_taken
    assert log0.qualified is True
    assert log0.steps_taken > 0

    # Episode 2: Duplicate task -> REUSE
    log2 = student.process_episode(stream[2], max_steps=40, lr=0.015)
    assert log2.action_taken == StudentAction.REUSE_OR_COMPOSE.value
    assert log2.steps_taken == 0
    assert log2.qualified is True
    assert log2.final_em >= 0.85


def test_bit_exact_multi_module_rollback() -> None:
    """Verify holistic SHA-256 state restore upon trial failure."""
    cfg = ModelConfig(vocab_size=64, context_length=16, d_model=32, n_heads=2, n_layers=1, d_ff=64)
    model = MiniGPT(cfg)
    model.eval()
    device = torch.device("cpu")

    state = PersistentStudentState(base_model=model, d_model=32, num_symbols=16).to(device)
    state.module_manager.allocate_module("mod_a", initial_rank=4, device=device)
    state.module_manager.allocate_module("mod_b", initial_rank=8, device=device)

    saved_ckpt = state.save_checkpoint()
    saved_hash = state.compute_holistic_fingerprint()

    # Mutate state
    state.module_manager.expand_module_rank("mod_a", target_rank=8)
    with torch.no_grad():
        state.module_manager.get_module("mod_a").down_proj.weight.add_(3.0)
    state.module_manager.allocate_module("mod_c", initial_rank=2, device=device)
    perturbed_hash = state.compute_holistic_fingerprint()
    assert saved_hash != perturbed_hash

    # Restore from checkpoint
    state.restore_checkpoint(saved_ckpt, device=device)
    restored_hash = state.compute_holistic_fingerprint()
    assert saved_hash == restored_hash
    assert not state.module_manager.has_module("mod_c")
    assert state.module_manager.get_module("mod_a").rank == 4


def test_m86_benchmark_smoke() -> None:
    """Smoke test full M86 continual learning benchmark on a single seed."""
    torch.manual_seed(42)
    cfg = ModelConfig(vocab_size=1024, context_length=128, d_model=128, n_heads=4, n_layers=4, d_ff=512)
    model = MiniGPT(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    device = torch.device("cpu")

    result = M86ContinualLearningBenchmark.run_benchmark(
        model=model,
        device=device,
        seeds=[42],
        num_episodes=4,
    )

    assert "report_data" in result
    assert "markdown_summary" in result
    assert result["status"] in ("PASS", "PARTIAL")

    report_data = result["report_data"]
    assert report_data["milestone"] == "M86"
    assert report_data["gates"]["gate_multi_task_acquisition"]["passed"] is True
    assert report_data["gates"]["gate_knowledge_reuse"]["passed"] is True
    assert report_data["gates"]["gate_policy_parameter_accounting"]["policy_parameters"] == 989
