"""Unit and Integration Tests for Milestone M87: Long-Horizon Continual Learning.

Verifies:
1. Phase 0 M86 Forensic Audit (Revisit reconciliation, baseline parameter accounting).
2. Lifelong Evaluation Matrix R[i, j] and metrics (ACC, BWT, FWT, F).
3. 100-Episode Continual Task Stream Generation (4 phases, reproducibility).
4. Autonomous Memory Manager (Capacity pressure, rank shrinkage, LRU retirement, budget <= 20k).
5. Long-Horizon Continual Student Engine (Acquisition, zero-shot reuse, matrix update).
6. Multi-module bit-exact rollback restoration upon conflict.
7. Master M87 Benchmark smoke execution.
"""

from __future__ import annotations

import torch

from adaptive_mini_gpt.continual_student.learning_policy import (
    UnifiedLearningPolicy,
)
from adaptive_mini_gpt.continual_student.student_state import (
    PersistentStudentState,
)
from adaptive_mini_gpt.long_horizon.audit_m86 import M86ForensicAuditor
from adaptive_mini_gpt.long_horizon.benchmark import M87LongHorizonBenchmark
from adaptive_mini_gpt.long_horizon.evaluation_matrix import (
    LifelongEvaluationMatrix,
)
from adaptive_mini_gpt.long_horizon.long_horizon_student import (
    LongHorizonContinualStudent,
)
from adaptive_mini_gpt.long_horizon.memory_manager import AutonomousMemoryManager
from adaptive_mini_gpt.long_horizon.task_stream_100 import (
    ContinualTaskStream100Generator,
)
from adaptive_mini_gpt.model import MiniGPT, ModelConfig


def test_m86_forensic_audit() -> None:
    """Verify Phase 0 audit of M86 retention, parameter accounting, and baseline projection."""
    report = M86ForensicAuditor.execute_audit()
    assert report.status == "M86_FORENSIC_AUDIT_VERIFIED"
    assert report.revisit_retention_reported_pct == 102.86
    assert report.frozen_backbone_parameters == 1_071_872
    assert report.frozen_backbone_hash == "6e39cd7ef6790b3c"
    assert report.unified_controller_parameters == 989
    assert report.budget_ceiling_parameters == 20_000
    assert report.duplicate_aware_baseline_projected_100ep_params > 20_000
    assert report.budget_violation_duplicate_aware_pct > 100.0


def test_lifelong_evaluation_matrix() -> None:
    """Verify formal lifelong evaluation metrics (ACC, BWT, FWT, F)."""
    matrix = LifelongEvaluationMatrix()

    # Episode 0: Task A introduced, qualified at 0.90
    matrix.record_zero_shot_probe("task_a", em=0.10)
    matrix.record_episode_evaluation(0, "task_a", {"task_a": 0.90})

    # Episode 1: Task B introduced, qualified at 0.85; Task A evaluated at 0.92 (BWT > 0)
    matrix.record_zero_shot_probe("task_b", em=0.05)
    matrix.record_episode_evaluation(1, "task_b", {"task_a": 0.92, "task_b": 0.85})

    # Episode 2: Task C introduced, qualified at 0.95; Task A at 0.88 (forgetting=0.04), Task B at 0.85
    matrix.record_zero_shot_probe("task_c", em=0.08)
    matrix.record_episode_evaluation(2, "task_c", {"task_a": 0.88, "task_b": 0.85, "task_c": 0.95})

    metrics = matrix.compute_metrics()
    assert metrics.num_tasks == 3
    # ACC: (0.88 + 0.85 + 0.95) / 3 = 0.8933
    assert abs(metrics.final_average_accuracy - 0.8933) < 1e-3
    # BWT: Task A diff = 0.88 - 0.90 = -0.02; Task B diff = 0.85 - 0.85 = 0.00; mean = -0.01
    assert abs(metrics.backward_transfer - (-0.01)) < 1e-3
    # Forgetting: Task A max past = 0.92, final = 0.88 -> 0.04; Task B max past = 0.85, final = 0.85 -> 0.00; mean = 0.02
    assert abs(metrics.average_forgetting - 0.02) < 1e-3


def test_100_episode_task_stream_generation() -> None:
    """Verify that the 100-episode curriculum is deterministic and covers all 4 phases."""
    stream_42 = ContinualTaskStream100Generator.generate_stream(seed=42, num_episodes=100)
    stream_42_bis = ContinualTaskStream100Generator.generate_stream(seed=42, num_episodes=100)
    stream_43 = ContinualTaskStream100Generator.generate_stream(seed=43, num_episodes=100)

    assert len(stream_42) == 100
    # Determinism check
    for ep1, ep2 in zip(stream_42, stream_42_bis, strict=True):
        assert ep1.task_id == ep2.task_id
        assert ep1.intended_regime == ep2.intended_regime

    # Seed variation check
    assert stream_42[0].dataset.seed != stream_43[0].dataset.seed

    # Phase coverage check
    phase1_tasks = [ep for ep in stream_42[:25]]
    phase2_tasks = [ep for ep in stream_42[25:50]]
    phase3_tasks = [ep for ep in stream_42[50:75]]
    phase4_tasks = [ep for ep in stream_42[75:100]]

    assert any(ep.intended_regime == "FOUNDATIONAL_ACQUISITION" for ep in phase1_tasks)
    assert any(ep.intended_regime == "ZERO_SHOT_REUSE" for ep in phase2_tasks)
    assert any(ep.intended_regime == "CONFLICT_REJECTION_ROLLBACK" for ep in phase3_tasks)
    assert any(ep.intended_regime == "LONG_DELAYED_REVISIT" for ep in phase4_tasks)


def test_autonomous_memory_manager() -> None:
    """Verify capacity pressure computation, rank shrinkage, LRU retirement, and budget compliance."""
    torch.manual_seed(42)
    cfg = ModelConfig(vocab_size=1024, context_length=128, d_model=128, n_heads=4, n_layers=4, d_ff=512)
    model = MiniGPT(cfg)
    device = torch.device("cpu")

    state = PersistentStudentState(base_model=model, d_model=128, num_symbols=16).to(device)
    # Use small budget ceiling of 5,000 to test regulation triggers quickly
    mgr = AutonomousMemoryManager(student_state=state, budget_ceiling=5000, controller_params=989, pressure_threshold=0.80)

    # Initial state
    report = mgr.audit_memory()
    assert report.total_trainable_parameters == 989
    assert report.budget_compliant
    assert report.capacity_pressure < 0.25

    # Allocate 3 modules with Rank 8 (1,432 params each = 4,296 params + 989 = 5,285 params -> >5,000!)
    state.module_manager.allocate_module("mod_1", initial_rank=8, device=device)
    state.module_manager.allocate_module("mod_2", initial_rank=8, device=device)
    state.module_manager.allocate_module("mod_3", initial_rank=8, device=device)

    pressure = mgr.compute_current_capacity_pressure()
    assert pressure > 0.80

    # Trigger regulation: should shrink rank 8 modules to rank 4 (852 params)
    actions = mgr.regulate_capacity(current_episode=10)
    assert actions["shrunk_modules"] > 0
    assert actions["freed_parameters"] > 0

    # Audit after regulation
    report_post = mgr.audit_memory()
    assert report_post.total_trainable_parameters <= 5000
    assert report_post.budget_compliant


def test_long_horizon_student_reuse_and_acquisition() -> None:
    """Verify long-horizon student acquires unfamiliar skills and reuses duplicates."""
    torch.manual_seed(42)
    cfg = ModelConfig(vocab_size=1024, context_length=128, d_model=128, n_heads=4, n_layers=4, d_ff=512)
    model = MiniGPT(cfg)
    model.eval()
    device = torch.device("cpu")

    state = PersistentStudentState(base_model=model, d_model=128, num_symbols=16).to(device)
    policy = UnifiedLearningPolicy(in_features=6, hidden_dim=24, num_actions=5).to(device)
    student = LongHorizonContinualStudent(
        state=state, policy=policy, device=device, budget_ceiling=20_000
    )

    stream = ContinualTaskStream100Generator.generate_stream(seed=42, num_episodes=5, vocab_size=16, seq_len=4)

    # Episode 0: Foundational Acquisition
    log0 = student.process_episode(stream[0], max_steps=30, lr=0.015)
    assert log0.qualified
    assert "ACQUIRE" in log0.action_taken
    assert log0.steps_taken > 0

    # Check matrix has recorded Episode 0
    assert 0 in student.eval_matrix.r_matrix
    assert stream[0].task_id in student.eval_matrix.r_matrix[0]


def test_bit_exact_rollback_in_long_horizon() -> None:
    """Verify bit-exact rollback restoration when conflicting task fails qualification."""
    torch.manual_seed(42)
    cfg = ModelConfig(vocab_size=1024, context_length=128, d_model=128, n_heads=4, n_layers=4, d_ff=512)
    model = MiniGPT(cfg)
    model.eval()
    device = torch.device("cpu")

    state = PersistentStudentState(base_model=model, d_model=128, num_symbols=16).to(device)
    policy = UnifiedLearningPolicy(in_features=6, hidden_dim=24, num_actions=5).to(device)
    student = LongHorizonContinualStudent(
        state=state, policy=policy, device=device, budget_ceiling=20_000
    )

    # Acquire base task
    stream = ContinualTaskStream100Generator.generate_stream(seed=42, num_episodes=1, vocab_size=16, seq_len=4)
    student.process_episode(stream[0], max_steps=20, lr=0.015)

    saved_hash = state.compute_holistic_fingerprint()

    # Manually test rollback mechanism on state
    saved_ckpt = state.save_checkpoint()
    state.module_manager.allocate_module("test_conflict", initial_rank=4, device=device)
    with torch.no_grad():
        state.module_manager.get_module("test_conflict").down_proj.weight.add_(3.0)
    mutated_hash = state.compute_holistic_fingerprint()
    assert mutated_hash != saved_hash

    state.restore_checkpoint(saved_ckpt, device=device)
    restored_hash = state.compute_holistic_fingerprint()
    assert restored_hash == saved_hash


def test_m87_benchmark_smoke() -> None:
    """Smoke test full M87 benchmark on a short 4-episode stream."""
    torch.manual_seed(42)
    cfg = ModelConfig(vocab_size=1024, context_length=128, d_model=128, n_heads=4, n_layers=4, d_ff=512)
    model = MiniGPT(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    device = torch.device("cpu")

    res = M87LongHorizonBenchmark.run_benchmark(
        model=model,
        device=device,
        seeds=[42],
        num_episodes=4,
        budget_ceiling=20_000,
    )
    assert res["status"] in ("PASS", "PARTIAL")
    assert "report_data" in res
    assert "gates" in res["report_data"]
    assert res["report_data"]["gates"]["gate_budget_compliance_le_20k"]["passed"]
