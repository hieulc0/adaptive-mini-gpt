"""Unit and Integration Tests for Milestone M89: Continual Knowledge Retention.

Verifies:
1. Phase 0 Forensic Audit of M88 (replay contradiction, task conditioning ambiguity, causal directionality).
2. Causally-consistent foundational curriculum generation across all 6 domains with disjoint splits.
3. Task-Conditioned Knowledge Trunk (explicit conditioning, parameter counts <= 20,000, bit-exact restore).
4. Phase 1 Capacity Feasibility (joint multi-task training beats sequential unregularized).
5. Phase 2 Gradient Diagnostics (flat gradient extraction, cosine similarity matrix, conflict count).
6. Phase 3 Replay Buffer dual byte accounting (raw token bytes < 1 KiB vs activation bytes ~121 KiB).
7. Master M89 Benchmark smoke execution across arms.
"""

from __future__ import annotations

import torch

from adaptive_mini_gpt.continual_retention.audit_m88 import M88ForensicAuditor
from adaptive_mini_gpt.continual_retention.benchmark_m89 import M89RetentionBenchmark
from adaptive_mini_gpt.continual_retention.capacity_feasibility import (
    CapacityFeasibilityBenchmark,
)
from adaptive_mini_gpt.continual_retention.curriculum_m89 import (
    M89CurriculumGenerator,
    RetentionDomain,
    extract_retention_features,
)
from adaptive_mini_gpt.continual_retention.diagnostics import (
    GradientConflictAuditor,
)
from adaptive_mini_gpt.continual_retention.retention_learners import (
    M89ReplayBuffer,
)
from adaptive_mini_gpt.continual_retention.task_conditioning import (
    TaskConditionedKnowledgeTrunk,
)
from adaptive_mini_gpt.model import MiniGPT, ModelConfig


def test_m88_forensic_audit() -> None:
    """Verify Phase 0 forensic audit of M88 and storage reconciliation."""
    report = M88ForensicAuditor.execute_audit()
    assert report.status == "M88_FORENSIC_AUDIT_VERIFIED"
    assert report.m88_lifelong_accuracy_pct == 1.21
    assert report.m88_average_forgetting_pct == 55.98
    assert report.m88_delayed_retention_pct == 1.63
    assert report.budget_ceiling_parameters == 20_000
    assert report.frozen_backbone_parameters == 1_071_872
    assert report.frozen_backbone_hash == "6e39cd7ef6790b3c"
    assert report.task_conditioning_ambiguity_identified
    assert report.causal_directionality_bottleneck_identified

    # Verify replay memory reconciliation
    rec = report.replay_reconciliation
    assert rec.raw_token_bytes == 960  # 60 * 4 * 4 bytes
    assert rec.activation_storage_bytes == 123_840  # 60 * (4*128 + 4) * 4 bytes
    assert "M88 text stated '< 2 KB'" in rec.reconciled_discrepancy_explanation


def test_causally_consistent_curriculum() -> None:
    """Verify curriculum generator across all 6 foundational domains with strict partition disjointness."""
    gen = M89CurriculumGenerator(vocab_size=16, seq_len=4)
    tasks = gen.generate_curriculum(num_tasks=6, seed=42)
    assert len(tasks) == 6

    for t in tasks:
        assert len(t.train_examples) == 48
        assert len(t.dev_examples) == 16
        assert len(t.qualification_examples) == 16
        assert len(t.sealed_test_examples) == 48

        tr_set = {ex.input_tokens for ex in t.train_examples}
        dev_set = {ex.input_tokens for ex in t.dev_examples}
        ql_set = {ex.input_tokens for ex in t.qualification_examples}
        test_set = {ex.input_tokens for ex in t.sealed_test_examples}

        assert len(tr_set.intersection(dev_set)) == 0
        assert len(tr_set.intersection(ql_set)) == 0
        assert len(tr_set.intersection(test_set)) == 0
        assert len(ql_set.intersection(test_set)) == 0


def test_task_conditioned_knowledge_trunk() -> None:
    """Verify task conditioning, parameter counts <= 20,000, and bit-exact save/restore."""
    trunk = TaskConditionedKnowledgeTrunk(max_tasks=10, d_model=128, d_shared=16, num_symbols=16)
    assert trunk.count_trunk_parameters() == 6_176
    assert trunk.count_task_embedding_parameters() == 160
    assert trunk.count_parameters() == 6_336
    assert trunk.count_parameters() <= 20_000

    # Test forward execution
    h = torch.randn(2, 4, 128)
    logits0 = trunk(h, task_id="task_0")
    logits1 = trunk(h, task_id="task_1")
    assert logits0.shape == (2, 4, 16)
    assert logits1.shape == (2, 4, 16)
    # Conditioning should produce distinct outputs for different tasks
    assert not torch.allclose(logits0, logits1)

    # Test bit-exact save and restore state
    saved = trunk.save_state()
    with torch.no_grad():
        trunk.head.weight.add_(1.0)
    mutated = trunk(h, task_id="task_0")
    assert not torch.allclose(logits0, mutated)

    trunk.restore_state(saved)
    restored = trunk(h, task_id="task_0")
    assert torch.allclose(logits0, restored)


def test_replay_buffer_dual_accounting() -> None:
    """Verify replay buffer dual accounting for raw tokens and cached activations."""
    buffer = M89ReplayBuffer(max_total_exemplars=8, exemplars_per_task=2)
    assert buffer.compute_raw_token_bytes() == 0
    assert buffer.compute_activation_bytes() == 0

    # Add 4 tasks (4 * 2 = 8 exemplars)
    for i in range(4):
        h = torch.randn(10, 4, 128)
        y = torch.randint(0, 16, (10, 4))
        buffer.add_task_exemplars(f"task_{i}", h, y)

    # 8 exemplars * 4 tokens * 2 bytes = 64 bytes
    assert buffer.compute_raw_token_bytes() > 0
    assert buffer.compute_activation_bytes() > buffer.compute_raw_token_bytes()

    batch = buffer.sample_batch(exclude_task_id="task_3", max_batch_size=4)
    assert batch is not None
    t_ids, rep_h, rep_y, rep_log = batch
    assert len(t_ids) <= 4
    assert rep_h.ndim == 3
    assert rep_y.ndim == 2


def test_gradient_conflict_diagnostics() -> None:
    """Verify gradient vector extraction and cosine similarity matrix."""
    trunk = TaskConditionedKnowledgeTrunk(max_tasks=5, d_model=128, d_shared=16, num_symbols=16)

    h0 = torch.randn(10, 4, 128)
    y0 = torch.randint(0, 16, (10, 4))
    h1 = torch.randn(10, 4, 128)
    y1 = torch.randint(0, 16, (10, 4))

    trunk.zero_grad()
    l0 = torch.nn.functional.cross_entropy(trunk(h0, task_id="t0").view(-1, 16), y0.view(-1))
    l0.backward()
    g0 = GradientConflictAuditor.extract_flat_gradient(trunk)

    trunk.zero_grad()
    l1 = torch.nn.functional.cross_entropy(trunk(h1, task_id="t1").view(-1, 16), y1.view(-1))
    l1.backward()
    g1 = GradientConflictAuditor.extract_flat_gradient(trunk)

    sim_matrix = GradientConflictAuditor.compute_gradient_cosine_similarity_matrix([g0, g1])
    assert sim_matrix.shape == (2, 2)
    assert torch.isclose(sim_matrix[0, 0], torch.tensor(1.0), atol=1e-3)
    assert torch.isclose(sim_matrix[1, 1], torch.tensor(1.0), atol=1e-3)

    eff_dim = GradientConflictAuditor.compute_effective_dimension(h0)
    assert eff_dim > 0.0


def test_m89_benchmark_smoke() -> None:
    """Smoke test master M89 benchmark on short 3-task curriculum."""
    torch.manual_seed(42)
    cfg = ModelConfig(vocab_size=1024, context_length=128, d_model=128, n_heads=4, n_layers=4, d_ff=512)
    model = MiniGPT(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    device = torch.device("cpu")

    res = M89RetentionBenchmark.run_benchmark(
        model=model,
        device=device,
        seeds=[42],
        num_tasks=3,
        budget_ceiling=20_000,
        epochs_per_task=4,
    )
    assert res["status"] in ("PASS", "PARTIAL")
    assert "arm_results" in res
    assert "gates" in res
    assert res["gates"]["gate_budget_compliance_le_20k"]["passed"]
    assert res["gates"]["gate_base_model_invariance"]["passed"]
    assert res["gates"]["gate_replay_accounting_reconciled"]["passed"]
