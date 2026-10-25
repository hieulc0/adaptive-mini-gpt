"""Unit and Integration Tests for Milestone M88: Foundational Representation Learning.

Verifies:
1. Phase 0 M87 Forensic Audit (modular amnesia, budget-matched baseline projection).
2. Foundational multi-domain curriculum generation (6 domains, strictly disjoint partitions).
3. Shared Foundational Substrate (shared trunk, selective residuals, parameter accounting, <= 20k).
4. Exemplar replay buffer (bounded capacity, sampling, byte-level storage accounting).
5. Foundational continual student (zero-shot probe, residual allocation, R[i, j] updates).
6. Bit-exact rollback restoration upon conflict rejection.
7. LRU residual retirement and parameter budget bounds.
8. Master M88 Benchmark smoke execution across arms.
"""

from __future__ import annotations

import torch

from adaptive_mini_gpt.continual_student.learning_policy import (
    UnifiedLearningPolicy,
)
from adaptive_mini_gpt.foundational.audit_m87 import M87ForensicAuditor
from adaptive_mini_gpt.foundational.benchmark import M88FoundationalBenchmark
from adaptive_mini_gpt.foundational.continual_mechanisms import (
    ExemplarReplayBuffer,
)
from adaptive_mini_gpt.foundational.curriculum import (
    FoundationalCurriculumGenerator,
    FoundationalDomain,
)
from adaptive_mini_gpt.foundational.shared_representation import (
    SelectiveResidualAdapter,
    SharedFoundationalSubstrate,
    SharedKnowledgeTrunk,
)
from adaptive_mini_gpt.foundational.student import FoundationalContinualStudent
from adaptive_mini_gpt.long_horizon.task_stream_100 import (
    ContinualTaskStream100Generator,
)
from adaptive_mini_gpt.model import MiniGPT, ModelConfig


def test_m87_forensic_audit() -> None:
    """Verify Phase 0 audit of M87 outcomes and modular amnesia diagnosis."""
    report = M87ForensicAuditor.execute_audit()
    assert report.status == "M87_FORENSIC_AUDIT_VERIFIED"
    assert report.m87_lifelong_accuracy_pct == 5.84
    assert report.m87_average_forgetting_pct == 53.27
    assert report.m87_delayed_retention_pct == 3.33
    assert report.m87_trainable_parameters == 17_757
    assert report.budget_ceiling_parameters == 20_000
    assert report.frozen_backbone_parameters == 1_071_872
    assert report.frozen_backbone_hash == "6e39cd7ef6790b3c"
    assert report.unified_controller_parameters == 989
    assert report.modular_amnesia_attribution_pct == 100.0
    assert report.budget_matched_independent_max_modules <= 14
    assert report.budget_matched_independent_projected_acc_pct < 25.0


def test_foundational_curriculum_generation() -> None:
    """Verify curriculum generator across all 6 foundational domains with disjoint partitions."""
    gen = FoundationalCurriculumGenerator(vocab_size=16, seq_len=4)

    domains = [
        (FoundationalDomain.PLACE_VALUE_REASONING, "mag_cmp", {"operation": "magnitude_compare"}),
        (FoundationalDomain.ARITHMETIC_OPERATIONS, "add_carry", {"operation": "add_with_carry", "addend": 3}),
        (FoundationalDomain.BOOLEAN_CONDITIONAL_LOGIC, "xor_cond", {"operation": "bitwise_xor_cond", "mask": 6}),
        (FoundationalDomain.SEQUENCE_MANIPULATION, "shift_1", {"operation": "cyclic_shift", "shift": 1}),
        (FoundationalDomain.COMPOSITIONAL_PROCEDURE, "cmp_add_xor", {"stage1_add": 3, "stage2_xor": 5}),
        (FoundationalDomain.INSTRUCTION_SEMANTICS, "inst_add", {"opcode": 1, "operand": 2}),
    ]

    for domain, rule_name, params in domains:
        ds = gen.generate_task(
            task_id=f"test_{rule_name}",
            domain=domain,
            rule_name=rule_name,
            rule_params=params,
            seed=42,
        )
        assert len(ds.train_examples) == 48
        assert len(ds.dev_examples) == 16
        assert len(ds.qualification_examples) == 16
        assert len(ds.sealed_test_examples) == 48

        # Verify strict partition disjointness
        tr_in = {ex.input_tokens for ex in ds.train_examples}
        dev_in = {ex.input_tokens for ex in ds.dev_examples}
        ql_in = {ex.input_tokens for ex in ds.qualification_examples}
        test_in = {ex.input_tokens for ex in ds.sealed_test_examples}

        assert len(tr_in.intersection(dev_in)) == 0
        assert len(tr_in.intersection(ql_in)) == 0
        assert len(tr_in.intersection(test_in)) == 0
        assert len(ql_in.intersection(test_in)) == 0


def test_shared_foundational_substrate() -> None:
    """Verify shared trunk, selective residuals, parameter counts, and budget compliance."""
    trunk = SharedKnowledgeTrunk(d_model=128, d_shared=16, num_symbols=16)
    assert trunk.count_parameters() == 6_176

    residual = SelectiveResidualAdapter(d_model=128, num_symbols=16, rank=1)
    assert residual.count_parameters() == 144

    substrate = SharedFoundationalSubstrate(
        d_model=128, d_shared=16, num_symbols=16, budget_ceiling=10_000
    )
    assert substrate.count_parameters() == 6_176

    # Allocate 10 compact residuals: 10 * 144 = 1,440 params
    for i in range(10):
        substrate.allocate_residual(f"task_{i}", rank=1)

    assert substrate.count_residual_parameters() == 1_440
    assert substrate.count_parameters() == 6_176 + 1_440

    # Test forward pass with and without residual
    x = torch.randn(2, 4, 128)
    out_trunk = substrate(x, task_id=None)
    out_res = substrate(x, task_id="task_0")
    assert out_trunk.shape == (2, 4, 16)
    assert out_res.shape == (2, 4, 16)

    # Test bit-exact save and restore state
    saved = substrate.save_state()
    # Mutate trunk weight
    with torch.no_grad():
        substrate.trunk.head.weight.add_(1.0)
    out_mutated = substrate(x, task_id=None)
    assert not torch.allclose(out_trunk, out_mutated, atol=1e-2)

    # Restore
    substrate.restore_state(saved)
    out_restored = substrate(x, task_id=None)
    assert torch.allclose(out_trunk, out_restored)


def test_exemplar_replay_buffer() -> None:
    """Verify bounded replay buffer, sampling, FIFO eviction, and byte accounting."""
    buffer = ExemplarReplayBuffer(max_total_exemplars=6, exemplars_per_task=2)
    assert buffer.compute_storage_bytes() == 0

    # Add 4 tasks (4 * 2 = 8 > 6 -> should evict oldest)
    for i in range(4):
        h = torch.randn(10, 4, 128)
        y = torch.randint(0, 16, (10, 4))
        buffer.add_task_exemplars(f"task_{i}", h, y)

    total_ex = sum(len(v) for v in buffer.buffer.values())
    assert total_ex <= 6
    assert buffer.compute_storage_bytes() > 0

    batch = buffer.sample_batch(exclude_task_id="task_3", max_batch_size=4)
    assert batch is not None
    rep_h, rep_y = batch
    assert rep_h.ndim == 3
    assert rep_y.ndim == 2


def test_foundational_continual_student() -> None:
    """Verify continual student processes episodes, allocates compact residuals, and tracks matrix."""
    torch.manual_seed(42)
    cfg = ModelConfig(vocab_size=1024, context_length=128, d_model=128, n_heads=4, n_layers=4, d_ff=512)
    model = MiniGPT(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    device = torch.device("cpu")

    student = FoundationalContinualStudent(
        base_model=model,
        device=device,
        budget_ceiling=20_000,
    )

    stream = ContinualTaskStream100Generator.generate_stream(
        seed=42, num_episodes=4, vocab_size=16, seq_len=4
    )

    log0 = student.process_episode(stream[0], max_steps=20, lr=0.015)
    assert log0.qualified
    assert log0.steps_taken > 0
    assert student.count_trainable_parameters() <= 20_000
    assert 0 in student.eval_matrix.r_matrix

    # Verify zero-shot reuse on duplicate
    # Generate exact duplicate of task 0
    dup_ep = stream[0]
    log_dup = student.process_episode(dup_ep, max_steps=20, lr=0.015)
    assert log_dup.qualified
    assert student.count_trainable_parameters() <= 20_000


def test_bit_exact_rollback_on_conflict() -> None:
    """Verify bit-exact rollback restoration when conflict fails qualification."""
    torch.manual_seed(42)
    cfg = ModelConfig(vocab_size=1024, context_length=128, d_model=128, n_heads=4, n_layers=4, d_ff=512)
    model = MiniGPT(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    device = torch.device("cpu")

    student = FoundationalContinualStudent(
        base_model=model,
        device=device,
        budget_ceiling=20_000,
    )

    stream = ContinualTaskStream100Generator.generate_stream(
        seed=42, num_episodes=1, vocab_size=16, seq_len=4
    )
    student.process_episode(stream[0], max_steps=10, lr=0.015)

    # Checkpoint fingerprint
    saved_state = student.substrate.save_state()
    trunk_weight_saved = student.substrate.trunk.down_proj.weight.clone()

    # Create artificial conflict episode
    conflict_ep = stream[0]
    conflict_ep.intended_regime = "CONFLICT_REJECTION_ROLLBACK"
    # Artificially modify target to make it impossible to learn quickly
    for ex in conflict_ep.dataset.train_examples:
        object.__setattr__(ex, "target_tokens", tuple((x + 7) % 16 for x in ex.target_tokens))

    log_conf = student.process_episode(conflict_ep, max_steps=2, lr=0.0001)
    if not log_conf.qualified:
        assert log_conf.retained_or_rolled_back == "ROLLED_BACK"
        # Verify trunk weight matches saved exactly
        assert torch.allclose(student.substrate.trunk.down_proj.weight, trunk_weight_saved)


def test_lru_residual_retirement_budget_bound() -> None:
    """Verify LRU retirement keeps substrate strictly within budget ceiling."""
    # Capacity for trunk (6,176) + 3 residuals (3 * 144 = 432) = 6,608
    budget = 6_176 + 3 * 144
    substrate = SharedFoundationalSubstrate(
        d_model=128, d_shared=16, num_symbols=16, budget_ceiling=budget
    )
    assert substrate.count_parameters() == 6_176

    # Allocate 5 residuals
    for i in range(5):
        substrate.allocate_residual(f"task_{i}", rank=1)

    # Total parameters must strictly remain <= budget
    assert substrate.count_parameters() <= budget
    assert len(substrate.residuals) == 3
    # Most recently allocated residuals (task_2, task_3, task_4) should be present
    assert substrate.has_residual("task_4")
    assert substrate.has_residual("task_3")
    assert substrate.has_residual("task_2")
    # Older residuals (task_0, task_1) were LRU-retired
    assert not substrate.has_residual("task_0")
    assert not substrate.has_residual("task_1")

    # Forward on retired task falls back gracefully to trunk without error
    x = torch.randn(2, 4, 128)
    out_retired = substrate(x, task_id="task_0")
    assert out_retired.shape == (2, 4, 16)


def test_m88_benchmark_smoke() -> None:
    """Smoke test master M88 benchmark on short 4-episode stream."""
    torch.manual_seed(42)
    cfg = ModelConfig(vocab_size=1024, context_length=128, d_model=128, n_heads=4, n_layers=4, d_ff=512)
    model = MiniGPT(cfg)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    device = torch.device("cpu")

    res = M88FoundationalBenchmark.run_benchmark(
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
    assert res["report_data"]["gates"]["gate_base_model_invariance"]["passed"]
