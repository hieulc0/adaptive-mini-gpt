"""Milestone M82: Fast CPU-Only Unit Tests for Meta-Acquisition Suite.

Verifies:
1. M81 forensic audit reconciliation (30% reduction, df=4 statistics, target gates).
2. Meta-environment generation, procedural rules, and strict 48/16/16/48 partition isolation.
3. LearnedAcquisitionController architecture, parameter budget (< 2,000 params), and action selection.
4. Non-leaking diagnostic state representation and safety stopping guards.
5. Bit-exact adapter rollback and parameter hash invariance.
"""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn

from adaptive_mini_gpt.learning_to_learn.capacity_adaptation import BottleneckSequenceAdapter
from adaptive_mini_gpt.meta_acquisition.audit_m81 import M81ForensicAuditor
from adaptive_mini_gpt.meta_acquisition.controller import (
    AcquisitionAction,
    AcquisitionState,
    InitChoice,
    LearnedAcquisitionController,
    LRChoice,
    RankChoice,
    StoppingChoice,
)
from adaptive_mini_gpt.meta_acquisition.environment import (
    MetaEnvironmentGenerator,
    SplitType,
    TaskFamily,
    apply_procedural_rule,
)


def test_m81_forensic_audit_and_reconciliation() -> None:
    """Verify that M81 forensic auditor verifies reported values and sets M82 gates."""
    audit = M81ForensicAuditor.execute_audit()
    assert audit.status == "M81_AUDIT_COMPLETED"
    assert audit.verdict == "PASS_VERIFIED_PROCEED_TO_M82"

    metrics = audit.m81_reported_metrics
    assert metrics["fresh_median_steps"] == 50
    assert metrics["experienced_median_steps"] == 35
    assert abs(metrics["relative_step_reduction_pct"] - 30.0) < 1e-3

    stats = audit.statistical_verification_df4
    assert stats["steps_df"] == 4
    assert stats["steps_p_value"] == 0.0061
    assert stats["steps_t_statistic"] == 5.3079
    assert stats["steps_ci_95"] == (8.59, 27.41)

    assert "primary_gate_step_reduction" in audit.m82_target_gates
    assert "retention_invariant" in audit.m82_target_gates


def test_meta_environment_generation_and_partition_isolation() -> None:
    """Verify procedural task generation with strictly disjoint 48/16/16/48 partitions."""
    gen = MetaEnvironmentGenerator(vocab_size=16, seq_len=4)

    task = gen.generate_task(
        task_id="test_task_001",
        family=TaskFamily.PALINDROME_REFLECT,
        split=SplitType.META_TEST,
        rule_params={"alt_mask": 5},
        rule_instance_id="PALINDROME_M5",
        seed=42,
    )

    assert len(task.train_examples) == 48
    assert len(task.dev_examples) == 16
    assert len(task.qualification_examples) == 16
    assert len(task.sealed_test_examples) == 48

    # Verify input token disjointness across all 4 sets
    train_in = {ex.input_tokens for ex in task.train_examples}
    dev_in = {ex.input_tokens for ex in task.dev_examples}
    qual_in = {ex.input_tokens for ex in task.qualification_examples}
    test_in = {ex.input_tokens for ex in task.sealed_test_examples}

    assert len(train_in & dev_in) == 0
    assert len(train_in & qual_in) == 0
    assert len(train_in & test_in) == 0
    assert len(dev_in & qual_in) == 0
    assert len(dev_in & test_in) == 0
    assert len(qual_in & test_in) == 0

    # Verify procedural transformation logic for PALINDROME_REFLECT
    # Reverse sequence and XOR alt_mask on even indices (0, 2)
    # Input: (1, 2, 3, 4) -> reversed: (4, 3, 2, 1)
    # index 0: 4 ^ 5 = 1
    # index 1: 3
    # index 2: 2 ^ 5 = 7
    # index 3: 1
    out = apply_procedural_rule(TaskFamily.PALINDROME_REFLECT, {"alt_mask": 5}, (1, 2, 3, 4), 16)
    assert out == (1, 3, 7, 1)


def test_learned_acquisition_controller_architecture() -> None:
    """Verify controller parameter count, forward pass, action sampling, and hash."""
    controller = LearnedAcquisitionController(in_features=8, hidden_dim=32)

    # 1. Parameter budget verification (< 2,000 parameters)
    param_count = controller.count_parameters()
    assert param_count < 2000, f"Controller has {param_count} parameters, exceeding 2,000 budget!"
    assert param_count == 1835

    # 2. Forward pass dimensions
    dummy_state = AcquisitionState(
        probe_loss=1.5,
        loss_slope=0.2,
        prediction_entropy=1.8,
        grad_norm=0.5,
        replay_loss=0.1,
        progress_ratio=0.25,
        candidate_capacity_ratio=0.5,
        token_acc=0.6,
    )
    state_tensor = dummy_state.to_tensor()
    assert state_tensor.shape == (8,)

    init_logits, lr_logits, rank_logits, stop_logits = controller(state_tensor)
    assert init_logits.shape == (1, 2)
    assert lr_logits.shape == (1, 4)
    assert rank_logits.shape == (1, 3)
    assert stop_logits.shape == (1, 2)

    # 3. Action selection
    action = controller.select_action(dummy_state, deterministic=True, qual_em_so_far=0.90)
    assert isinstance(action.init_choice, InitChoice)
    assert isinstance(action.lr_choice, LRChoice)
    assert isinstance(action.rank_choice, RankChoice)
    assert isinstance(action.stopping_choice, StoppingChoice)

    # 4. Hash reproducibility
    h1 = controller.compute_parameter_hash()
    h2 = controller.compute_parameter_hash()
    assert h1 == h2
    assert len(h1) == 16


def test_state_computation_and_safety_invariants() -> None:
    """Verify state normalization and safety stopping guards."""
    controller = LearnedAcquisitionController(in_features=8, hidden_dim=32)

    state = AcquisitionState(
        probe_loss=0.1,
        loss_slope=0.5,
        prediction_entropy=0.2,
        grad_norm=0.1,
        replay_loss=0.0,
        progress_ratio=0.5,
        candidate_capacity_ratio=0.5,
        token_acc=0.9,
    )

    # If qual_em_so_far < 0.85, stopping decision MUST be overridden to CONTINUE
    action = controller.select_action(state, deterministic=False, qual_em_so_far=0.75, admission_threshold=0.85)
    assert action.stopping_choice == StoppingChoice.CONTINUE


def test_bit_exact_rollback_and_parameter_accounting() -> None:
    """Verify bit-exact state restoration upon trial rejection."""
    adapter = BottleneckSequenceAdapter(d_model=128, bottleneck_dim=8, num_symbols=16)
    initial_hash = adapter.compute_parameter_hash()
    saved_state = copy.deepcopy(adapter.state_dict())

    # Apply perturbation
    with torch.no_grad():
        for p in adapter.parameters():
            p.add_(torch.randn_like(p))

    perturbed_hash = adapter.compute_parameter_hash()
    assert perturbed_hash != initial_hash

    # Rollback
    adapter.load_state_dict(saved_state)
    restored_hash = adapter.compute_parameter_hash()
    assert restored_hash == initial_hash
