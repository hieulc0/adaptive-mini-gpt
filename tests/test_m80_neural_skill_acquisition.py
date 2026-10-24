"""Milestone M80: Unit Tests for Autonomous Neural Skill Acquisition and Transfer.

Fast, CPU-only, deterministic unit tests verifying:
1. Adapter parameter counts, forward shapes, and weight shuffling.
2. Skill gap diagnostic state and detector contracts.
3. Dynamic repertoire expansion, 17-plan hypothesis space, and bit-exact rollback.
4. M79 forensic baseline audit reconciliation.
5. Procedural rules (BITWISE_XOR_MASK, MODULAR_DOUBLING) and transfer task generation.
"""

from __future__ import annotations

import torch

from adaptive_mini_gpt.neural_plan_discovery.plan_space import PlanType
from adaptive_mini_gpt.neural_skill_acquisition.adapter_architecture import (
    EvolvableSkillAdapterFactory,
    Rank8SequenceAdapter,
)
from adaptive_mini_gpt.neural_skill_acquisition.audit_m79 import M79ForensicAuditor
from adaptive_mini_gpt.neural_skill_acquisition.repertoire_expansion import (
    DynamicRepertoireManager,
)
from adaptive_mini_gpt.neural_skill_acquisition.skill_gap_detector import (
    SkillGapDiagnosis,
    SkillGapState,
)
from adaptive_mini_gpt.neural_skill_acquisition.transfer_and_composition_benchmark import (
    BenchmarkTransferTaskGenerator,
    compute_t_test,
)
from adaptive_mini_gpt.neural_skill_composition.primitive_skills import (
    QualifiedPrimitiveRecord,
)
from adaptive_mini_gpt.neural_weight_learning.procedural_tasks import (
    ProceduralRuleFamily,
    apply_procedural_rule,
)
from adaptive_mini_gpt.neural_weight_learning.sequence_adapter import Rank16SequenceAdapter


def test_adapter_architecture_parameter_counts_and_forward() -> None:
    """Verify exact parameter allocation for Rank-16 and Rank-8 sequence adapters."""
    rank16 = EvolvableSkillAdapterFactory.create_adapter(rank=16, d_model=128, num_symbols=16)
    rank8 = EvolvableSkillAdapterFactory.create_adapter(rank=8, d_model=128, num_symbols=16)

    assert isinstance(rank16, Rank16SequenceAdapter)
    assert isinstance(rank8, Rank8SequenceAdapter)

    # Rank-16: ln (256) + down (128*16 + 16 = 2064) + up (16*16 + 16 = 272) = 2592
    assert rank16.count_parameters() == 2592

    # Rank-8: ln (256) + down (128*8 + 8 = 1032) + up (8*16 + 16 = 144) = 1432
    assert rank8.count_parameters() == 1432

    # Forward pass tensor shapes [B=2, L=3, d_model=128] -> [2, 3, 16]
    dummy_input = torch.randn(2, 3, 128)
    out16 = rank16(dummy_input)
    out8 = rank8(dummy_input)

    assert out16.shape == (2, 3, 16)
    assert out8.shape == (2, 3, 16)

    # Parameter hash and shuffling
    hash1 = rank8.compute_parameter_hash()
    assert len(hash1) == 16
    shuffled = rank8.shuffle_weights(seed=42)
    hash2 = shuffled.compute_parameter_hash()
    assert hash1 != hash2, "Shuffling must modify parameter tensor hash"


def test_dynamic_repertoire_expansion_and_rollback() -> None:
    """Verify repertoire expands from 10 to 17 candidate plans and rolls back cleanly."""
    # Create 3 initial dummy primitives
    dummy_prim1 = QualifiedPrimitiveRecord(
        skill_id="prim1",
        rule_family="REFLECT",
        adapter=Rank16SequenceAdapter(),
        qualification_exact_match=0.90,
        qualification_loss=0.25,
        trainable_parameters=2592,
        parameter_hash="hash1",
        training_steps=60,
        is_frozen=True,
    )
    dummy_prim2 = QualifiedPrimitiveRecord(
        skill_id="prim2",
        rule_family="SHIFT3",
        adapter=Rank16SequenceAdapter(),
        qualification_exact_match=0.92,
        qualification_loss=0.22,
        trainable_parameters=2592,
        parameter_hash="hash2",
        training_steps=60,
        is_frozen=True,
    )
    dummy_prim3 = QualifiedPrimitiveRecord(
        skill_id="prim3",
        rule_family="AFFINE",
        adapter=Rank16SequenceAdapter(),
        qualification_exact_match=0.88,
        qualification_loss=0.30,
        trainable_parameters=2592,
        parameter_hash="hash3",
        training_steps=60,
        is_frozen=True,
    )

    initial_prims = {"prim1": dummy_prim1, "prim2": dummy_prim2, "prim3": dummy_prim3}
    mgr = DynamicRepertoireManager(initial_prims)

    # Initial candidate plans: 1 abstain + 3 single + 3*2 = 6 compose = 10 plans
    initial_plans = mgr.generate_expanded_candidate_plans(max_depth=2)
    assert len(initial_plans) == 10
    abstain_plans = [p for p in initial_plans if p.plan_type == PlanType.ABSTAIN]
    single_plans = [p for p in initial_plans if p.plan_type == PlanType.SINGLE_PRIMITIVE]
    compose_plans = [p for p in initial_plans if p.plan_type == PlanType.COMPOSED_TWO_STAGE]
    assert len(abstain_plans) == 1
    assert len(single_plans) == 3
    assert len(compose_plans) == 6

    # Register 4th primitive: expanded plans = 1 abstain + 4 single + 4*3 = 12 compose = 17 plans
    dummy_prim4 = QualifiedPrimitiveRecord(
        skill_id="prim4",
        rule_family="BITWISE_XOR",
        adapter=Rank16SequenceAdapter(),
        qualification_exact_match=0.95,
        qualification_loss=0.15,
        trainable_parameters=2592,
        parameter_hash="hash4",
        training_steps=60,
        is_frozen=True,
    )
    mgr.register_new_skill("prim4", dummy_prim4)
    expanded_plans = mgr.generate_expanded_candidate_plans(max_depth=2)
    assert len(expanded_plans) == 17
    assert len([p for p in expanded_plans if p.plan_type == PlanType.SINGLE_PRIMITIVE]) == 4
    assert len([p for p in expanded_plans if p.plan_type == PlanType.COMPOSED_TWO_STAGE]) == 12

    # Verify rollback
    mgr.rollback_skill("prim4")
    rolled_back_plans = mgr.generate_expanded_candidate_plans(max_depth=2)
    assert len(rolled_back_plans) == 10
    assert "prim4" not in mgr.primitives


def test_skill_gap_diagnosis_data_structure() -> None:
    """Verify SkillGapDiagnosis structure and SkillGapState taxonomy."""
    assert SkillGapState.KNOWN_CONFIDENT == "KNOWN_CONFIDENT"
    assert SkillGapState.KNOWN_AMBIGUOUS == "KNOWN_AMBIGUOUS"
    assert SkillGapState.SKILL_GAP_UNSUPPORTED == "SKILL_GAP_UNSUPPORTED"
    assert SkillGapState.INCONSISTENT_NOISY == "INCONSISTENT_NOISY"

    diag = SkillGapDiagnosis(
        task_id="task_test_01",
        state=SkillGapState.SKILL_GAP_UNSUPPORTED,
        requires_neural_acquisition=True,
        top_candidate_plan="NONE",
        top_likelihood=-2.85,
        evidence_margin=0.01,
        best_token_accuracy=0.12,
        explanation="Test unsupported gap",
    )
    assert diag.requires_neural_acquisition is True
    assert diag.top_likelihood < -1.65


def test_m79_forensic_auditor() -> None:
    """Verify M79 baseline forensic reconciliation and query accounting."""
    audit = M79ForensicAuditor.execute_audit()
    assert audit.status in ("AUDIT_VERIFIED", "PASS")
    assert audit.reconciled_metrics["M79_ACTIVE_DISCRIMINATIVE_EXACT_MATCH"] == 0.9331
    assert audit.reconciled_metrics["M79_PASSIVE_M78_EXACT_MATCH"] == 0.8156
    assert audit.reconciled_metrics["M79_PAIRED_P_VALUE_VS_PASSIVE"] == 0.0375
    assert audit.reconciled_metrics["M79_UNSUPPORTED_QUERIES_WASTED"] == 0.0
    assert "nominal" in audit.oracle_probe_underperformance_explanation.lower() or "oracle" in audit.oracle_probe_underperformance_explanation.lower()


def test_procedural_rules_and_transfer_generation() -> None:
    """Verify BITWISE_XOR_MASK and MODULAR_DOUBLING procedural mappings."""
    # Test BITWISE_XOR_MASK: s -> (s ^ 6) % 16
    inp = (3, 5, 7)
    # 3 ^ 6 = 5; 5 ^ 6 = 3; 7 ^ 6 = 1
    out_xor = apply_procedural_rule(ProceduralRuleFamily.BITWISE_XOR_MASK, inp, vocab_size=16)
    assert out_xor == (5, 3, 1)

    # Test MODULAR_DOUBLING: s -> (2 * s) % 16
    out_double = apply_procedural_rule(ProceduralRuleFamily.MODULAR_DOUBLING, inp, vocab_size=16)
    assert out_double == (6, 10, 14)

    # Test transfer compound task generator
    compound_exs = BenchmarkTransferTaskGenerator.generate_transfer_compound_task(
        task_id="test_compound",
        stage1_name=ProceduralRuleFamily.COMPLEMENT_REFLECT.value,
        stage2_name=ProceduralRuleFamily.BITWISE_XOR_MASK.value,
        seed=42,
        seq_length=3,
        n_train=2,
        n_dev=2,
        n_qual=2,
        n_test=2,
        vocab_size=16,
    )
    assert len(compound_exs) == 8
    # For each example, check that target == (15 - x) ^ 6
    for ex in compound_exs:
        expected = tuple(((15 - s) ^ 6) % 16 for s in ex.input_tokens)
        assert ex.target_tokens == expected


def test_t_test_computation() -> None:
    """Verify paired Student-t test computation utility."""
    sample_a = [0.90, 0.92, 0.94, 0.91, 0.93]
    sample_b = [0.50, 0.52, 0.51, 0.53, 0.50]
    t, p, ci_l, ci_u = compute_t_test(sample_a, sample_b)
    assert t > 10.0
    assert p < 0.01
    assert ci_l > 0.35
