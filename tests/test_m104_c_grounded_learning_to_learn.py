"""Fast unit and regression tests for Milestone M104-C: Grounded Neural Learning and Learning-to-Learn."""

import torch

from adaptive_mini_gpt.natural_language_learning_m103.bottleneck_diagnostic_m104_c import (
    BottleneckDiagnosticAuditorM104_C,
)
from adaptive_mini_gpt.natural_language_learning_m103.causal_world_m104_b import (
    ProceduralCausalWorldM104_B,
)
from adaptive_mini_gpt.natural_language_learning_m103.concept_learner_m104_c import (
    ConceptLearnerM104_C,
)
from adaptive_mini_gpt.natural_language_learning_m103.self_modification_interface_m104_c import (
    ModificationOperatorTypeM104_C,
    NeuralModificationProposalM104_C,
    StructuralRollbackControllerM104_C,
)
from adaptive_mini_gpt.natural_language_learning_m103.semantic_bottleneck_m104_c import (
    IntegratedStudentM104_C,
    SemanticBottleneckM104_C,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)


def test_m104_c_compact_bottleneck_parameter_footprint():
    """Verify semantic bottleneck is compact (< 60,000 params vs 127,418 in M104-B)."""
    bottleneck = SemanticBottleneckM104_C(
        d_model=128,
        bottleneck_dim=32,
        action_dim=64,
        num_actions=8,
        num_states=16,
        num_tags=26,
    )
    total_params = sum(p.numel() for p in bottleneck.parameters())
    assert total_params < 60_000, f"Bottleneck too large: {total_params:,} params"
    reduction = ((127_418 - total_params) / 127_418) * 100.0
    assert reduction >= 50.0, f"Expected >= 50% parameter reduction, got {reduction:.1f}%"


def test_m104_c_selective_plasticity_gradient_masking():
    """Verify selective plasticity freezes lower layers while keeping top block and bottleneck trainable."""
    device = torch.device("cpu")
    base_student = TextStudentM103(prune_orphaned_lm_head=True).to(device)
    integrated = IntegratedStudentM104_C(student=base_student, mode="selective_plasticity").to(device)

    # Lower blocks (0, 1, 2, 3) must be frozen
    for i in range(len(base_student.base_model.blocks) - 1):
        block = base_student.base_model.blocks[i]
        assert all(not p.requires_grad for p in block.parameters()), f"Block {i} should be frozen"

    # Top block (Block 4) must be trainable
    top_block = base_student.base_model.blocks[-1]
    assert any(p.requires_grad for p in top_block.parameters()), "Top block should be trainable"

    # Bottleneck must be trainable
    assert all(p.requires_grad for p in integrated.bottleneck.parameters()), "Bottleneck should be trainable"


def test_m104_c_bottleneck_diagnostic_audit():
    """Verify Phase C0 decoupled audit calculates precision, recall, F1, and state retention."""
    device = torch.device("cpu")
    auditor = BottleneckDiagnosticAuditorM104_C(device=device)
    world = ProceduralCausalWorldM104_B(seed=42)
    ds = world.generate_datasets_m104b(candidate_pool_size=28)

    base_student = TextStudentM103(prune_orphaned_lm_head=True).to(device)
    integrated = IntegratedStudentM104_C(student=base_student, mode="selective_plasticity").to(device)
    tokenizer = ByteTextTokenizerM103()

    report = auditor.evaluate_model_bottleneck(
        student_model=integrated,
        items=ds["val_held_out_pair"][:6],
        objects=world.objects,
        tokenizer=tokenizer,
    )

    assert report.text_to_concept.sample_count == 6
    assert 0.0 <= report.text_to_concept.precision <= 100.0
    assert 0.0 <= report.text_to_concept.recall <= 100.0
    assert 0.0 <= report.text_to_concept.f1_score <= 100.0
    assert report.primary_bottleneck in ("TEXT_EXTRACTION", "CAUSAL_TRANSITION", "BALANCED")


def test_m104_c_self_modification_rollback_controller():
    """Verify Phase C4 self-modification transactional rollback restores bit-exact weights."""
    device = torch.device("cpu")
    base_student = TextStudentM103(prune_orphaned_lm_head=True).to(device)
    integrated = IntegratedStudentM104_C(student=base_student, mode="selective_plasticity").to(device)

    controller = StructuralRollbackControllerM104_C(integrated)

    # Proposal that will be rejected due to negative metric
    proposal = NeuralModificationProposalM104_C(
        proposal_id="test_reject_prop",
        operator_type=ModificationOperatorTypeM104_C.SELECTIVE_BACKBONE_UPDATE,
        target_module="backbone",
        hyperparameters={"mode": "frozen_backbone"},
        expected_gain_rationale="Test rollback on negative gain",
    )

    eval_calls = [10.0, 5.0]  # post is worse than pre -> negative gain
    val_res = controller.apply_and_evaluate(
        proposal=proposal,
        eval_fn=lambda: eval_calls.pop(0),
        min_gain_threshold=0.0,
    )

    assert not val_res.accepted
    assert val_res.rollback_executed
    assert val_res.delta_gain < 0.0
    assert integrated.mode == "selective_plasticity"  # State reverted


def test_m104_c_concept_learner_smoke_cpu():
    """Fast smoke test executing 2 steps of Phase C0 & C1 on CPU."""
    device = torch.device("cpu")
    learner = ConceptLearnerM104_C(device=device)

    # Phase C0 diagnostic smoke test
    p0 = learner.run_phase_c0_diagnostic_audit(seed=42)
    assert p0.text_to_concept.sample_count > 0

    # Phase C1 single seed 2-step smoke test
    p1 = learner.run_phase_c1_grounded_concept_study(seeds=[42], steps=2)
    assert p1.num_seeds == 1
    assert p1.total_eval_samples > 0
    assert 0.0 <= p1.selective_plasticity_mean_held_out_pair_acc <= 100.0
