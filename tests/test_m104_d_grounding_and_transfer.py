"""Fast unit and regression tests for Milestone M104-D and M105-A preparation."""

import torch

from adaptive_mini_gpt.natural_language_learning_m103.causal_world_m104_b import (
    ProceduralCausalWorldM104_B,
)
from adaptive_mini_gpt.natural_language_learning_m103.concept_learner_m104_d import (
    ConceptLearnerM104_D,
)
from adaptive_mini_gpt.natural_language_learning_m103.grounded_extractor_m104_d import (
    GroundedBottleneckAuditorM104_D,
)
from adaptive_mini_gpt.natural_language_learning_m103.multi_world_transfer_m104_d import (
    WORLD_FAMILIES_M104_D,
    ProceduralTransferWorldM104_D,
)
from adaptive_mini_gpt.natural_language_learning_m103.self_modification_m105_a import (
    ModificationOperatorTypeM105_A,
    NeuralModificationProposalM105_A,
    StructuralRollbackControllerM105_A,
)
from adaptive_mini_gpt.natural_language_learning_m103.semantic_bottleneck_m104_d import (
    IntegratedStudentM104_D,
    SemanticBottleneckM104_D,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)


def test_m104_d_entity_targeted_sequence_pooling():
    """Verify entity-targeted pooling isolates question tokens from rule explanation tokens."""
    tokenizer = ByteTextTokenizerM103()
    rule_text = "Magical items charge when sparked."
    question_text = "The silver amulet is shiny. What happens when sparked?"
    prompt = f"<bos><rule>{rule_text}<sep><question>{question_text}<sep><answer>"

    tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    seq_len = tokens.size(1)
    d_model = 128
    h_seq = torch.randn(1, seq_len, d_model)

    device = torch.device("cpu")
    base_s = TextStudentM103(prune_orphaned_lm_head=True).to(device)
    student = IntegratedStudentM104_D(student=base_s, mode="selective_plasticity").to(device)

    entity_repr, global_repr = student.pool_representations(h_seq, tokens)

    assert entity_repr.shape == (1, d_model)
    assert global_repr.shape == (1, d_model)
    # Entity representation and global representation must differ because entity excludes rule tokens
    assert not torch.allclose(entity_repr, global_repr, atol=1e-4)


def test_m104_d_compact_bottleneck_and_parameter_accounting():
    """Verify bottleneck footprint is compact (< 80,000 params) and total model < 1.26M params."""
    bottleneck = SemanticBottleneckM104_D(
        d_model=128,
        bottleneck_dim=32,
        action_dim=64,
        num_actions=8,
        num_states=16,
        num_tags=26,
    )
    b_params = sum(p.numel() for p in bottleneck.parameters())
    assert b_params < 80_000, f"Bottleneck too large: {b_params:,} params"

    base_s = TextStudentM103(prune_orphaned_lm_head=True)
    base_params = sum(p.numel() for p in base_s.parameters())
    assert base_params == 1_178_474, f"Unexpected backbone parameter count: {base_params:,}"

    total_params = base_params + b_params
    assert total_params < 1_260_000, f"Total model parameters exceeded budget: {total_params:,}"


def test_m104_d_calibrated_grounding_auditor():
    """Verify GroundedBottleneckAuditorM104_D tracks precision, recall, and 26-class metrics."""
    auditor = GroundedBottleneckAuditorM104_D(device=torch.device("cpu"))
    assert len(auditor.TAG_VOCAB) == 26

    world = ProceduralCausalWorldM104_B(seed=42)
    ds = world.generate_datasets_m104b(candidate_pool_size=28)
    tokenizer = ByteTextTokenizerM103()

    base_s = TextStudentM103(prune_orphaned_lm_head=True)
    student = IntegratedStudentM104_D(student=base_s, mode="selective_plasticity")

    report = auditor.evaluate_model_grounding(
        student_model=student,
        items=ds["val_held_out_pair"][:4],
        objects=world.objects,
        tokenizer=tokenizer,
        decision_threshold=0.50,
    )

    assert report.stage1_extraction.sample_count == 4
    assert len(report.stage1_extraction.per_attribute_metrics) == 26
    assert 0.0 <= report.stage1_extraction.precision <= 100.0
    assert 0.0 <= report.stage1_extraction.recall <= 100.0
    assert 0.0 <= report.stage1_extraction.f1_score <= 100.0


def test_m104_d_multi_world_generator_distinctness():
    """Verify that all 5 transfer world families have distinct IDs, actions, and objects."""
    assert len(WORLD_FAMILIES_M104_D) == 5
    world_ids = [spec.world_id for spec in WORLD_FAMILIES_M104_D]
    assert len(set(world_ids)) == 5, "Duplicate world IDs found"

    for spec in WORLD_FAMILIES_M104_D:
        p_world = ProceduralTransferWorldM104_D(spec=spec, seed=42)
        ds = p_world.generate_datasets()
        assert len(ds["train"]) > 0
        assert len(ds["val_held_out_pair"]) > 0
        assert len(ds["novel_entity"]) > 0
        assert len(p_world.objects) > 0


def test_m104_d_m105a_rollback_exactness():
    """Verify StructuralRollbackControllerM105_A restores bit-exact state on rejected modification."""
    device = torch.device("cpu")
    base_s = TextStudentM103(prune_orphaned_lm_head=True).to(device)
    student = IntegratedStudentM104_D(student=base_s, mode="selective_plasticity").to(device)

    controller = StructuralRollbackControllerM105_A(student)
    orig_state = {k: v.detach().clone() for k, v in student.state_dict().items()}

    proposal = NeuralModificationProposalM105_A(
        proposal_id="test_prop",
        operator_type=ModificationOperatorTypeM105_A.SELECTIVE_BACKBONE_UPDATE,
        target_module="backbone",
        hyperparameters={"mode": "frozen_backbone"},
        expected_gain_rationale="Test rollback exactness",
    )

    eval_calls = [10.0, 5.0]  # Regressed from 10.0 to 5.0 -> negative gain
    result = controller.apply_and_evaluate(
        proposal=proposal,
        eval_fn=lambda: eval_calls.pop(0),
        min_gain_threshold=0.0,
    )

    assert not result.accepted
    assert result.rollback_executed
    assert student.mode == "selective_plasticity"

    # Bit-exact check across all parameters
    for k, v in student.state_dict().items():
        assert torch.equal(v, orig_state[k]), f"Parameter {k} not bit-exact after rollback!"


def test_m104_d_smoke_cpu():
    """Fast 1-step smoke test of all phases on CPU."""
    device = torch.device("cpu")
    learner = ConceptLearnerM104_D(device=device)

    # Phase D0 smoke test
    p0 = learner.run_phase_d0_audit(seed=42)
    assert p0.stage1_fp_audit_completed
    assert p0.m104d_pos_weight == 2.0

    # Phase D1 smoke test (1 seed, 2 steps)
    p1 = learner.run_phase_d1_grounded_study(seeds=[42], steps=2)
    assert p1.num_seeds == 1
    assert 0.0 <= p1.selective_plasticity_held_out_acc <= 100.0

    # Phase M105-A smoke test
    p105 = learner.run_phase_m105a_preview(seed=42)
    assert p105.rollback_restores_bit_exact
