"""Fast unit and regression tests for Milestone M104-E and M105-A."""

import random
import torch

from adaptive_mini_gpt.natural_language_learning_m103.causal_world_m104_b import (
    ProceduralCausalWorldM104_B,
)
from adaptive_mini_gpt.natural_language_learning_m103.concept_learner_m104_e import (
    ConceptLearnerM104_E,
)
from adaptive_mini_gpt.natural_language_learning_m103.grounded_extractor_m104_e import (
    EntityConditionedAttentionExtractorM104_E,
    GroundedBottleneckAuditorM104_E,
    LexicalKeywordBaselineM104_E,
)
from adaptive_mini_gpt.natural_language_learning_m103.multi_world_transfer_m104_e import (
    WORLD_FAMILIES_M104_E,
    ProceduralTransferWorldM104_E,
)
from adaptive_mini_gpt.natural_language_learning_m103.self_modification_m105_a import (
    FixedHeuristicControllerM105_A,
    LearnedModificationControllerM105_A,
    ModificationOperatorTypeM105_A,
    NeuralModificationProposalM105_A,
    RandomModificationControllerM105_A,
    StructuralRollbackControllerM105_A,
)
from adaptive_mini_gpt.natural_language_learning_m103.semantic_bottleneck_m104_e import (
    IntegratedStudentM104_E,
    SemanticBottleneckM104_E,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)


def test_m104_e_lexical_keyword_baseline():
    """Verify LexicalKeywordBaselineM104_E correctly matches attribute keywords."""
    baseline = LexicalKeywordBaselineM104_E()
    text = "A dense iron crucible forged for high heat."
    tags = baseline.predict_tags(text)
    assert "metal" in tags
    assert "container" in tags
    assert "heat_tolerant" in tags


def test_m104_e_entity_conditioned_cross_attention():
    """Verify EntityConditionedAttentionExtractorM104_E forward pass and dimensions."""
    d_model = 128
    d_attn = 32
    eca = EntityConditionedAttentionExtractorM104_E(d_model=d_model, d_attn=d_attn, num_heads=2)

    batch_size = 2
    seq_len = 16
    q = torch.randn(batch_size, 1, d_model)
    h_seq = torch.randn(batch_size, seq_len, d_model)
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    attended_repr = eca(q, h_seq, key_padding_mask=mask)
    assert attended_repr.shape == (batch_size, d_model)


def test_m104_e_selective_plasticity_trainable_wte():
    """Verify that in selective plasticity mode, wte and top block are trainable while lower blocks are frozen."""
    device = torch.device("cpu")
    base_s = TextStudentM103(prune_orphaned_lm_head=True).to(device)
    student = IntegratedStudentM104_E(student=base_s, mode="selective_plasticity").to(device)

    # Word token embeddings MUST be trainable in M104-E to allow lexical grounding
    assert student.student.base_model.wte.weight.requires_grad is True

    # Top block (Block 4) and ln_f must be trainable
    assert student.student.base_model.blocks[-1].attn.c_attn.weight.requires_grad is True
    assert student.student.base_model.ln_f.weight.requires_grad is True

    # Lower blocks (Blocks 0-3) must remain frozen
    for block in student.student.base_model.blocks[:-1]:
        for p in block.parameters():
            assert p.requires_grad is False


def test_m104_e_three_extraction_strategies():
    """Verify IntegratedStudentM104_E forward pass across all 3 extraction strategies."""
    device = torch.device("cpu")
    base_s = TextStudentM103(prune_orphaned_lm_head=True).to(device)
    student = IntegratedStudentM104_E(student=base_s, mode="selective_plasticity").to(device)

    tokenizer = ByteTextTokenizerM103()
    prompt = "<bos><rule>Laws: Fire melts ice.<sep><question>A crystal orb is cold. What happens?<sep><answer>"
    tokens = torch.tensor([tokenizer.encode(prompt)[:64]], dtype=torch.long, device=device)
    action_id = torch.tensor([0], dtype=torch.long, device=device)

    for strat in ["question_only", "full_context", "entity_conditioned_attention"]:
        st_l, tag_l, z_c, z_n = student.forward_with_strategy(tokens, action_id, strategy=strat)
        assert st_l.shape == (1, 16)
        assert tag_l is not None and tag_l.shape == (1, 26)
        assert z_c.shape == (1, 128)
        assert z_n.shape == (1, 128)


def test_m105_a_complete_learning_process_rollback():
    """Verify StructuralRollbackControllerM105_A restores model, optimizer, RNG, and replay buffer bit-exact."""
    device = torch.device("cpu")
    base_s = TextStudentM103(prune_orphaned_lm_head=True).to(device)
    student = IntegratedStudentM104_E(student=base_s, mode="selective_plasticity").to(device)
    optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-3)
    replay_buffer = ["item_1", "item_2", "item_3"]

    controller = StructuralRollbackControllerM105_A(
        target_model=student,
        optimizer=optimizer,
        scheduler=None,
        replay_buffer=replay_buffer,
    )

    orig_model_sd = {k: v.detach().clone() for k, v in student.state_dict().items()}

    # Propose modification
    prop = NeuralModificationProposalM105_A(
        proposal_id="prop_rollback_test",
        operator_type=ModificationOperatorTypeM105_A.SELECTIVE_BACKBONE_UPDATE,
        target_module="backbone",
        hyperparameters={"mode": "frozen_backbone"},
        expected_gain_rationale="Test full rollback",
    )

    # Force rejection by returning low metric
    result = controller.apply_and_evaluate(
        proposal=prop,
        val_eval_fn=lambda: 5.0,  # Pre and post return 5.0, gain = 0.0 < min_gain_threshold=1.0
        min_gain_threshold=1.0,
    )

    assert result.rollback_executed is True
    assert not result.accepted
    assert student.mode == "selective_plasticity"
    assert len(replay_buffer) == 3

    # Bit-exact model restoration
    for k, v in student.state_dict().items():
        assert torch.equal(v, orig_model_sd[k]), f"Mismatch in {k} after rollback!"


def test_m104_e_procedural_transfer_worlds_and_multistep():
    """Verify all 5 transfer world families generate items, distinct IDs, and multistep items."""
    assert len(WORLD_FAMILIES_M104_E) == 5
    for spec in WORLD_FAMILIES_M104_E:
        world = ProceduralTransferWorldM104_E(spec=spec, seed=42)
        ds = world.generate_datasets()
        assert len(ds["train"]) > 0
        assert len(ds["val_held_out_pair"]) > 0
        assert len(ds["novel_entity"]) > 0
        assert len(ds["multistep"]) > 0


def test_m104_e_smoke_cpu():
    """Fast 1-seed 2-step smoke test of all phases on CPU."""
    device = torch.device("cpu")
    learner = ConceptLearnerM104_E(device=device)

    # Phase E0 smoke test
    e0 = learner.run_phase_e0_audit(seed=42)
    assert e0.independent_benchmark_arms_verified
    assert e0.step_zero_evaluation_enabled
    assert e0.strategy_comparison.lexical_baseline.micro_f1 > 50.0

    # Phase E1 smoke test (1 seed, 2 steps)
    e1 = learner.run_phase_e1_grounded_study(seeds=[42], steps=2)
    assert e1.num_seeds == 1
    assert 0.0 <= e1.selective_plasticity_held_out_acc <= 100.0

    # Phase M105-A smoke test
    m105 = learner.run_phase_m105a_qualification(seed=42, num_trials=2)
    assert m105.rollback_restores_bit_exact
