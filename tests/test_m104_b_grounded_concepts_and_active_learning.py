"""Fast unit and regression tests for Milestone M104-B: Grounded Concepts and Active Learning."""

import pytest
import torch

from adaptive_mini_gpt.natural_language_learning_m103.active_practice_m104_b import (
    ActivePracticeEnvironmentM104_B,
)
from adaptive_mini_gpt.natural_language_learning_m103.causal_world_m104_b import (
    ProceduralCausalWorldM104_B,
)
from adaptive_mini_gpt.natural_language_learning_m103.concept_learner_m104_b import (
    ConceptLearnerM104_B,
)
from adaptive_mini_gpt.natural_language_learning_m103.semantic_adapter_m104_b import (
    IntegratedSemanticStudentM104_B,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)


def test_m104_b_causal_world_disjointness_and_pools():
    """Verify procedural generation creates disjoint splits and exact candidate pool sizes."""
    world = ProceduralCausalWorldM104_B(seed=42, world_id="test_world")

    ds_28 = world.generate_datasets_m104b(candidate_pool_size=28)
    ds_100 = world.generate_datasets_m104b(candidate_pool_size=100)
    ds_200 = world.generate_datasets_m104b(candidate_pool_size=200)

    assert len(ds_28["active_pool"]) == 28
    assert len(ds_100["active_pool"]) == 100
    assert len(ds_200["active_pool"]) == 200

    tr_pairs = {(x.object_name, x.action_name) for x in ds_100["train"]}
    val_pairs = {(x.object_name, x.action_name) for x in ds_100["val_held_out_pair"]}
    assert len(tr_pairs.intersection(val_pairs)) == 0

    novel_names = {x.object_name for x in ds_100["novel_entity"]}
    tr_names = {x.object_name for x in ds_100["train"]}
    assert len(novel_names.intersection(tr_names)) == 0

    assert len(ds_100["paraphrase_test"]) >= 10
    for p_item in ds_100["paraphrase_test"]:
        assert p_item.paraphrased_question is not None
        assert p_item.paraphrased_question != p_item.question_text


def test_m104_b_semantic_concept_adapter_shapes_and_accounting():
    """Verify semantic concept adapter tensor shapes and parameter accounting."""
    device = torch.device("cpu")
    base_student = TextStudentM103(prune_orphaned_lm_head=True).to(device)
    backbone_params = sum(p.numel() for p in base_student.parameters())
    assert backbone_params == 1_178_474

    integrated = IntegratedSemanticStudentM104_B(student=base_student, mode="integrated_adapter").to(device)
    total_params = sum(p.numel() for p in integrated.parameters())
    adapter_params = sum(p.numel() for p in integrated.adapter.parameters())
    assert total_params == backbone_params + adapter_params
    assert adapter_params > 0

    # Test forward pass
    dummy_input = torch.randint(0, 256, (2, 24), dtype=torch.long, device=device)
    dummy_act = torch.tensor([1, 3], dtype=torch.long, device=device)

    state_logits, tag_logits, z_concept, z_next = integrated(dummy_input, dummy_act)

    num_states = len(ProceduralCausalWorldM104_B.STATES)
    assert state_logits.shape == (2, num_states)
    assert tag_logits.shape == (2, 26)   # 26 property tags
    assert z_concept.shape == (2, 128)    # latent concept dimension
    assert z_next.shape == (2, 128)       # next-state continuous prediction


def test_m104_b_semantic_adapter_ablation_modes():
    """Verify that different ablation modes configure parameter gradients appropriately."""
    base_student = TextStudentM103(prune_orphaned_lm_head=True)

    # 1. Integrated Adapter: all parameters require grad
    m_int = IntegratedSemanticStudentM104_B(student=base_student, mode="integrated_adapter")
    assert any(p.requires_grad for p in m_int.student.parameters())
    assert any(p.requires_grad for p in m_int.adapter.parameters())

    # 2. Frozen Backbone: student backbone parameters are frozen
    m_froz = IntegratedSemanticStudentM104_B(student=base_student, mode="frozen_backbone")
    assert all(not p.requires_grad for p in m_froz.student.parameters())
    assert any(p.requires_grad for p in m_froz.adapter.parameters())

    # 3. No Adapter: adapter is bypassed
    m_no = IntegratedSemanticStudentM104_B(student=base_student, mode="no_adapter")
    dummy_in = torch.randint(0, 256, (2, 16))
    dummy_act = torch.tensor([0, 1])
    s_l, _, _, _ = m_no(dummy_in, dummy_act)
    num_states = len(ProceduralCausalWorldM104_B.STATES)
    assert s_l.shape == (2, num_states)


def test_m104_b_active_practice_environment():
    """Verify active practice simulator query budget isolation and state transition correctness."""
    world = ProceduralCausalWorldM104_B(seed=123)
    ds = world.generate_datasets_m104b(candidate_pool_size=28)
    pool = ds["active_pool"]

    env = ActivePracticeEnvironmentM104_B(
        causal_world=world,
        candidate_pool=pool,
        simulator_budget=5,
        seed=123,
    )

    assert env.simulator_calls_used == 0
    item = pool[0]
    res1 = env.query_simulator(item)
    assert env.simulator_calls_used == 1
    assert isinstance(res1, str)

    # Query until budget exhausted
    for _ in range(4):
        env.query_simulator(pool[1])
    assert env.simulator_calls_used == 5

    with pytest.raises(RuntimeError, match="exhausted"):
        env.query_simulator(pool[2])


def test_m104_b_concept_learner_smoke_cpu():
    """Fast smoke test executing 2 steps of Phase B1 on CPU to ensure end-to-end integration."""
    device = torch.device("cpu")
    learner = ConceptLearnerM104_B(device=device)

    # Run quick 2-step single seed test
    p1 = learner.run_phase_b1_grounded_concept_study(seeds=[42], steps=2)
    assert p1.num_seeds == 1
    assert p1.total_eval_samples > 0
    assert 0.0 <= p1.integrated_mean_held_out_pair_acc <= 100.0
    assert 0.0 <= p1.frozen_backbone_held_out_acc <= 100.0
    assert 0.0 <= p1.no_adapter_backbone_held_out_acc <= 100.0
