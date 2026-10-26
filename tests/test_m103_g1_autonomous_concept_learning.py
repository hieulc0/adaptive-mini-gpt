"""Unit tests for Milestone M103-G.1 & M104-A: Autonomous Concept Learning & Active Practice."""

import pytest
import torch

from adaptive_mini_gpt.natural_language_learning_m103.active_practice_m104_a import (
    ActivePracticeEnvironmentM104_A,
    TrainablePolicyNetworkM104_A,
)
from adaptive_mini_gpt.natural_language_learning_m103.baselines_m103_g1 import (
    AttributeOnlyHeuristicBaselineM103_G1,
    MajorityClassBaselineM103_G1,
    MatchedLinearProbeBaselineM103_G1,
    NearestNeighborBaselineM103_G1,
)
from adaptive_mini_gpt.natural_language_learning_m103.capacity_auditor_m103_e1 import (
    DynamicCapacityAuditorM103_E1,
)
from adaptive_mini_gpt.natural_language_learning_m103.causal_world_m103_g1 import (
    ProceduralCausalWorldM103_G1,
)
from adaptive_mini_gpt.natural_language_learning_m103.concept_learner_m103_g1 import (
    ConceptLearnerM103_G1,
    IntegratedCausalStudentM103_G1,
    StandaloneLatentPredictorM103_G1,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)


def test_capacity_accounting_m103_g1():
    """Verify persistent student capacity: exactly 1,178,474 active parameters and 0 discrepancy."""
    student = TextStudentM103(prune_orphaned_lm_head=True)
    report = DynamicCapacityAuditorM103_E1.audit_student(student)

    assert report.active_forward_path_params == 1_178_474
    assert report.unique_storage_params == 1_178_474
    assert report.orphaned_lm_head_params == 0
    assert report.discrepancy == 0
    assert report.reconciliation_certified is True


def test_procedural_causal_world_disjointness_and_scale():
    """Verify zero train/val leakage and >= 100 held-outs across 5 independent seeds."""
    for seed in [42, 43, 44, 45, 46]:
        world = ProceduralCausalWorldM103_G1(seed=seed)
        ds = world.generate_datasets()

        assert len(ds["train"]) > 0
        assert len(ds["val_held_out_pair"]) > 0
        assert len(ds["novel_entity"]) > 0
        assert len(ds["multistep"]) > 0
        assert len(ds["counterfactual"]) > 0
        assert len(ds["active_pool"]) > 0

        # Disjointness checks
        tr_pairs = {(x.object_name, x.action_name) for x in ds["train"]}
        val_pairs = {(x.object_name, x.action_name) for x in ds["val_held_out_pair"]}
        assert len(tr_pairs.intersection(val_pairs)) == 0

        tr_objs = {x.object_name for x in ds["train"]}
        nov_objs = {x.object_name for x in ds["novel_entity"]}
        assert len(tr_objs.intersection(nov_objs)) == 0

        # Scale check: >= 100 held-out cases per seed
        total_held_out = len(ds["val_held_out_pair"]) + len(ds["novel_entity"]) + len(ds["multistep"]) + len(ds["counterfactual"])
        assert total_held_out >= 100


def test_baselines_prediction():
    """Verify non-learning and matched neural baselines execute correctly."""
    world = ProceduralCausalWorldM103_G1(seed=42)
    ds = world.generate_datasets()
    train_items = ds["train"]
    test_item = ds["val_held_out_pair"][0]
    test_obj = world.objects[test_item.object_name]

    # Majority-class
    maj = MajorityClassBaselineM103_G1()
    maj.fit(train_items)
    pred_maj = maj.predict(test_item, test_obj)
    assert isinstance(pred_maj, str)
    assert pred_maj in ProceduralCausalWorldM103_G1.STATES

    # Attribute-only heuristic
    attr_heur = AttributeOnlyHeuristicBaselineM103_G1()
    pred_attr = attr_heur.predict(test_item, test_obj)
    assert isinstance(pred_attr, str)

    # Nearest-neighbor
    nn_base = NearestNeighborBaselineM103_G1()
    nn_base.fit(train_items, world.objects)
    pred_nn = nn_base.predict(test_item, test_obj)
    assert isinstance(pred_nn, str)

    # Matched linear probe
    lin = MatchedLinearProbeBaselineM103_G1(num_tags=26, num_actions=7, num_states=16)
    tag_v = torch.zeros(2, 26)
    act_v = torch.zeros(2, 7)
    out_lin = lin(tag_v, act_v)
    assert out_lin.shape == (2, 16)


def test_integrated_causal_student_forward():
    """Verify forward pass of student-integrated causal head and standalone predictor."""
    base_student = TextStudentM103(prune_orphaned_lm_head=True)
    integrated = IntegratedCausalStudentM103_G1(student=base_student, num_actions=7, num_states=16)

    prompts = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
    actions = torch.tensor([0, 1], dtype=torch.long)
    logits, z_next = integrated(prompts, actions)

    assert logits.shape == (2, 16)
    assert z_next.shape == (2, 128)

    # Standalone predictor
    standalone = StandaloneLatentPredictorM103_G1(num_tags=26, num_actions=7, num_states=16, hidden_dim=64)
    tags = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    acts = torch.tensor([0, 1], dtype=torch.long)
    out_std = standalone(tags, offsets, acts)
    assert out_std.shape == (2, 16)


def test_active_practice_environment_and_budget_enforcement():
    """Verify M104-A active practice environment query logic and strict budget enforcement."""
    world = ProceduralCausalWorldM103_G1(seed=42)
    ds = world.generate_datasets()
    pool = ds["active_pool"][:5]

    env = ActivePracticeEnvironmentM104_A(
        causal_world=world,
        candidate_pool=pool,
        simulator_budget=2,
        seed=42,
    )
    assert env.simulator_calls_used == 0

    res1 = env.query_simulator(pool[0])
    assert isinstance(res1, str)
    assert env.simulator_calls_used == 1

    res2 = env.query_simulator(pool[1])
    assert isinstance(res2, str)
    assert env.simulator_calls_used == 2

    # Third query should raise budget exhausted error
    with pytest.raises(RuntimeError, match="budget exhausted"):
        env.query_simulator(pool[2])


def test_trainable_policy_network_forward():
    """Verify trainable policy network forward pass."""
    net = TrainablePolicyNetworkM104_A(hidden_dim=128)
    z_cand = torch.randn(5, 128)
    feats = torch.randn(5, 2)
    scores = net(z_cand, feats)
    assert scores.shape == (5,)


def test_smoke_learner_phases_cpu():
    """Smoke test learner methods on CPU with 2 steps for fast local unit validation."""
    device = torch.device("cpu")
    learner = ConceptLearnerM103_G1(device=device)

    # Fast Phase 1 smoke test (1 seed, 2 steps)
    p1 = learner.run_phase1_requalification(seeds=[42], steps=2)
    assert p1.num_seeds == 1
    assert p1.integrated_student_mean_held_out_pair_acc >= 0.0

    # Fast Phase 2 smoke test (2 steps, 2 feedback episodes)
    p2 = learner.run_phase2_corrective_learning(seed=42, steps=2, feedback_episodes=2)
    assert isinstance(p2.unprotected_backward_retention, float)

    # Fast Phase 3 smoke test (2 simulator calls)
    p3 = learner.run_phase3_active_practice(seed=42, simulator_budget=2, eval_interval=2)
    assert len(p3.trajectories) == 4
