"""Unit tests for Milestone M103-G: Neural Concept Representation & Generalization."""

from pathlib import Path
import tempfile
import torch

from adaptive_mini_gpt.natural_language_learning_m103.capacity_auditor_m103_e1 import (
    DynamicCapacityAuditorM103_E1,
)
from adaptive_mini_gpt.natural_language_learning_m103.causal_world_m103_g import (
    CausalWorldM103_G,
)
from adaptive_mini_gpt.natural_language_learning_m103.concept_learner_m103_g import (
    ConceptLearnerM103_G,
)
from adaptive_mini_gpt.natural_language_learning_m103.neural_state_transition_m103_g import (
    CausalLatentPredictorM103_G,
    ContinuousStateEncoderM103_G,
    NeuralStateTransitionModuleM103_G,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)


def test_capacity_accounting_pruned_m103_g():
    """Verify that active forward-path parameters == 1,178,474 with 0 orphaned parameters."""
    student = TextStudentM103(prune_orphaned_lm_head=True)
    report = DynamicCapacityAuditorM103_E1.audit_student(student)

    assert report.active_forward_path_params == 1_178_474
    assert report.unique_storage_params == 1_178_474
    assert report.orphaned_lm_head_params == 0
    assert report.discrepancy == 0
    assert report.reconciliation_certified is True


def test_causal_world_disjointness_and_dynamics():
    """Verify deterministic transition dynamics and zero train/val overlap in Alchemy World."""
    world = CausalWorldM103_G(seed=42)
    ds = world.generate_datasets()

    assert len(ds["train"]) > 0
    assert len(ds["val_held_out_pair"]) > 0
    assert len(ds["novel_entity"]) > 0
    assert len(ds["multistep"]) > 0
    assert len(ds["counterfactual"]) > 0

    # Test disjointness
    train_pairs = {(it.object_name, it.action_name) for it in ds["train"]}
    val_pairs = {(it.object_name, it.action_name) for it in ds["val_held_out_pair"]}
    assert len(train_pairs.intersection(val_pairs)) == 0

    # Test transition logic
    assert world.transition_state("Torch", "Apply_Fire") == "Ignited"
    assert world.transition_state("Crucible", "Apply_Fire") == "Tempered"
    assert world.transition_state("Torch", "Apply_Frost", "Ignited") == "Extinguished"
    assert world.transition_state("Torch", "Apply_Fire", realm="Shadow") == "Ash"


def test_neural_state_transition_forward():
    """Verify neural state transition module forward pass across both representation modes."""
    # Continuous metric
    mod_cont = NeuralStateTransitionModuleM103_G(hidden_dim=64, num_states=45, mode="continuous_metric")
    states = torch.tensor([2, 5, 10], dtype=torch.long)
    actions = torch.tensor([0, 1, 2], dtype=torch.long)
    logits, z_next = mod_cont(states, actions)
    assert logits.shape == (3, 45)
    assert z_next.shape == (3, 64)

    # Discrete embedding
    mod_disc = NeuralStateTransitionModuleM103_G(hidden_dim=64, num_states=45, mode="discrete_embedding")
    logits_d, z_next_d = mod_disc(states, actions)
    assert logits_d.shape == (3, 45)
    assert z_next_d.shape == (3, 64)


def test_causal_latent_predictor_forward():
    """Verify latent causal predictor forward pass."""
    pred = CausalLatentPredictorM103_G(hidden_dim=64)
    attrs = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    actions = torch.tensor([0, 1], dtype=torch.long)
    logits = pred(attrs, offsets, actions)
    assert logits.shape == (2, 15)


def test_learner_smoke_execution():
    """Smoke test learner methods on CPU with minimal steps."""
    device = torch.device("cpu")
    learner = ConceptLearnerM103_G(device=device)

    # Fast 2-step G1 test
    g1_reports = learner.run_phase_g1_neural_transition_study(steps=2)
    assert "neural_state_transition_continuous" in g1_reports
    assert "neural_state_transition_discrete" in g1_reports
    assert "autoregressive_bytes" in g1_reports

    # Fast 2-step G2 test
    g2_reports = learner.run_phase_g2_causal_concept_study(steps=2)
    assert "latent_state_prediction" in g2_reports
    assert "direct_text_generation" in g2_reports
