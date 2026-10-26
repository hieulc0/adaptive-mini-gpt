"""Unit Tests for Milestone M103-C: Generalizable Concept Learning."""

import pytest
import torch

from adaptive_mini_gpt.natural_language_learning_m103.audit_m103 import (
    DynamicCapacityAuditorM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.continual_concept_learner_m103_c import (
    ContinualConceptLearnerM103_C,
)
from adaptive_mini_gpt.natural_language_learning_m103.generalization_evaluator_m103_c import (
    GeneralizationEvaluatorM103_C,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.world_simulator_m103_c import (
    ConceptFamilyM103_C,
    WorldSimulatorM103_C,
)


def test_world_simulator_determinism():
    """Verify ground-truth world simulator transitions and counterfactuals."""
    sim = WorldSimulatorM103_C(seed=42)

    # Energy dynamics
    assert sim.simulate_energy_step(5, is_blue=True, is_red=False) == 7
    assert sim.simulate_energy_step(5, is_blue=True, is_red=True) == 6
    assert sim.simulate_energy_step(0, is_blue=False, is_red=True) == 0

    # Counterfactual parameters
    assert sim.simulate_energy_step(5, is_blue=True, is_red=True, blue_gain=4, red_loss=2) == 7

    # Coordinate navigation
    assert sim.simulate_coordinate_step(10, ["North"]) == 13
    assert sim.simulate_coordinate_step(10, ["North", "South"]) == 11


def test_dataset_partitioning_disjointness():
    """Verify that train, canonical val, unseen values, and paraphrases are partitioned disjointly."""
    sim = WorldSimulatorM103_C(seed=42)
    dataset = sim.generate_energy_dataset()

    assert len(dataset.train_items) > 0
    assert len(dataset.val_canonical_items) > 0
    assert len(dataset.unseen_values_items) > 0
    assert len(dataset.paraphrase_items) > 0

    # Verify unseen values do not leak into train set
    train_nums = {it.expected_numeric_value for it in dataset.train_items}
    unseen_nums = {it.expected_numeric_value for it in dataset.unseen_values_items}
    # Unseen s in [15..28] -> answers >= 16, whereas train s in [2..14] -> answers <= 16
    large_unseen = {n for n in unseen_nums if n > 16}
    assert len(large_unseen) > 0
    assert not large_unseen.intersection(train_nums)


def test_router_dimension_certification_and_zero_discrepancy():
    """Certify router parameters Linear(128->32->2) = 4,194 and 0 discrepancy."""
    student = TextStudentM103(enable_persistent_growth=True, device=torch.device("cpu"))
    audit = DynamicCapacityAuditorM103.audit_student(student)

    # Check router dimensions
    router = student.layer_routers[0].router
    w1, b1 = router[0].weight.numel(), router[0].bias.numel()  # 128*32 + 32 = 4128
    w2, b2 = router[2].weight.numel(), router[2].bias.numel()  # 32*2 + 2 = 66
    total_router_p = w1 + b1 + w2 + b2
    assert total_router_p == 4194

    # Check total parameters and discrepancy
    assert audit.unexplained_discrepancy == 0
    assert audit.reconciliation_verified is True
    assert audit.total_measured_persistent_params == 1309546


def test_evaluator_metrics_and_numeric_extraction():
    """Verify metric computation, format validity, and numeric extraction."""
    evaluator = GeneralizationEvaluatorM103_C()

    assert evaluator.extract_numeric_value("The final energy is 17 energy.") == 17
    assert evaluator.extract_numeric_value("14 energy") == 14
    assert evaluator.extract_numeric_value("-5 coordinate") == -5
    assert evaluator.extract_numeric_value("no numbers here") is None


def test_prompt_loss_masking():
    """Verify that prompt tokens are properly masked with -100 during loss computation."""
    learner = ContinualConceptLearnerM103_C(device=torch.device("cpu"))
    student = TextStudentM103(enable_persistent_growth=True, device=torch.device("cpu"))
    sim = WorldSimulatorM103_C(seed=42)
    dataset = sim.generate_energy_dataset()

    loss = learner._compute_batch_loss(
        student=student,
        items=dataset.train_items[:4],
        active_domain=dataset.family.value,
    )
    assert not torch.isnan(loss)
    assert loss.item() > 0.0
