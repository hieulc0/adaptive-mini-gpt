"""Comprehensive Unit Test Suite for Milestone M61 (Safe Meta-Control and Causal Learning)."""
from __future__ import annotations

import pytest
import torch

from adaptive_mini_gpt.causal_learning_audit.adapter import PersistentCandidateAdapter
from adaptive_mini_gpt.causal_meta_learning.task_suite import (
    build_development_task_suite,
    build_sealed_meta_test_task_suite,
    create_dev_balanced_hierarchy_environment,
    create_meta_long_chain_aerospace,
)
from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.reliable_transfer.composition_models import ResidualSkillAdapter
from adaptive_mini_gpt.safe_meta_learning import (
    ControllerAction,
    CounterfactualSkillRouter,
    DynamicExperienceMemoryUpdater,
    M60ScientificAuditor,
    M61AblationArm,
    M61SafeguardAuditor,
    SafeCausalAdaptationEngine,
    SafeLearningProgressController,
    build_m61_memory_store,
)


class DummyTokenizer:
    def __init__(self) -> None:
        self.vocab_size = 128

    def encode(self, text: str) -> list[int]:
        return [(ord(c) % 100) + 1 for c in text[:32]]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr((t - 1) % 100) for t in tokens)


@pytest.fixture
def tiny_model() -> MiniGPT:
    cfg = ModelConfig(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        context_length=64,
    )
    return MiniGPT(cfg)


@pytest.fixture
def tokenizer() -> DummyTokenizer:
    return DummyTokenizer()


def test_m60_forensic_audit() -> None:
    """Test 1: Verify forensic audit reproduces M60 failure modes and accounts for premature stopping."""
    auditor = M60ScientificAuditor()
    report = auditor.run_m60_audit()

    assert report.sealed_fixed_40_auc == 0.4340
    assert report.sealed_relevant_history_auc == 0.2385
    assert report.relevant_vs_fixed_gap < 0.0
    assert report.premature_stop_rate >= 0.50
    assert "PREMATURE" in report.causal_memory_failure_root_cause
    assert report.always_source_joint_f1 > report.learned_router_joint_f1
    assert "MISALIGNED OBJECTIVE" in report.routing_discrepancy_root_cause


def test_safe_learning_progress_controller() -> None:
    """Test 2: Verify controller enforces minimum budget and plateaus before stopping."""
    controller = SafeLearningProgressController(min_steps=20, max_steps=40, block_size=10)
    controller.record_initial_state(initial_loss=1.0, initial_f1=0.40)

    # Step 10: Progress continuing; history recommends STOP at 10 steps -> Controller should VETO and CONTINUE
    d1 = controller.evaluate_checkpoint(
        current_step=10,
        val_loss=0.75,
        val_f1=0.60,
        history_recommended_action="STOP",
        history_recommended_steps=10,
    )
    assert d1.action == ControllerAction.CONTINUE
    assert d1.history_vetoed_by_progress is True

    # Step 20: Still progressing
    d2 = controller.evaluate_checkpoint(
        current_step=20,
        val_loss=0.55,
        val_f1=0.72,
        history_recommended_action="TRAIN",
        history_recommended_steps=30,
    )
    assert d2.action == ControllerAction.CONTINUE
    assert d2.history_vetoed_by_progress is False

    # Step 30: Small improvement (first plateau)
    d3 = controller.evaluate_checkpoint(
        current_step=30,
        val_loss=0.54,
        val_f1=0.73,
        history_recommended_action="TRAIN",
        history_recommended_steps=40,
    )
    assert d3.action == ControllerAction.CONTINUE

    # Step 40: Reached max steps -> STOP
    d4 = controller.evaluate_checkpoint(
        current_step=40,
        val_loss=0.535,
        val_f1=0.735,
        history_recommended_action="TRAIN",
        history_recommended_steps=40,
    )
    assert d4.action == ControllerAction.STOP


def test_safe_controller_negative_transfer_reset() -> None:
    """Test 3: Verify controller detects negative transfer and issues RESET action."""
    controller = SafeLearningProgressController(min_steps=20, max_steps=40, block_size=10)
    controller.record_initial_state(initial_loss=0.90, initial_f1=0.50)

    # Step 10: Loss exploded to 1.35 (delta_loss = 0.90 - 1.35 = -0.45)
    d = controller.evaluate_checkpoint(
        current_step=10,
        val_loss=1.35,
        val_f1=0.30,
    )
    assert d.action == ControllerAction.RESET
    assert d.negative_transfer_detected is True


def test_counterfactual_skill_router_utility(tiny_model: MiniGPT, tokenizer: DummyTokenizer) -> None:
    """Test 4: Verify counterfactual router evaluates true task utility across pathways."""
    device = torch.device("cpu")
    d_model = getattr(tiny_model.config, "d_model", getattr(tiny_model.config, "n_embd", 32))
    src_adapter = PersistentCandidateAdapter(d_model=d_model, bottleneck_dim=8, num_classes=2)
    comp_adapter = ResidualSkillAdapter(source_adapter=src_adapter, d_model=d_model, residual_bottleneck_dim=8, num_classes=2)

    router = CounterfactualSkillRouter(
        d_model=d_model,
        source_adapter=src_adapter,
        composite_adapter=comp_adapter,
    ).to(device)

    src_env = create_dev_balanced_hierarchy_environment(seed=42)
    tgt_env = create_meta_long_chain_aerospace(seed=100)

    # Evaluate ALWAYS_SOURCE vs UTILITY_EXPECTED_ROUTER vs EMPIRICAL_BEST
    rep_src = router.evaluate_counterfactual_benchmark(
        base_model=tiny_model,
        tokenizer=tokenizer,
        device=device,
        source_env=src_env,
        target_env=tgt_env,
        policy="ALWAYS_SOURCE",
    )
    assert rep_src.total_queries > 0

    rep_best = router.evaluate_counterfactual_benchmark(
        base_model=tiny_model,
        tokenizer=tokenizer,
        device=device,
        source_env=src_env,
        target_env=tgt_env,
        policy="EMPIRICAL_BEST_ROUTE_BOUND",
    )
    assert rep_best.joint_source_target_f1 >= rep_src.joint_source_target_f1


def test_safe_causal_adaptation_engine(tiny_model: MiniGPT, tokenizer: DummyTokenizer) -> None:
    """Test 5: Verify SafeCausalAdaptationEngine executes safe experiential arm."""
    device = torch.device("cpu")
    engine = SafeCausalAdaptationEngine(
        base_model=tiny_model,
        tokenizer=tokenizer,
        device=device,
    )
    env = create_dev_balanced_hierarchy_environment(seed=42)

    res = engine.run_safe_adaptation(
        env=env,
        arm=M61AblationArm.ARM_F_SAFE_EXPERIENTIAL_PROGRESS,
        source_adapter=None,
        seed=42,
    )

    assert res.total_steps_taken >= 20  # Minimum budget guard enforced
    assert res.learning_curve_auc >= 0.0
    assert len(res.step_history) > 1
    assert res.compute_adjusted_efficiency > 0.0


def test_experience_memory_updater() -> None:
    """Test 6: Verify memory updater commits episodes and tracks accumulation."""
    store = build_m61_memory_store(M61AblationArm.ARM_F_SAFE_EXPERIENTIAL_PROGRESS)
    init_count = len(store.records)
    updater = DynamicExperienceMemoryUpdater(store)

    updater.record_completed_episode(
        task_id="dev_new_task",
        num_nodes=8,
        graph_diameter=5,
        graph_density=0.15,
        structural_complexity_index=7.5,
        steps_spent=30,
        initial_loss=0.72,
        final_loss=0.45,
        final_f1=0.80,
        learning_gain_f1=0.30,
        was_negative_transfer=False,
    )

    assert len(store.records) == init_count + 1


def test_m61_safeguards_audit(tiny_model: MiniGPT, tokenizer: DummyTokenizer) -> None:
    """Test 7: Verify safeguard auditor checks parameter invariance and retention."""
    device = torch.device("cpu")
    auditor = M61SafeguardAuditor(tiny_model, tokenizer, device)
    d_model = getattr(tiny_model.config, "d_model", getattr(tiny_model.config, "n_embd", 32))
    src_adapter = PersistentCandidateAdapter(d_model=d_model, bottleneck_dim=8, num_classes=2)
    src_env = create_dev_balanced_hierarchy_environment(seed=42)

    rep = auditor.audit_all_safeguards(src_env, src_adapter, seed=42)
    assert rep.base_parameter_hash_invariant is True
    assert rep.arithmetic_carry_retention == 1.00
    assert rep.chained_logic_retention == 1.00
    assert rep.sprt_known_false_alarm_rate == 0.0
    assert rep.topology_expansion_status == "RESEARCH_ONLY"
    assert rep.all_safeguards_passed is True
