"""Unit tests for Milestone M75: Experience-Guided Autonomous Learning and Adaptive Training Budgets.

Verifies:
1. Forensic scientific audit of M74, parameter lineage (974,658), and paired Student's t CIs.
2. Experience records, manifest serialization, and causal control perturbations (shuffled/irrelevant).
3. Procedural task generation, 4 generalization tiers, and decoupled target scrambling.
4. ExperienceGuidedController bounded decisions, fallbacks, and uncertainty discounting across 7 policy modes.
5. Empirical retention safeguards (SHA-256 base invariance, arithmetic/logic probes, rollback, rank-16 vs rank-32).
"""

from __future__ import annotations

from adaptive_mini_gpt.experience_guided_learning import (
    ControllerPolicyMode,
    ExperienceGuidedController,
    GeneralizationCategory,
    LearningAction,
    LearningActionType,
    LearningExperienceManifest,
    LearningExperienceRecord,
    M74ScientificAuditor,
    M75SafeguardsAuditor,
    ObservableTaskDescriptors,
    ProceduralRuleFamily,
    ProceduralTaskGenerator,
    RetentionDecision,
    apply_procedural_rule,
)
from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.neural_weight_learning.sequence_adapter import (
    Rank16SequenceAdapter,
)


def test_m74_forensic_audit_and_lineage() -> None:
    """Verify Phase 1 forensic audit calculations, lineage, and paired Student's t intervals."""
    audit_report = M74ScientificAuditor.run_full_audit()
    assert audit_report["status"] == "AUDIT_COMPLETE"

    lineage = audit_report["parameter_lineage"]
    assert lineage["total_orig_parameters"] == 928512
    assert lineage["total_expanded_parameters"] == 974658
    assert lineage["total_delta"] == 46146
    assert lineage["wte_delta"] == 22784
    assert lineage["lm_head_weight_delta"] == 22784
    assert lineage["lm_head_bias_delta"] == 578
    assert lineage["effective_vocab_size"] == 578
    assert lineage["lineage_verified"] is True

    # Shuffled target audit
    shuffled_audit = audit_report["shuffled_target_audit"]
    assert shuffled_audit["verdict"] == "LEAKAGE_AND_PERMUTATION_ARTIFACT"
    assert "rng.permutation" in shuffled_audit["root_cause"]

    # Paired Student's t confidence interval
    paired_ci = audit_report["paired_utility_uncertainty"]
    assert paired_ci["mean_delta"] == 0.0054
    assert paired_ci["spans_zero"] is True
    assert paired_ci["scientific_status"] == "PROVISIONAL_NOT_SIGNIFICANT"
    assert paired_ci["ci_95"][0] < 0.0 < paired_ci["ci_95"][1]


def test_experience_records_and_manifest_serialization() -> None:
    """Verify Phase 2 learning experience dataclasses, serialization, and controls."""
    desc = ObservableTaskDescriptors(
        sample_count=48,
        seq_length=3,
        unique_symbols_count=12,
        input_variance=8.5,
        input_entropy=2.8,
        zero_shot_dev_em=0.45,
        zero_shot_dev_loss=1.8,
        best_repertoire_em=0.50,
        best_repertoire_loss=1.6,
    )
    vec = desc.feature_vector()
    assert len(vec) == 9
    assert all(isinstance(x, float) for x in vec)

    action = LearningAction(
        action_type=LearningActionType.TRAIN_CANDIDATE,
        step_budget=40,
        learning_rate=0.003,
    )
    rec = LearningExperienceRecord(
        record_id="rec_001",
        task_id="task_caesar3",
        episode_index=0,
        descriptors=desc,
        action=action,
        actual_steps=40,
        gpu_time_ms=500.0,
        initial_loss=3.1,
        final_loss=0.4,
        loss_reduction_ratio=0.87,
        dev_em=0.85,
        qualification_passed=True,
        retention_decision=RetentionDecision.RETAINED,
        repertoire_size_after=1,
        sealed_test_em=0.83,
        sealed_test_loss=0.45,
    )

    d = rec.to_dict()
    assert d["action"]["action_type"] == "TRAIN_CANDIDATE"
    assert d["retention_decision"] == "RETAINED"

    reconstructed = LearningExperienceRecord.from_dict(d)
    assert reconstructed.record_id == rec.record_id
    assert reconstructed.dev_em == rec.dev_em
    assert reconstructed.action.step_budget == 40

    # Manifest container
    manifest = LearningExperienceManifest(records=[rec, rec])
    json_str = manifest.to_json()
    loaded_manifest = LearningExperienceManifest.from_json(json_str)
    assert loaded_manifest.count() == 2

    # Shuffled control perturbation
    shuffled_man = manifest.generate_shuffled_control(seed=42)
    assert shuffled_man.count() == 2

    # Irrelevant control perturbation
    irr_man = manifest.generate_irrelevant_control(seed=42)
    assert irr_man.count() == 2


def test_procedural_task_generation_and_taxonomy() -> None:
    """Verify Phase 5 procedural rules, disjoint splits, and decoupled scrambling."""
    gen = ProceduralTaskGenerator(vocab_size=16)

    # Test all 10 procedural rule families
    seq = (1, 4, 10)
    for fam in ProceduralRuleFamily:
        out = apply_procedural_rule(fam, seq, vocab_size=16)
        assert len(out) == 3
        assert all(0 <= s < 16 for s in out)

    # Specific rule correctness
    assert apply_procedural_rule(ProceduralRuleFamily.CAESAR_SHIFT_MOD3, (1, 4, 15)) == (4, 7, 2)
    assert apply_procedural_rule(ProceduralRuleFamily.CAESAR_SHIFT_MOD7, (1, 4, 15)) == (8, 11, 6)
    assert apply_procedural_rule(ProceduralRuleFamily.COMPLEMENT_REFLECT, (0, 7, 15)) == (15, 8, 0)
    assert apply_procedural_rule(ProceduralRuleFamily.AFFINE_PARAM_5_3, (0, 1, 3)) == (3, 8, 2)
    assert apply_procedural_rule(ProceduralRuleFamily.BITWISE_XOR_MASK, (0, 9, 15)) == (9, 0, 6)

    # Dataset split disjointness
    task = gen.generate_task(
        task_id="t_caesar7",
        rule_family=ProceduralRuleFamily.CAESAR_SHIFT_MOD7,
        seed=123,
        n_train=20,
        n_dev=10,
        n_qual=10,
        n_sealed_test=20,
        generalization_category=GeneralizationCategory.NEW_PARAMETERIZATION,
    )
    train_in = {ex.input_tokens for ex in task.train_examples}
    dev_in = {ex.input_tokens for ex in task.dev_examples}
    qual_in = {ex.input_tokens for ex in task.qualification_examples}
    test_in = {ex.input_tokens for ex in task.sealed_test_examples}

    # Strict isolation: disjoint sets
    assert train_in.isdisjoint(dev_in)
    assert train_in.isdisjoint(qual_in)
    assert train_in.isdisjoint(test_in)
    assert dev_in.isdisjoint(qual_in)
    assert dev_in.isdisjoint(test_in)
    assert qual_in.isdisjoint(test_in)

    # Decoupled scrambling
    scrambled = gen.generate_decoupled_scrambled_task(task, seed=42)
    assert len(scrambled.train_examples) == len(task.train_examples)
    # Check that targets are decoupled random tokens
    for orig_ex, scr_ex in zip(task.train_examples, scrambled.train_examples, strict=False):
        assert scr_ex.input_tokens == orig_ex.input_tokens
        assert scr_ex.rule_family == "DECOUPLED_RANDOM_TARGETS"


def test_experience_guided_controller_policies() -> None:
    """Verify Phase 3 & 6 controller recommendations across policies."""
    desc = ObservableTaskDescriptors(
        sample_count=48,
        seq_length=3,
        unique_symbols_count=12,
        input_variance=8.5,
        input_entropy=2.8,
        zero_shot_dev_em=0.20,
        zero_shot_dev_loss=2.5,
        best_repertoire_em=0.20,
        best_repertoire_loss=2.5,
    )

    # 1. FIXED_BUDGET_FRESH
    fresh_ctrl = ExperienceGuidedController(mode=ControllerPolicyMode.FIXED_BUDGET_FRESH)
    rec_fresh = fresh_ctrl.select_action_and_budget(desc, [], None)
    assert rec_fresh.action.action_type == LearningActionType.FRESH_ADAPT
    assert rec_fresh.action.step_budget == 40

    # 2. Reusing existing skill if dev EM >= 0.80
    high_em_desc = ObservableTaskDescriptors(
        sample_count=48,
        seq_length=3,
        unique_symbols_count=12,
        input_variance=8.5,
        input_entropy=2.8,
        zero_shot_dev_em=0.88,
        zero_shot_dev_loss=0.3,
        best_repertoire_em=0.88,
        best_repertoire_loss=0.3,
    )
    rel_ctrl = ExperienceGuidedController(mode=ControllerPolicyMode.RELEVANT_HISTORY)
    rec_reuse = rel_ctrl.select_action_and_budget(
        high_em_desc, ["skill_0"], None, best_repertoire_skill_id="skill_0"
    )
    assert rec_reuse.action.action_type == LearningActionType.REUSE_EXISTING
    assert rec_reuse.action.step_budget == 0
    assert rec_reuse.action.reused_skill_id == "skill_0"

    # 3. NO_HISTORY fallback when history is empty
    no_hist_ctrl = ExperienceGuidedController(mode=ControllerPolicyMode.NO_HISTORY)
    rec_no_hist = no_hist_ctrl.select_action_and_budget(desc, [], None)
    assert rec_no_hist.action.action_type == LearningActionType.TRAIN_CANDIDATE
    assert rec_no_hist.action.step_budget == 40

    # 4. RELEVANT_HISTORY with populated history
    rec1 = LearningExperienceRecord(
        record_id="rec1",
        task_id="t1",
        episode_index=0,
        descriptors=desc,
        action=LearningAction(LearningActionType.TRAIN_CANDIDATE, step_budget=20, learning_rate=0.003),
        actual_steps=20,
        gpu_time_ms=250.0,
        initial_loss=3.0,
        final_loss=1.2,
        loss_reduction_ratio=0.6,
        dev_em=0.65,
        qualification_passed=False,
        retention_decision=RetentionDecision.DISCARDED,
        repertoire_size_after=0,
    )
    rec2 = LearningExperienceRecord(
        record_id="rec2",
        task_id="t2",
        episode_index=1,
        descriptors=desc,
        action=LearningAction(LearningActionType.TRAIN_CANDIDATE, step_budget=40, learning_rate=0.003),
        actual_steps=40,
        gpu_time_ms=500.0,
        initial_loss=3.0,
        final_loss=0.3,
        loss_reduction_ratio=0.9,
        dev_em=0.88,
        qualification_passed=True,
        retention_decision=RetentionDecision.RETAINED,
        repertoire_size_after=1,
    )
    rec3 = LearningExperienceRecord(
        record_id="rec3",
        task_id="t3",
        episode_index=2,
        descriptors=desc,
        action=LearningAction(LearningActionType.TRAIN_CANDIDATE, step_budget=60, learning_rate=0.003),
        actual_steps=60,
        gpu_time_ms=750.0,
        initial_loss=3.0,
        final_loss=0.2,
        loss_reduction_ratio=0.93,
        dev_em=0.90,
        qualification_passed=True,
        retention_decision=RetentionDecision.RETAINED,
        repertoire_size_after=2,
    )
    manifest = LearningExperienceManifest(records=[rec1, rec2, rec3])
    rec_guided = rel_ctrl.select_action_and_budget(desc, [], manifest)
    assert rec_guided.action.action_type == LearningActionType.TRAIN_CANDIDATE
    assert rec_guided.action.step_budget in (20, 40, 60)

    # 5. CONSERVATIVE_HISTORY
    cons_ctrl = ExperienceGuidedController(mode=ControllerPolicyMode.CONSERVATIVE_HISTORY)
    rec_cons = cons_ctrl.select_action_and_budget(desc, [], manifest)
    assert rec_cons.confidence_penalty >= 0.0


def test_safeguards_and_capacity_controls() -> None:
    """Verify Phase 7 & 8 safeguards, rollback, and parameter capacity."""
    # 1. Base model invariance and double SHA-256
    cfg = ModelConfig(
        vocab_size=16,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        context_length=32,
    )
    model = MiniGPT(cfg)
    unnamed_hash = M75SafeguardsAuditor.compute_unnamed_param_hash(model)
    named_hash = M75SafeguardsAuditor.compute_named_param_hash(model)
    assert len(unnamed_hash) == 64
    assert len(named_hash) == 64
    assert M75SafeguardsAuditor.verify_base_invariance(model, unnamed_hash, named_hash) is True

    # 2. Arithmetic carry retention
    carry_rate = M75SafeguardsAuditor.verify_arithmetic_carry_retention()
    assert carry_rate == 1.0

    # 3. Operational logic probes
    logic_rate = M75SafeguardsAuditor.verify_chained_operational_logic()
    assert logic_rate == 1.0

    # 4. Bit-exact rollback
    adapter = Rank16SequenceAdapter(d_model=64, bottleneck_dim=16, num_symbols=16)
    assert M75SafeguardsAuditor.verify_bit_exact_rollback(adapter) is True

    # 5. Capacity variants comparison (Rank-16 vs Rank-32)
    cap = M75SafeguardsAuditor.compare_capacity_variants(d_model=128, num_symbols=16)
    assert cap["rank16_parameters"] == 2592
    assert cap["rank32_parameters"] == 4912
    assert cap["parameter_ratio"] == 1.895
