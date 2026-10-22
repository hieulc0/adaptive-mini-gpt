"""Unit tests for M43: Selective Long-Term Memory, Experience Reuse, and Continual Adaptation."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.memory.audit import M42ScientificAuditor
from adaptive_mini_gpt.memory.episodic_memory import EpisodicMemoryBank, LearningExperience
from adaptive_mini_gpt.memory.learning_experiments import (
    ControlledLearningExperimentRunner,
    build_m43_task_family,
    get_m43_benchmark_families,
)
from adaptive_mini_gpt.memory.lifecycle_consolidation import MemoryLifecycleBenchmark
from adaptive_mini_gpt.memory.retrieval import MemoryRetriever, RetrievalScoringWeights
from adaptive_mini_gpt.memory.sequential_retention import SequentialRetentionEvaluator
from adaptive_mini_gpt.memory.tri_state_policy import (
    TriStateAndMemoryDecisionEngine,
)
from adaptive_mini_gpt.meta.benchmark_suite import MetaTask
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT


@pytest.fixture
def dummy_setup() -> tuple[MiniGPT, BPETokenizer, torch.device]:
    device = torch.device("cpu")
    tok = BPETokenizer()
    corpus = (
        "YES NO UNKNOWN [MEMORY] [PREMISES] [QUESTION] [ANSWER] EnzymeAlpha SubstrateBeta ProductGamma "
        "MetaboliteDelta Qubit0 Qubit1 Qubit2 Qubit3 RootCert IntermediateCA SignerNode LeafDevice "
        "MoonLuna PlanetTerra StarSol CoreGalaxy ProtoRootX ArchaicFormY DialectZ ModernLexemeW "
        "HydrogenIsotope HeliumCore CarbonPlasma IronPeak catalyzes inhibited_by entangled_with "
        "endorses endorsed_by orbits_around orbited_by derives_from gives_rise_to fuses_into "
        "Does Is it verified that GhostElement NovelVoidX UnseenParticle Ex"
    )
    tok.train(corpus, target_vocab_size=340)

    config = ModelConfig(
        vocab_size=tok.vocab_size,
        context_length=64,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        dropout=0.0,
    )
    model = MiniGPT(config).to(device)
    return model, tok, device


def test_m42_scientific_audit_and_qualification_correction() -> None:
    report = M42ScientificAuditor.audit_m42_results()
    assert report.audit_verdict == "AUDIT_VERIFIED_AND_LANGUAGE_CORRECTED"
    assert report.readiness_for_m43_episodic_memory is True

    # Check audited metrics
    metrics = report.metrics
    assert "reptile_auc_vs_fresh" in metrics
    assert metrics["reptile_auc_vs_fresh"].qualification_status == "INCONCLUSIVE"
    assert metrics["final_32_accuracy"].qualification_status == "INCONCLUSIVE"
    assert metrics["neural_unknown_recall"].qualification_status == "PARTIAL_IMPROVEMENT_UNRELIABLE"
    assert metrics["neural_action_accuracy"].qualification_status == "FAIL_NO_GO"
    assert metrics["modular_retention_and_bwt"].qualification_status == "QUALIFIED_FOR_ISOLATION_ONLY"

    # Check statistical significance
    stat = report.statistical_significance_assessment
    assert stat["statistically_significant"] is False
    assert stat["relative_auc_gain"] == "+1.99%"


def test_episodic_memory_bank_lifecycle_and_provenance() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage_file = Path(tmp_dir) / "episodic_memory.json"
        bank = EpisodicMemoryBank(storage_path=storage_file, max_active_capacity=5)

        # 1. Provenance security rejection on unapproved source
        with pytest.raises(ValueError, match="Security violation"):
            bank.add_experience(
                task_id="unapproved_task",
                family="unapproved_family",
                premises=["A relates B"],
                question="Does A relate B?",
                model_prediction="YES",
                verified_answer="YES",
                error_category="none",
                corrected_procedure="none",
                skill_family="none",
                evidence_source="unapproved_internet_crawler",
            )

        # 2. Add verified experiences
        exp1 = bank.add_experience(
            task_id="t1",
            family="metabolic_cascade",
            premises=["EnzymeAlpha catalyzes SubstrateBeta", "SubstrateBeta catalyzes ProductGamma"],
            question="Does EnzymeAlpha catalyzes ProductGamma?",
            model_prediction="YES",
            verified_answer="YES",
            error_category="correct",
            corrected_procedure="Transitive catalysis",
            skill_family="metabolic_cascade",
            evidence_source="verified_oracle:graph_oracle_v1",
            episode_order=1,
        )
        assert exp1.experience_id in bank.active_experiences
        assert exp1.usefulness_score > 0.0

        # Add duplicate pattern to test consolidation
        exp2 = bank.add_experience(
            task_id="t2_duplicate",
            family="metabolic_cascade",
            premises=["EnzymeAlpha catalyzes SubstrateBeta", "SubstrateBeta catalyzes ProductGamma"],
            question="Does EnzymeAlpha catalyzes ProductGamma?",
            model_prediction="YES",
            verified_answer="YES",
            error_category="correct",
            corrected_procedure="Transitive catalysis",
            skill_family="metabolic_cascade",
            evidence_source="verified_oracle:graph_oracle_v1",
            episode_order=1,
        )

        # 3. Test redundant consolidation
        consolidated_count = bank.consolidate_redundant_experiences()
        assert consolidated_count >= 1
        assert exp1.access_count >= 2

        # 4. Test archival and restoration
        assert bank.archive_experience(exp1.experience_id) is True
        assert exp1.experience_id in bank.archived_experiences
        assert exp1.experience_id not in bank.active_experiences

        assert bank.restore_experience(exp1.experience_id) is True
        assert exp1.experience_id in bank.active_experiences

        # 5. Test superseding / revising contradicted experience
        revised = bank.revise_experience(
            old_experience_id=exp1.experience_id,
            new_verified_answer="NO",
            new_procedure="Updated pathway: negative regulation",
            new_evidence_source="verified_oracle:graph_oracle_v2",
            episode_order=2,
        )
        assert revised.verified_answer == "NO"
        assert bank.inspect_experience(exp1.experience_id).is_superseded is True
        assert exp1.experience_id in revised.consolidated_from

        # 6. Test persistence roundtrip
        bank2 = EpisodicMemoryBank(storage_path=storage_file)
        assert len(bank2.active_experiences) == len(bank.active_experiences)
        assert len(bank2.archived_experiences) == len(bank.archived_experiences)


def test_memory_retrieval_methods_and_noise_rejection() -> None:
    bank = EpisodicMemoryBank()

    # Seed memory bank with relevant and noisy experiences
    exp_relevant = bank.add_experience(
        task_id="rel_1",
        family="metabolic_cascade",
        premises=["EnzymeAlpha catalyzes SubstrateBeta"],
        question="Does EnzymeAlpha catalyzes SubstrateBeta?",
        model_prediction="YES",
        verified_answer="YES",
        error_category="correct",
        corrected_procedure="Direct catalysis",
        skill_family="metabolic_cascade",
        evidence_source="verified_oracle:graph_oracle_v1",
        episode_order=1,
    )
    exp_noise = bank.add_experience(
        task_id="noise_1",
        family="cryptographic_trust",
        premises=["RootCert endorses IntermediateCA"],
        question="Does RootCert endorses IntermediateCA?",
        model_prediction="YES",
        verified_answer="YES",
        error_category="correct",
        corrected_procedure="Trust endorsement",
        skill_family="cryptographic_trust",
        evidence_source="verified_oracle:graph_oracle_v1",
        episode_order=1,
    )

    retriever = MemoryRetriever(bank, max_exemplars=2)

    query = "Does EnzymeAlpha catalyzes SubstrateBeta?"
    premises = ["EnzymeAlpha catalyzes SubstrateBeta"]

    # Method A: None
    assert len(retriever.retrieve_none(query, premises)) == 0

    # Method B: Keyword
    kw = retriever.retrieve_keyword(query, premises)
    assert len(kw) >= 1
    assert kw[0].experience_id == exp_relevant.experience_id

    # Method C: Similarity
    sim = retriever.retrieve_similarity(query, premises)
    assert len(sim) >= 1
    assert sim[0].experience_id == exp_relevant.experience_id

    # Method D: Selective Multi-Factor
    sel = retriever.retrieve_selective(query, premises, current_episode=1)
    assert len(sel) >= 1
    assert sel[0].experience_id == exp_relevant.experience_id
    # Ensure noise was rejected
    assert all(e.experience_id != exp_noise.experience_id for e in sel)

    # Test prompt formatting
    prompt = retriever.format_in_context_prompt(query, premises, sel)
    assert "[MEMORY]" in prompt
    assert "[PREMISES]" in prompt
    assert "[QUESTION]" in prompt
    assert "[ANSWER]" in prompt


def test_controlled_learning_approaches_sample_efficiency(
    dummy_setup: tuple[MiniGPT, BPETokenizer, torch.device],
) -> None:
    model, tok, device = dummy_setup
    bank = EpisodicMemoryBank()

    families = get_m43_benchmark_families()
    target_family = families["metabolic_cascade"]

    runner = ControlledLearningExperimentRunner(model, tok, device, bank)

    # Run Approach 1 (Fresh, small budget [0, 2])
    rep1 = runner.run_approach_1_fresh(target_family, budgets=[0, 2])
    assert rep1.approach_name == "APPROACH_1_FRESH_NO_MEMORY"
    assert rep1.evaluation_mode == "neural_only"
    assert len(rep1.curve) == 2
    assert 0.0 <= rep1.area_under_curve <= 1.0

    # Populate memory bank for Approach 2, 3, 4
    for t in target_family.train_practice_pool[:4]:
        bank.add_experience(
            task_id=t.task_id,
            family=t.family,
            premises=t.premises,
            question=t.question,
            model_prediction=t.expected_verdict,
            verified_answer=t.expected_verdict,
            error_category=t.reasoning_type,
            corrected_procedure="Verified deduction",
            skill_family=t.family,
            evidence_source="verified_oracle:graph_oracle_v1",
            episode_order=1,
        )

    # Run Approach 2 (Memory Inference Frozen)
    rep2 = runner.run_approach_2_memory_inference(target_family, budgets=[0, 2])
    assert "APPROACH_2_MEMORY_INFERENCE_FROZEN" in rep2.approach_name
    assert rep2.evaluation_mode == "memory_assisted"

    # Run Approach 3 (Memory Trained -> Neural-Only Evaluation)
    rep3 = runner.run_approach_3_memory_assisted_training(target_family, budgets=[0, 2])
    assert rep3.approach_name == "APPROACH_3_MEMORY_TRAINED_NEURAL_ONLY"
    assert rep3.evaluation_mode == "neural_only"

    # Run Approach 4 (Selective Combined)
    rep4_neural, rep4_assisted = runner.run_approach_4_selective_memory_combined(target_family, budgets=[0, 2])
    assert rep4_neural.evaluation_mode == "neural_only"
    assert rep4_assisted.evaluation_mode == "memory_assisted"


def test_tri_state_head_and_memory_decision_policy(
    dummy_setup: tuple[MiniGPT, BPETokenizer, torch.device],
) -> None:
    model, tok, device = dummy_setup
    engine = TriStateAndMemoryDecisionEngine(model, tok, device)

    families = get_m43_benchmark_families()
    dev_family = families["orbital_mechanics"]

    # 1. Test Tri-State classification head
    diag = engine.evaluate_tri_state_head(
        train_tasks=dev_family.train_practice_pool[:4],
        eval_tasks=dev_family.held_out_test_pool[:4],
        epochs=3,
    )
    assert diag.total_eval_samples == 4
    assert 0.0 <= diag.overall_accuracy <= 1.0
    assert 0.0 <= diag.unknown_recall <= 1.0

    # 2. Test Deterministic Controller vs Neural Head
    test_task = dev_family.held_out_test_pool[0]
    det_dec = engine.decide_deterministic(test_task)
    neu_dec = engine.decide_neural(test_task)

    assert det_dec.source == "DETERMINISTIC_CONTROLLER"
    assert neu_dec.source == "NEURAL_POLICY"
    assert det_dec.action in ["DIRECT_ANSWER", "RETRIEVE_MEMORY", "ABSTAIN", "STORE_VERIFIED", "TRAIN_ADAPTER"]

    # 3. Test Policy Comparison
    rep = engine.evaluate_policy_benchmark(dev_family.held_out_test_pool[:6])
    assert rep.total_tasks == 6
    assert rep.deterministic_accuracy == 1.0


def test_sequential_retention_and_bwt(
    dummy_setup: tuple[MiniGPT, BPETokenizer, torch.device],
) -> None:
    model, tok, device = dummy_setup
    bank = EpisodicMemoryBank()
    evaluator = SequentialRetentionEvaluator(model, tok, device, bank)

    families = get_m43_benchmark_families()
    episodes = [
        families["metabolic_cascade"],
        families["quantum_entanglement"],
    ]

    report = evaluator.run_sequential_retention_benchmark(episodes, practice_budget=2)
    assert len(report.evaluated_families) == 2
    assert report.canonical_carry_intact is True
    assert report.canonical_logic_intact is True
    assert report.base_parameter_hash_invariant is True
    assert len(report.neural_only_matrix) == 2
    assert len(report.memory_assisted_matrix) == 2
