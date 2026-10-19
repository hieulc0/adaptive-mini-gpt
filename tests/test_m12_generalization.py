"""Unit and integration tests for M12 Generalization and Continual-Learning Integrity (M12.1)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

from adaptive_mini_gpt.autonomous import (
    ControllerState,
    GrowthAction,
    LearningController,
)
from adaptive_mini_gpt.autonomous.tasks import (
    TASK_REGISTRY,
    evaluate_task_holdout,
    generate_instruction_qa_data,
    generate_instruction_symbolic_data,
    generate_reverse_data,
    get_string_split,
    score_instruction_qa,
    verify_addition,
    verify_instruction_qa,
    verify_instruction_reasoning,
    verify_instruction_symbolic,
    verify_multiplication,
    verify_parity,
    verify_reverse,
    verify_subtraction,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.evaluation.evaluator import evaluate_text_retention_suite
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT


def test_reversal_train_val_test_disjoint() -> None:
    """Verify that string reversal generator creates strictly disjoint train/val/test splits."""
    # Generate large sample pools for train, val, and test
    train_text, train_fp = generate_instruction_symbolic_data(
        num_samples=200, seed=42, split="train"
    )
    val_text, val_fp = generate_instruction_symbolic_data(num_samples=100, seed=43, split="val")
    test_text, test_fp = generate_instruction_symbolic_data(num_samples=100, seed=44, split="test")

    assert train_fp != test_fp
    assert train_fp != val_fp

    # Extract all generated target words
    train_words = set(re.findall(r"word '([a-zA-Z0-9]+)'", train_text))
    val_words = set(re.findall(r"word '([a-zA-Z0-9]+)'", val_text))
    test_words = set(re.findall(r"word '([a-zA-Z0-9]+)'", test_text))

    assert len(train_words) > 0
    assert len(val_words) > 0
    assert len(test_words) > 0

    # Strict independence at example level: 0 overlap between train and test
    overlap_train_test = train_words.intersection(test_words)
    assert len(overlap_train_test) == 0, f"Found train/test overlap: {overlap_train_test}"

    overlap_train_val = train_words.intersection(val_words)
    assert len(overlap_train_val) == 0, f"Found train/val overlap: {overlap_train_val}"

    overlap_val_test = val_words.intersection(test_words)
    assert len(overlap_val_test) == 0, f"Found val/test overlap: {overlap_val_test}"

    # Verify length distribution: test split contains lengths 7-8 which NEVER occur in train
    train_lengths = {len(w) for w in train_words}
    test_lengths = {len(w) for w in test_words}
    assert all(3 <= length <= 6 for length in train_lengths), (
        f"Train lengths out of bounds: {train_lengths}"
    )
    assert any(length in (7, 8) for length in test_lengths), (
        f"Test split must contain OOD lengths: {test_lengths}"
    )


def test_procedural_reverse_partitioning() -> None:
    """Verify procedural rev(s)=reversed generator creates disjoint splits."""
    train_text, _ = generate_reverse_data(num_samples=200, seed=42, split="train")
    test_text, _ = generate_reverse_data(num_samples=100, seed=44, split="test")

    train_strings = set(re.findall(r"rev\(([a-zA-Z0-9]+)\)", train_text))
    test_strings = set(re.findall(r"rev\(([a-zA-Z0-9]+)\)", test_text))

    overlap = train_strings.intersection(test_strings)
    assert len(overlap) == 0, f"Found procedural reverse train/test overlap: {overlap}"


def test_factual_qa_train_val_test_partitioning() -> None:
    """Verify factual QA cleanly partitions into train facts, paraphrases, and unseen facts."""
    train_text, _ = generate_instruction_qa_data(num_samples=20, seed=42, split="train")
    val_text, _ = generate_instruction_qa_data(num_samples=20, seed=43, split="val")
    test_text, _ = generate_instruction_qa_data(num_samples=20, seed=44, split="test")

    # Unseen facts (e.g. Germany, Canada, Mars) must not appear in train
    assert "Germany" not in train_text
    assert "Canada" not in train_text
    assert "Red Planet" not in train_text
    assert "Berlin" not in train_text

    # Unseen facts MUST appear in test
    assert any(term in test_text for term in ("Germany", "Canada", "Mars", "Berlin", "Ottawa"))

    # Paraphrase facts test the same facts with different phrasings
    assert "serves as France's capital" in val_text or "Japan's capital city" in val_text


def test_strict_symbolic_verifier() -> None:
    """Verify that symbolic verifier strictly parses answer and rejects leaks and contradictions."""
    prompt = "<|user|>\nReverse the sequence of characters in the word 'photon'.\n<|assistant|>\n"

    # 1. Accepts properly formatted correct answer
    assert verify_instruction_symbolic(prompt, "The reversed sequence is: notohp.<|eos|>")
    assert verify_instruction_symbolic(prompt, "The reversed sequence is: notohp")
    assert verify_instruction_symbolic(prompt, "notohp.")
    assert verify_instruction_symbolic(prompt, "notohp<|eos|>")
    assert verify_instruction_symbolic(prompt, "Result: notohp.<|eos|>")

    # 2. Rejects incorrect reversed word
    assert not verify_instruction_symbolic(prompt, "The reversed sequence is: photon.<|eos|>")
    assert not verify_instruction_symbolic(prompt, "The reversed sequence is: nootph.<|eos|>")

    # 3. P2 Fix: Rejects contradictory text following an extracted answer
    assert not verify_instruction_symbolic(
        prompt, "The reversed sequence is: notohp. Actually, the answer is photon."
    )
    assert not verify_instruction_symbolic(
        prompt, "The sequence notohp is invalid because the reverse is not notohp."
    )

    # 4. Rejects empty completion
    assert not verify_instruction_symbolic(prompt, "")
    assert not verify_instruction_symbolic(prompt, "   ")

    # 5. Rejects malformed prompt
    assert not verify_instruction_symbolic("Malformed prompt without target", "notohp")


def test_strict_procedural_verifiers_prevent_permissive_matches() -> None:
    """P2 Fix: Verify older procedural verifiers do NOT allow prefix-based startswith matches."""
    # Addition: 5+7=12. "123" must be rejected!
    assert verify_addition("5+7=", "12")
    assert verify_addition("5+7=", "12\n")
    assert not verify_addition("5+7=", "123")
    assert not verify_addition("5+7=", "125")

    # Subtraction: 15-3=12. "123" must be rejected!
    assert verify_subtraction("15-3=", "12")
    assert not verify_subtraction("15-3=", "123")

    # Multiplication: 3*4=12. "120" must be rejected!
    assert verify_multiplication("3*4=", "12")
    assert not verify_multiplication("3*4=", "120")

    # Parity: parity 8=even. "evenly" must be rejected!
    assert verify_parity("parity 8=", "even")
    assert not verify_parity("parity 8=", "evenly")
    assert verify_parity("parity 7=", "odd")
    assert not verify_parity("parity 7=", "oddly")

    # Reverse: rev(abc)=cba. "cbad" must be rejected!
    assert verify_reverse("rev(abc)=", "cba")
    assert not verify_reverse("rev(abc)=", "cbad")

    # Instruction reasoning: unsupported prompt must be rejected
    assert not verify_instruction_reasoning("What is your name?", "I am Mini-GPT.")
    assert not verify_instruction_reasoning("Describe photosynthesis.", "Plants use light.")
    assert verify_instruction_reasoning("Calculate 12 + 15.", "The result is 27.")
    assert not verify_instruction_reasoning("Calculate 12 + 15.", "The result is 99.")


def test_strict_factual_qa_verifier() -> None:
    """Verify that factual QA verifier strictly fails unknown questions."""
    # 1. Known question with correct answer
    prompt_france = "<|user|>\nWhat is the capital of France?\n<|assistant|>\n"
    assert verify_instruction_qa(prompt_france, "The capital of France is Paris.<|eos|>")
    assert verify_instruction_qa(prompt_france, "Paris.")

    # 2. Known question with incorrect answer
    assert not verify_instruction_qa(prompt_france, "The capital of France is London.<|eos|>")
    assert not verify_instruction_qa(prompt_france, "Berlin.")

    # 3. CRITICAL M11 FIX: Unknown question must return False (NEVER True on non-empty)
    prompt_unknown = "<|user|>\nWhat is the capital of Atlantis?\n<|assistant|>\n"
    assert not verify_instruction_qa(prompt_unknown, "Atlantis city is very beautiful.")
    assert not verify_instruction_qa(prompt_unknown, "Unknown response.")

    # 4. Verify score_instruction_qa status codes
    corr, status = score_instruction_qa(prompt_france, "Paris.")
    assert corr is True
    assert status == "correct"

    incorr, status_inc = score_instruction_qa(prompt_france, "Madrid.")
    assert incorr is False
    assert status_inc == "incorrect"

    unsc, status_unsc = score_instruction_qa(prompt_unknown, "Some non-empty answer.")
    assert unsc is False
    assert status_unsc == "unscorable"


def test_hash_partition_deterministic() -> None:
    """Verify that string split assignment is deterministic and covers all buckets."""
    assert get_string_split("quantum") == get_string_split("quantum")
    assert get_string_split("tensor") == get_string_split("tensor")

    splits = [get_string_split(f"word_{i}") for i in range(500)]
    train_count = splits.count("train")
    val_count = splits.count("val")
    test_count = splits.count("test")

    assert train_count > 300  # ~70%
    assert val_count > 40  # ~15%
    assert test_count > 40  # ~15%


@pytest.fixture
def bpe_base_checkpoint(tmp_path: Path) -> tuple[Path, BPETokenizer, ModelConfig]:
    """Fixture providing a minimal BPE checkpoint for controller tests."""
    ckpt_dir = tmp_path / "bpe_base"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    corpus = (
        "The quick brown fox jumps over the lazy dog.\n"
        "William Shakespeare wrote Hamlet, Macbeth, and King Lear.\n"
        "Paris is the capital of France, and Tokyo is the capital of Japan.\n"
        "<|user|>\nWhat is the capital of France?\n<|assistant|>\nParis.<|eos|>\n"
        "<|user|>\nReverse the sequence of characters in the word 'tensor'.\n"
        "<|assistant|>\nrosnet.<|eos|>\n"
    ) * 10

    tokenizer = BPETokenizer()
    tokenizer.train(corpus, target_vocab_size=280)

    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        context_length=64,
        dropout=0.0,
    )
    model = MiniGPT(cfg)

    ckpt_data = {
        "model_state_dict": model.state_dict(),
        "model_config": cfg.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        "tokenizer_fingerprint": tokenizer.get_fingerprint(),
        "step": 100,
        "tokens_seen": 2000,
        "best_val_loss": 3.2,
        "run_id": "test_m12_bpe_base",
    }
    torch.save(ckpt_data, ckpt_dir / "latest.pt")

    manifest_data = {
        "run_id": "test_m12_bpe_base",
        "step": 100,
        "tokens_seen": 2000,
        "best_val_loss": 3.2,
        "latest_checkpoint": "latest.pt",
        "history": [],
    }
    with open(ckpt_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    return ckpt_dir, tokenizer, cfg


def test_autonomous_task_selection_no_force(
    bpe_base_checkpoint: tuple[Path, BPETokenizer, ModelConfig], tmp_path: Path
) -> None:
    """Verify autonomous selection picks weakest skill without force_task and logs rationale."""
    base_dir, _, _ = bpe_base_checkpoint
    out_dir = tmp_path / "auto_selection_test"

    controller = LearningController(
        base_checkpoint_dir=base_dir,
        output_dir=out_dir,
        force_task=None,
        device="cpu",
    )

    controller.journal.baseline_metrics = {
        "tasks": {tid: {"accuracy": 0.50} for tid in TASK_REGISTRY}
    }
    controller.journal.baseline_metrics["tasks"]["instruction_symbolic"] = {"accuracy": 0.00}

    selected = controller.step_select_task()
    assert selected == "instruction_symbolic"
    assert controller.journal.selected_task == "instruction_symbolic"
    assert controller.journal.selection_reason is not None
    assert "instruction_symbolic" in controller.journal.selection_reason
    assert "weakest unmastered skill" in controller.journal.selection_reason


def test_evaluation_split_isolation_uses_val_for_selection_and_gating(
    bpe_base_checkpoint: tuple[Path, BPETokenizer, ModelConfig], tmp_path: Path
) -> None:
    """P1 Fix: Verify that step_evaluate and step_verify strictly use 'val' split."""
    base_dir, tokenizer, cfg = bpe_base_checkpoint
    out_dir = tmp_path / "split_isolation_test"

    controller = LearningController(
        base_checkpoint_dir=base_dir,
        output_dir=out_dir,
        device="cpu",
        precision="fp32",
        seed=42,
    )

    model = MiniGPT(cfg)
    base_summary = controller.step_evaluate(model, tokenizer)
    assert base_summary["split_used"] == "val"
    assert controller.journal.baseline_metrics["split_used"] == "val"

    # Verify task evaluations in baseline used split="val"
    for tid in ("reverse", "instruction_symbolic", "instruction_qa"):
        assert base_summary["tasks"][tid]["split"] == "val"


def test_holdout_reports_unique_questions() -> None:
    """Verify evaluate_task_holdout reports unique question counts and accuracies."""
    cfg = ModelConfig(vocab_size=280, d_model=32, n_layers=1, n_heads=1, d_ff=64, context_length=64)
    model = MiniGPT(cfg)
    tokenizer = BPETokenizer()
    tokenizer.train("What is the capital of France? Paris. Tokyo. Rome.", target_vocab_size=280)

    res = evaluate_task_holdout(
        model=model,
        tokenizer=tokenizer,
        task=TASK_REGISTRY["instruction_qa"],
        device=torch.device("cpu"),
        num_samples=20,
        split="val",
    )

    assert "unique_questions" in res
    assert "unique_correct" in res
    assert "unique_accuracy" in res
    assert res["unique_questions"] > 0
    assert res["unique_questions"] <= res["total"]


def test_retention_audit_raw_cross_checkpoint_drift(
    bpe_base_checkpoint: tuple[Path, BPETokenizer, ModelConfig], tmp_path: Path
) -> None:
    """Verify retention audit compares real cross-checkpoint loss and detects regressions."""
    _, tokenizer, cfg = bpe_base_checkpoint
    model = MiniGPT(cfg)

    # 1. Baseline retention
    base_retention = evaluate_text_retention_suite(model, tokenizer, device="cpu")
    base_losses = {
        dom: base_retention[dom]["raw_loss"]
        for dom in ("clean_prose", "numerical_prose", "mixed_prose_math")
    }

    # 2. Candidate retention with identical model -> deltas should be approximately 0.0
    cand_retention = evaluate_text_retention_suite(
        model, tokenizer, device="cpu", baseline_losses=base_losses
    )
    for dom in ("clean_prose", "numerical_prose", "mixed_prose_math"):
        assert abs(cand_retention[dom]["delta_loss"]) < 1e-4

    # 3. Simulate degraded candidate model by perturbing weights
    degraded_model = MiniGPT(cfg)
    with torch.no_grad():
        for param in degraded_model.parameters():
            param.add_(torch.randn_like(param) * 0.5)

    degraded_retention = evaluate_text_retention_suite(
        degraded_model, tokenizer, device="cpu", baseline_losses=base_losses
    )
    assert degraded_retention["clean_prose"]["delta_loss"] > 0.0
    assert degraded_retention["clean_prose"]["loss_auto"] > base_losses["clean_prose"]


def test_end_to_end_autonomous_learning_cycle_and_post_qualification(
    bpe_base_checkpoint: tuple[Path, BPETokenizer, ModelConfig], tmp_path: Path
) -> None:
    """Run end-to-end cycle with M12.1 val split gating and post-promotion test qualification."""
    base_dir, _, _ = bpe_base_checkpoint
    out_dir = tmp_path / "m12_e2e_cycle"

    controller = LearningController(
        base_checkpoint_dir=base_dir,
        output_dir=out_dir,
        budget_steps=2,
        min_improvement=-1.0,  # Promote to inspect records
        force_task=None,  # Autonomous selection!
        growth_action=GrowthAction.CONTINUE,
        device="cpu",
        precision="fp32",
        seed=42,
    )

    cycle_res = controller.run_cycle()
    assert cycle_res["decision"] == ControllerState.PROMOTE.value

    # Verify candidate_metrics contains M12.1 raw_retention_audit and split_used == "val"
    cand_metrics = cycle_res["candidate_metrics"]
    assert cand_metrics["split_used"] == "val"
    assert "raw_retention_audit" in cand_metrics

    # Test post-selection qualification on untouched test split
    qual_report = controller.step_qualify(num_samples=10)
    assert "tasks_test_split" in qual_report
    assert "reversal_breakdown_test_split" in qual_report
    breakdown = qual_report["reversal_breakdown_test_split"]
    assert "unseen_in_distribution_len_3_6" in breakdown
    assert "ood_extrapolation_len_7_8" in breakdown
    assert "alphanumeric_combinations" in breakdown
    assert breakdown["unseen_in_distribution_len_3_6"]["train_overlap_count"] == 0
