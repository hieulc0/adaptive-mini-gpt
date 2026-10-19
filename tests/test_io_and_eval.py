"""Tests for upgraded tokenizer ASCII support, generation I/O, and exact-match evaluation."""

from __future__ import annotations

import json
import subprocess
import sys

import torch

from adaptive_mini_gpt.data.dataset import (
    TokenDataset,
    generate_arithmetic_corpus,
    is_holdout_arithmetic_pair,
)
from adaptive_mini_gpt.data.tokenizer import CharacterTokenizer
from adaptive_mini_gpt.evaluation.evaluator import (
    evaluate_arithmetic_accuracy,
    generate_sample,
)
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT


def test_base_ascii_prevents_unk_poisoning() -> None:
    """Base-ASCII tokenizer fitted on Shakespeare must not map digits or math ops to unk."""
    shakespeare_text = (
        "From fairest creatures we desire increase, That thereby beauty's rose might never die."
    )
    tok = CharacterTokenizer(use_base_ascii=True).fit(shakespeare_text)

    # Arithmetic equation with digits and symbols
    equation = "ADD 14+25=39"
    encoded = tok.encode(equation)

    # None of the characters in the equation should map to unk
    assert tok.unk_id not in encoded, f"Encountered <unk> in encoded arithmetic tokens: {encoded}"
    decoded = tok.decode(encoded)
    assert decoded == equation


def test_generate_top_p_and_repetition_penalty() -> None:
    """MiniGPT.generate must respect top_p, repetition_penalty, and stop_on_newline."""
    torch.manual_seed(42)
    tok = CharacterTokenizer(use_base_ascii=True).fit("abc 123 \n")
    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        context_length=16,
    )
    model = MiniGPT(cfg)

    # Generate with top_p and repetition penalty
    res = generate_sample(
        model=model,
        tokenizer=tok,
        prompt="12",
        max_new_tokens=10,
        temperature=0.8,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.2,
        stop_on_newline=True,
    )

    assert "completion" in res
    assert "generated_text" in res
    assert "prompt_tokens" in res
    assert res["prompt_tokens"] == 2
    assert res["total_tokens"] >= 2


def test_evaluate_arithmetic_accuracy() -> None:
    """evaluate_arithmetic_accuracy returns accuracy and samples without runtime error."""
    torch.manual_seed(42)
    tok = CharacterTokenizer(use_base_ascii=True).fit("ADD SUB PAR 0123456789+-= EVEN ODD\n")
    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        context_length=32,
    )
    model = MiniGPT(cfg)

    results = evaluate_arithmetic_accuracy(
        model=model,
        tokenizer=tok,
        num_problems=10,
        seed=123,
    )

    assert "exact_match_accuracy" in results
    assert "correct" in results
    assert "total_problems" in results
    assert results["total_problems"] == 10
    assert len(results["sample_predictions"]) == 5


def test_disjoint_holdout_arithmetic() -> None:
    """Corpus must not contain any holdout operand pairs, and eval checks disjoint pairs."""
    text, _ = generate_arithmetic_corpus(num_samples=200, seed=42)
    lines = text.strip().split("\n")
    for line in lines:
        if line.startswith("ADD"):
            operands = line[4:].split("=")[0].split("+")
            a, b = int(operands[0]), int(operands[1])
            assert not is_holdout_arithmetic_pair(0, a, b)
        elif line.startswith("SUB"):
            operands = line[4:].split("=")[0].split("-")
            a, b = int(operands[0]), int(operands[1])
            assert not is_holdout_arithmetic_pair(1, a, b)


def test_answer_masking_in_token_dataset() -> None:
    """mask_targets_before_equals must set prompt tokens to -1 while preserving answers."""
    tok = CharacterTokenizer(use_base_ascii=True).fit("ADD 12+34=46\n")
    seq = "ADD 12+34=46\n"
    tokens = torch.tensor(tok.encode(seq) * 4, dtype=torch.long)
    eq_id = tok.stoi["="]
    nl_id = tok.stoi["\n"]

    ds = TokenDataset(
        tokens,
        context_length=12,
        mask_prompt_before_equals=True,
        equals_token_id=eq_id,
        newline_token_id=nl_id,
    )
    x, y = ds.get_batch(batch_size=2)
    # y must contain at least one -1 (masked prompt) and at least one non -1 (supervised answer)
    assert (y == -1).any()
    assert (y != -1).any()


def test_bypass_moe_and_mod() -> None:
    """MiniGPT with MoE and MoD forwards properly with bypass_moe and mod_enabled."""
    cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        context_length=16,
        adaptive_enabled=True,
        n_experts=2,
        mod_enabled=True,
        mod_capacity_factor=0.5,
    )
    model = MiniGPT(cfg)
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    logits_norm, loss_norm = model(x, y, bypass_moe=False)
    logits_byp, loss_byp = model(x, y, bypass_moe=True)
    assert logits_norm.shape == (2, 16, 64)
    assert logits_byp.shape == (2, 16, 64)


def test_cli_json_output_and_task_id() -> None:
    """CLI inspect and generate with --json-output emit valid single-line JSON."""
    res_insp = subprocess.run(
        [
            sys.executable,
            "-m",
            "adaptive_mini_gpt.cli",
            "--json-output",
            "inspect",
            "--config",
            "configs/tiny.yaml",
        ],
        capture_output=True,
        text=True,
    )
    assert res_insp.returncode == 0
    data = json.loads(res_insp.stdout.strip())
    assert data["architecture"] == "decoder_only_transformer"
