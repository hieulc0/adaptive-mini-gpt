"""Unit tests for M22 Compact Scratchpads, RoPE, ALiBi, and Length Generalization."""

from __future__ import annotations

import pytest
import torch

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.rules.arithmetic_dataset import (
    ArithmeticDataset,
    ArithmeticProblem,
    format_problem,
    generate_controlled_addition_dataset,
    parse_arithmetic_completion,
)


def test_compact_scratchpad_formatting_and_parsing() -> None:
    """Verify compact scratchpad formatting and robust parsing."""
    p = ArithmeticProblem.create(127, 386)
    prompt, target = format_problem(p, "compact_scratchpad")

    assert "Add: 1 2 7 + 3 8 6" in prompt
    assert "Steps: " in prompt
    assert "7+6+0=13>3:1" in target
    assert "2+8+1=11>1:1" in target
    assert "1+3+1=5>5:0" in target
    assert "ANS: 5 1 3" in target

    # Parse perfect target completion
    parsed = parse_arithmetic_completion(target, "compact_scratchpad")
    assert parsed["is_valid"] is True
    assert parsed["predicted_answer"] == "513"
    assert parsed["predicted_carries"] == [1, 1, 0]

    # Overflow carry test: 99 + 1 = 100
    p_ov = ArithmeticProblem.create(99, 1)
    prompt_ov, target_ov = format_problem(p_ov, "compact_scratchpad")
    assert "9+1+0=10>0:1" in target_ov
    assert "0+0+1=1>1:0" in target_ov
    assert "ANS: 1 0 0" in target_ov

    parsed_ov = parse_arithmetic_completion(target_ov, "compact_scratchpad")
    assert parsed_ov["is_valid"] is True
    assert parsed_ov["predicted_answer"] == "100"
    assert parsed_ov["predicted_carries"] == [1, 1, 0]


def test_compact_scratchpad_token_efficiency() -> None:
    """Verify that compact scratchpad encodes multi-digit arithmetic in far fewer tokens."""
    tokenizer = BPETokenizer()
    tokenizer.train("Add: 0 1 2 3 4 5 6 7 8 9 + = > : Steps: ANS: col units tens", target_vocab_size=100)

    # 4-digit problem
    p4 = ArithmeticProblem.create(1234, 5678)
    _, target_verbose = format_problem(p4, "scratchpad")
    _, target_compact = format_problem(p4, "compact_scratchpad")

    toks_verbose = len(tokenizer.encode(target_verbose))
    toks_compact = len(tokenizer.encode(target_compact))

    # Compact must save at least 40% of token length compared to verbose
    assert toks_compact < toks_verbose
    assert toks_compact < 100


def test_positional_encodings_initialization_and_forward() -> None:
    """Test MiniGPT with absolute, RoPE, and ALiBi positional encodings."""
    # 1. Absolute
    cfg_abs = ModelConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        context_length=64,
        pos_encoding="absolute",
    )
    m_abs = MiniGPT(cfg_abs)
    assert m_abs.wpe is not None
    param_abs = sum(p.numel() for p in m_abs.parameters())

    # 2. RoPE
    cfg_rope = ModelConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        context_length=64,
        pos_encoding="rope",
    )
    m_rope = MiniGPT(cfg_rope)
    assert m_rope.wpe is None
    param_rope = sum(p.numel() for p in m_rope.parameters())
    # Parameter count difference is exactly the wpe embedding table (64 * 64 = 4096)
    assert param_rope == param_abs - 4096

    # 3. ALiBi
    cfg_alibi = ModelConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        context_length=64,
        pos_encoding="alibi",
    )
    m_alibi = MiniGPT(cfg_alibi)
    assert m_alibi.wpe is None
    param_alibi = sum(p.numel() for p in m_alibi.parameters())
    assert param_alibi == param_rope

    # Forward pass on all three
    x = torch.randint(0, 100, (2, 16))
    targets = torch.randint(0, 100, (2, 16))

    out_abs, loss_abs = m_abs(x, targets)
    out_rope, loss_rope = m_rope(x, targets)
    out_alibi, loss_alibi = m_alibi(x, targets)

    assert out_abs.shape == (2, 16, 100)
    assert out_rope.shape == (2, 16, 100)
    assert out_alibi.shape == (2, 16, 100)

    assert loss_abs is not None and not torch.isnan(loss_abs)
    assert loss_rope is not None and not torch.isnan(loss_rope)
    assert loss_alibi is not None and not torch.isnan(loss_alibi)


def test_rope_and_alibi_length_extrapolation_forward() -> None:
    """Verify that relative positional encodings forward beyond context_length without error."""
    cfg_rope = ModelConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        context_length=32,
        pos_encoding="rope",
    )
    m_rope = MiniGPT(cfg_rope)

    cfg_alibi = ModelConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        context_length=32,
        pos_encoding="alibi",
    )
    m_alibi = MiniGPT(cfg_alibi)

    # Sequence length 48 > context_length 32
    x_long = torch.randint(0, 100, (1, 48))

    # Absolute would raise ValueError
    cfg_abs = ModelConfig(
        vocab_size=100,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        context_length=32,
        pos_encoding="absolute",
    )
    m_abs = MiniGPT(cfg_abs)
    with pytest.raises(ValueError):
        m_abs(x_long)

    # RoPE and ALiBi succeed
    out_rope, _ = m_rope(x_long)
    out_alibi, _ = m_alibi(x_long)

    assert out_rope.shape == (1, 48, 100)
    assert out_alibi.shape == (1, 48, 100)


def test_language_retention_replay_mixing() -> None:
    """Verify that replay data is mixed into ArithmeticDataset with valid labels."""
    tokenizer = BPETokenizer()
    tokenizer.train("Paris is capital France 1 2 3 Add Steps ANS", target_vocab_size=100)

    probs = [ArithmeticProblem.create(12, 34)]
    replay_corpus = [
        "Paris is capital of France",
        "William Shakespeare wrote Hamlet",
    ]

    ds = ArithmeticDataset(
        problems=probs,
        tokenizer=tokenizer,
        arm="compact_scratchpad",
        replay_texts=replay_corpus,
        replay_ratio=2.0,  # 2 replay texts per arithmetic problem
        seed=42,
    )

    # 1 arithmetic problem + 2 replay texts = 3 items
    assert len(ds) == 3
    # Check that replay item has unmasked tokens
    inp, lab = ds[1]
    assert len(inp) == len(lab)
    assert lab[0].item() != -100
