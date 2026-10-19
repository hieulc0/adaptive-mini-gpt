"""Tests for M7: Rigorous Generalization, Retention & Architecture Verification."""

from __future__ import annotations

import torch

from adaptive_mini_gpt.adaptive.growth import grow_model_experts
from adaptive_mini_gpt.data.tokenizer import CharacterTokenizer
from adaptive_mini_gpt.evaluation.evaluator import (
    evaluate_arithmetic_detailed,
    evaluate_text_retention_suite,
    has_addition_carry,
    has_subtraction_borrow,
)
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT


def test_carry_and_borrow_math_detection() -> None:
    """Carry and borrow detectors must correctly flag decimal column dependencies."""
    # Addition carry
    assert not has_addition_carry(14, 23)  # 4+3=7, 1+2=3 -> no carry
    assert has_addition_carry(18, 25)  # 8+5=13 -> carry
    assert has_addition_carry(45, 57)  # 5+7=12 -> carry
    assert not has_addition_carry(0, 5)  # single digit

    # Subtraction borrow
    assert not has_subtraction_borrow(58, 23)  # 8>=3, 5>=2 -> no borrow
    assert has_subtraction_borrow(52, 27)  # 2<7 -> borrow
    assert has_subtraction_borrow(100, 1)  # borrow cascades
    assert not has_subtraction_borrow(40, 10)  # no borrow


def test_domain_trigger_token_ids_protects_numerical_prose() -> None:
    """When domain_trigger_token_ids=['='], prose with numbers must NOT activate MoE."""
    torch.manual_seed(42)
    tok = CharacterTokenizer(use_base_ascii=True).fit(
        "In the year 1594, Act 2, Scene 4. ADD 12+34=46\n"
    )
    eq_id = tok.stoi["="]
    digit_ids = [tok.stoi[d] for d in "0123456789"]

    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        context_length=32,
        dropout=0.0,
        tie_word_embeddings=False,
    )
    model = MiniGPT(cfg)
    model.eval()

    # Grow experts with selective token adaptation unlocked on digits + '='
    grow_model_experts(
        model,
        add_experts=2,
        freeze_backbone=True,
        trainable_token_ids=digit_ids + [eq_id],
    )
    # Set domain trigger explicitly to '=' only
    model.config.domain_trigger_token_ids = [eq_id]

    # 1. Numerical prose without '='
    num_prose = "In the year 1594, Act 2."
    x_num = torch.tensor([tok.encode(num_prose)], dtype=torch.long)

    with torch.no_grad():
        logits_auto, _ = model(x_num, bypass_moe=False)
        logits_bypassed, _ = model(x_num, bypass_moe=True)

    # Numerical prose MUST NOT trigger MoE -> bitwise identical logits!
    assert torch.equal(logits_auto, logits_bypassed), (
        "Specialist MoE was erroneously triggered on numerical prose without '='!"
    )

    # 2. Equation with '='
    eq_text = "ADD 12+34=46"
    x_eq = torch.tensor([tok.encode(eq_text)], dtype=torch.long)

    # Train moe_head so it produces non-zero output
    model.train()
    opt = model.configure_optimizers(learning_rate=1e-2, weight_decay=0.01)
    y_eq = torch.tensor([tok.encode("DD 12+34=46\n")], dtype=torch.long)
    opt.zero_grad()
    _, loss = model(x_eq, y_eq)
    assert loss is not None
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        logits_eq_auto, _ = model(x_eq, bypass_moe=False)
        logits_eq_bypassed, _ = model(x_eq, bypass_moe=True)

    # Equation with '=' MUST activate specialist path -> logits at and after '=' must differ!
    eq_pos = eq_text.index("=")
    assert not torch.equal(logits_eq_auto[:, eq_pos:, :], logits_eq_bypassed[:, eq_pos:, :]), (
        "Specialist MoE failed to activate on equation with '='!"
    )


def test_evaluate_arithmetic_detailed_and_retention_suite() -> None:
    """Verify detailed arithmetic and multi-domain prose evaluation run cleanly."""
    torch.manual_seed(42)
    tok = CharacterTokenizer(use_base_ascii=True).fit(
        "ADD SUB PAR 0123456789+-= EVEN ODD King Queen Shakespeare year 1594\n"
    )
    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        context_length=32,
    )
    model = MiniGPT(cfg)

    # 1. Detailed arithmetic evaluation
    res_arith = evaluate_arithmetic_detailed(
        model=model,
        tokenizer=tok,
        num_per_category=2,
        seed=42,
    )
    assert "in_domain_accuracy" in res_arith
    assert "ood_accuracy" in res_arith
    assert "accuracy_by_category" in res_arith
    for cat in [
        "ADD_NO_CARRY",
        "ADD_CARRY",
        "SUB_NO_BORROW",
        "SUB_BORROW",
        "PARITY",
        "OOD_EXTRAPOLATION",
    ]:
        assert cat in res_arith["accuracy_by_category"]

    # 2. Text retention suite evaluation
    res_retention = evaluate_text_retention_suite(
        model=model,
        tokenizer=tok,
        max_batches=2,
    )
    assert "clean_prose" in res_retention
    assert "numerical_prose" in res_retention
    assert "mixed_prose_math" in res_retention
    assert "delta_loss" in res_retention["clean_prose"]
    assert "delta_loss" in res_retention["numerical_prose"]
