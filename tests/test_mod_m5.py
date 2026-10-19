"""Tests for M5 Mixture-of-Depths (MoD), selective token adaptation, and CLI growth."""

from __future__ import annotations

import torch

from adaptive_mini_gpt.adaptive.growth import enable_model_mod, grow_model_experts
from adaptive_mini_gpt.data.dataset import TokenDataset, build_1d_answer_mask
from adaptive_mini_gpt.data.tokenizer import CharacterTokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT, TransformerBlock


def test_mod_router_gradients_and_flop_reduction() -> None:
    """Verify M5 MoD router receives non-zero gradients and reduces active_params_per_token."""
    torch.manual_seed(42)
    cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        context_length=16,
        adaptive_enabled=True,
        n_experts=4,
        top_k=2,
        mod_enabled=False,
    )
    model = MiniGPT(cfg)
    active_before_mod = model.get_active_params_per_token()

    enable_model_mod(model, capacity_factor=0.5)
    active_after_mod = model.get_active_params_per_token()
    assert active_after_mod < active_before_mod

    model.train()
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    _, loss = model(x, y)
    assert loss is not None
    loss.backward()

    block0 = model.blocks[0]
    assert isinstance(block0, TransformerBlock)
    assert block0.mod_router is not None
    assert block0.mod_router.gate.weight.grad is not None
    assert block0.mod_router.gate.weight.grad.abs().sum().item() > 0.0
    assert "mod_active_fraction" in model.last_aux_info
    assert "mod_aux_loss" in model.last_aux_info


def test_mod_causal_independence_in_eval_mode() -> None:
    """In eval mode, MoD must use local thresholding so future tokens never alter past logits."""
    torch.manual_seed(42)
    cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        d_ff=64,
        context_length=16,
        dropout=0.0,
        mod_enabled=True,
        mod_capacity_factor=0.5,
    )
    model = MiniGPT(cfg)
    model.eval()

    x1 = torch.randint(0, 64, (1, 12))
    x2 = x1.clone()
    k = 6
    x2[:, k:] = (x2[:, k:] + 3) % 64

    with torch.no_grad():
        logits1, _ = model(x1)
        logits2, _ = model(x2)

    assert torch.allclose(logits1[:, :k, :], logits2[:, :k, :], atol=1e-6), (
        "M5 MoD violated causal independence in eval mode!"
    )


def test_selective_unseen_token_embedding_mask() -> None:
    """Only specified unseen tokens in wte may update; lm_head stays strictly frozen."""
    torch.manual_seed(42)
    cfg = ModelConfig(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        context_length=8,
        tie_word_embeddings=False,
    )
    model = MiniGPT(cfg)
    grow_model_experts(
        model,
        add_experts=2,
        freeze_backbone=True,
        trainable_token_ids=[30, 31],
    )
    wte_before = model.wte.weight.detach().clone()
    assert not model.lm_head.weight.requires_grad, "lm_head must stay frozen!"

    opt = model.configure_optimizers(learning_rate=1e-2, weight_decay=0.01)
    x = torch.tensor([[1, 2, 3, 4, 30, 31, 5, 6]])
    y = torch.tensor([[2, 3, 4, 30, 31, 5, 6, 7]])

    opt.zero_grad()
    _, loss = model(x, y)
    assert loss is not None
    loss.backward()
    opt.step()

    wte_after = model.wte.weight.detach()
    # Seen tokens (0..29) must have strictly 0.0000 change
    assert torch.equal(wte_before[:30], wte_after[:30])
    # Unseen tokens (30, 31) must have updated
    assert not torch.equal(wte_before[30:], wte_after[30:])


def test_vectorized_dataset_masking() -> None:
    """Vectorized 1D target mask must match expected prompt and answer positions."""
    tok = CharacterTokenizer(use_base_ascii=True).fit("ADD 12+34=46\n")
    seq = "ADD 12+34=46\n"
    tokens = torch.tensor(tok.encode(seq) * 4, dtype=torch.long)
    eq_id = tok.stoi["="]
    nl_id = tok.stoi["\n"]

    mask_1d = build_1d_answer_mask(tokens, eq_id, nl_id)
    assert mask_1d.size(0) == tokens.size(0)

    ds = TokenDataset(
        tokens,
        context_length=12,
        mask_prompt_before_equals=True,
        equals_token_id=eq_id,
        newline_token_id=nl_id,
    )
    x, y = ds.get_batch(batch_size=4)
    assert (y == -1).any()
    assert (y != -1).any()
