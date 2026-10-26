"""REMOTE-REQUIRED: real neural/tensor qualification, never invoked locally."""

import torch
import torch.nn.functional as F

from adaptive_mini_gpt.model.attention import apply_rotary_emb
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda


def test_production_targets_are_already_shifted_and_both_masks_work():
    require_remote_cuda()
    cfg = ModelConfig(
        vocab_size=8, d_model=8, n_heads=2, n_layers=1, d_ff=16, context_length=8, dropout=0
    )
    model = MiniGPT(cfg).cuda().eval()
    x = torch.tensor([[0, 1, 2, 3]], device="cuda")
    y = torch.tensor([[1, -1, 3, -100]], device="cuda")
    logits, loss = model(x, y)
    expected = F.cross_entropy(
        logits.reshape(-1, 8), torch.tensor([1, -100, 3, -100], device="cuda"), ignore_index=-100
    )
    assert torch.equal(loss, expected)


def test_rope_half_split_frequency_pairing_preserves_vector_norm():
    require_remote_cuda()
    # Different frequencies must be paired with the same half-split rotation.
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], device="cuda")
    freq = torch.tensor([0.7, 0.01], device="cuda")
    cos = torch.cat([freq.cos(), freq.cos()])
    sin = torch.cat([freq.sin(), freq.sin()])
    rotated = apply_rotary_emb(x, cos, sin)
    assert torch.allclose(rotated.square().sum(), x.square().sum(), atol=1e-5)
    # Inspect the actual attention path via hooks to guard frequency construction.
    cfg = ModelConfig(
        vocab_size=8,
        d_model=8,
        n_heads=2,
        n_layers=1,
        d_ff=16,
        context_length=8,
        pos_encoding="rope",
        dropout=0,
    )
    from unittest.mock import patch

    from adaptive_mini_gpt.model.attention import CausalSelfAttention

    original = apply_rotary_emb

    def checked_rotation(q, c, s):
        result = original(q, c, s)
        assert torch.allclose(result.square().sum(-1), q.square().sum(-1), atol=1e-5)
        return result

    attn = CausalSelfAttention(cfg).cuda().eval()
    with patch("adaptive_mini_gpt.model.attention.apply_rotary_emb", side_effect=checked_rotation):
        attn(torch.randn(1, 4, 8, device="cuda"))
