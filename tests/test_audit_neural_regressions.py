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


def test_explicit_masks_cannot_disable_causality():
    require_remote_cuda()
    from adaptive_mini_gpt.model.attention import CausalSelfAttention

    for pos in ("absolute", "rope", "alibi"):
        cfg = ModelConfig(
            vocab_size=8,
            d_model=8,
            n_heads=2,
            n_layers=1,
            d_ff=16,
            context_length=8,
            pos_encoding=pos,
            dropout=0,
        )
        model = CausalSelfAttention(cfg).cuda().eval()
        x = torch.randn(1, 4, 8, device="cuda")
        changed = x.clone()
        changed[:, 2:] += 9
        for mask in (
            torch.ones(4, 4, dtype=torch.bool, device="cuda"),
            torch.zeros(4, 4, device="cuda"),
        ):
            assert torch.allclose(model(x, mask)[:, :2], model(changed, mask)[:, :2], atol=1e-5)
        blocked = torch.ones(4, 4, dtype=torch.bool, device="cuda")
        blocked[:, 1] = False
        changed = x.clone()
        changed[:, 1] += 9
        assert torch.allclose(model(x, blocked)[:, 2:], model(changed, blocked)[:, 2:], atol=1e-5)


def test_empty_target_supervision_fails_explicitly():
    require_remote_cuda()
    import pytest

    model = MiniGPT(
        ModelConfig(
            vocab_size=8, d_model=8, n_heads=2, n_layers=1, d_ff=16, context_length=8, dropout=0
        )
    ).cuda()
    x = torch.tensor([[0, 1, 2, 3]], device="cuda")
    with pytest.raises(ValueError, match="supervised"):
        model(x, torch.full_like(x, -100))


def test_unfreezing_unregistered_parameters_is_rejected():
    require_remote_cuda()
    from adaptive_mini_gpt.natural_language_learning_m103.self_modification_m105_a import (
        ModificationOperatorTypeM105_A,
        NeuralModificationProposalM105_A,
        StructuralRollbackControllerM105_A,
    )
    from adaptive_mini_gpt.natural_language_learning_m103.semantic_bottleneck_m104_e import (
        IntegratedStudentM104_E,
    )
    from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import TextStudentM103

    model = IntegratedStudentM104_E(
        TextStudentM103(prune_orphaned_lm_head=True), mode="frozen_backbone"
    ).cuda()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    controller = StructuralRollbackControllerM105_A(model, optimizer)
    prop = NeuralModificationProposalM105_A(
        "unfreeze",
        ModificationOperatorTypeM105_A.SELECTIVE_BACKBONE_UPDATE,
        "student",
        {"mode": "selective_plasticity"},
        "ownership check",
    )
    result = controller.apply_and_evaluate(prop, val_eval_fn=lambda: 100.0)
    assert not result.accepted and result.rollback_executed
    assert "Optimizer ownership" in result.validation_status
    assert model.mode == "frozen_backbone"


def test_nul_byte_is_not_padding_in_entity_pool():
    require_remote_cuda()
    from types import SimpleNamespace

    from adaptive_mini_gpt.natural_language_learning_m103.semantic_bottleneck_m104_e import (
        IntegratedStudentM104_E,
    )
    from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
        ByteTextTokenizerM103 as T,
    )

    tokens = torch.tensor([[T.QUESTION_ID, 0, T.ANSWER_ID, T.PAD_ID]], device="cuda")
    hidden = torch.tensor([[[2.0], [6.0], [12.0], [100.0]]], device="cuda")
    entity, _ = IntegratedStudentM104_E.pool_representations(SimpleNamespace(), hidden, tokens)
    assert entity.item() == 4.0
