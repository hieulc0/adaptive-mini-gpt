"""REMOTE-REQUIRED: tensor losses/gradients and actual canonical learner update."""

import pytest
import torch
import torch.nn.functional as F

from adaptive_mini_gpt.data.attribute_learning import generate, manifest
from adaptive_mini_gpt.training.learning_state import concept_supervision_loss
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda


def test_attribute_only_gradient_and_legacy_joint_equivalence():
    require_remote_cuda()
    tags = torch.zeros(3, 26, device="cuda")
    tags[:, 10] = 1
    logits = torch.zeros_like(tags, requires_grad=True)
    state = torch.zeros(3, 16, device="cuda", requires_grad=True)
    loss = concept_supervision_loss(state, logits, tags, attribute_ids=(10, 15))
    assert torch.equal(
        loss, F.binary_cross_entropy_with_logits(logits[:, (10, 15)], tags[:, (10, 15)])
    )
    loss.backward()
    assert state.grad is None
    assert logits.grad is not None
    assert logits.grad[:, 10].lt(0).all() and logits.grad[:, 15].gt(0).all()
    assert torch.count_nonzero(logits.grad) == 6
    labels = torch.zeros(3, device="cuda", dtype=torch.long)
    assert torch.equal(
        concept_supervision_loss(state, logits, tags, labels),
        F.binary_cross_entropy_with_logits(logits, tags) + F.cross_entropy(state, labels),
    )
    for ids in ((), (10, 10), (-1,), (26,), (True,)):
        with pytest.raises(ValueError, match="Attribute IDs"):
            concept_supervision_loss(state, logits, tags, attribute_ids=ids)
    tags[0, 10] = float("nan")
    with pytest.raises(ValueError, match="finite binary"):
        concept_supervision_loss(state, logits, tags)


def test_dataset_isolation_balance_and_all_annotation_families():
    require_remote_cuda()
    splits = generate()
    m = manifest(splits)
    assert m == manifest(generate())
    assert all(len(m["splits"][s]["families"]) == 9 for s in splits)
    assert all(0 < n < m["splits"][s]["cases"] for s in splits for n in m["splits"][s]["positives"])
    assert {r.question for r in splits["tiny"]} <= {r.question for r in splits["train"]}
    splits["names"] = splits["train"][:1]
    with pytest.raises(ValueError, match="leakage"):
        manifest(splits)


def test_canonical_attribute_update_does_not_train_transition_branch():
    require_remote_cuda()
    from test_foundation_structure import make_owner

    owner, batch = make_owner()
    before = {n: p.detach().clone() for n, p in owner.model.named_parameters()}
    owner.step(*batch[:3], attribute_ids=(10, 15))
    changed = []
    for n, p in owner.model.named_parameters():
        if not torch.equal(p, before[n]):
            changed.append(n)
        if any(k in n for k in ("transition_net", "state_head", "action_emb", "post_norm")):
            assert torch.equal(p, before[n]) and p.grad is None
    assert any("attribute_head" in n for n in changed)
    assert any("wte" in n for n in changed)
    assert owner.metadata["steps"] == 1
