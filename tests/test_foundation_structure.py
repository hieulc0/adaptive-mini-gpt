"""REMOTE-REQUIRED: real model invariants and learning-state recovery."""

from __future__ import annotations

import copy
import os
import random
import subprocess
import sys

import numpy as np
import torch

from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda


def test_negative_logit_expert_growth_preserves_outputs_and_new_gradients():
    require_remote_cuda()
    from adaptive_mini_gpt.adaptive.experts import AdaptiveMoELayer

    torch.manual_seed(42)
    m = AdaptiveMoELayer(8, 4, 2, top_k=2, dropout=0).cuda().eval()
    with torch.no_grad():
        m.router.gate.weight[0].fill_(-1)
        m.router.gate.weight[1].fill_(-2)
    x = torch.ones(1, 3, 8, device="cuda")
    before = m(x)[0].detach().clone()
    m.add_experts(2)
    assert torch.equal(before, m(x)[0])
    m(x)[0].square().sum().backward()
    assert m.experts[-1].w_down.weight.grad.abs().sum() > 0
    # Cohort boundaries survive standalone state-dict reconstruction.
    rebuilt = AdaptiveMoELayer(8, 4, 4, top_k=2, dropout=0).cuda().eval()
    rebuilt.load_state_dict(m.state_dict())
    assert torch.equal(before, rebuilt(x)[0])


def test_dynamic_routing_is_prefix_and_batch_independent():
    require_remote_cuda()
    from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import TextStudentM103

    torch.manual_seed(42)
    model = TextStudentM103(prune_orphaned_lm_head=True).cuda().eval()
    with torch.no_grad():
        model.base_model.wpe.weight.zero_()
        model.base_model.wte.weight[1, 0] = 1
        model.base_model.wte.weight[2, 0] = -100
        model.base_model.wte.weight[3, 0] = 100
        for i, r in enumerate(model.layer_routers):
            for p in r.parameters():
                p.zero_()
            if i == 0:
                r.router[0].weight[0, 0] = 1
                r.router[2].weight[1, 0] = 1
                r.router[2].bias[1] = -0.1
            else:
                r.router[2].bias[1] = 10
    a = torch.tensor([[1, 2, 2]], device="cuda")
    b = torch.tensor([[1, 3, 3]], device="cuda")
    with torch.no_grad():
        short = model.forward_lm(a[:, :1], allow_skipping=True)
        one = model.forward_lm(a, allow_skipping=True)
        other = model.forward_lm(b, allow_skipping=True)
        paired = model.forward_lm(torch.cat([a, b]), allow_skipping=True)
    assert torch.allclose(one[:, :1], other[:, :1], atol=1e-5)
    assert torch.allclose(one[:, :1], short, atol=1e-5)
    assert torch.allclose(one[0], paired[0], atol=1e-5)


def make_owner():
    from adaptive_mini_gpt.natural_language_learning_m103.semantic_bottleneck_m104_e import (
        IntegratedStudentM104_E,
    )
    from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import TextStudentM103
    from adaptive_mini_gpt.training.learning_state import LearningState

    torch.manual_seed(42)
    model = IntegratedStudentM104_E(
        TextStudentM103(prune_orphaned_lm_head=True),
        mode="selective_plasticity",
        default_strategy="question_only",
    ).cuda()
    owner = LearningState(model)
    owner.scheduler = torch.optim.lr_scheduler.StepLR(owner.optimizer, step_size=2, gamma=0.5)
    batch = (
        torch.tensor([[257, 261, 65, 262], [257, 261, 66, 262]], device="cuda"),
        torch.tensor([0, 2], device="cuda"),
        torch.zeros(2, 26, device="cuda"),
        torch.tensor([0, 1], device="cuda"),
    )
    owner.external.update(next_batch=batch, memory={"fact": [1]}, controller={"count": 2})
    return owner, batch


def test_topology_optimizer_rollback_and_process_restart(tmp_path):
    require_remote_cuda()
    from adaptive_mini_gpt.training.learning_state import LearningState, migrate_adamw

    owner, batch = make_owner()
    owner.external["root_refs"] = (owner.model, owner.optimizer)
    owner.step(*batch)
    owner.model.eval()
    snapshot = owner.snapshot()
    old_n = len(owner.model.student.base_model.blocks)

    def mutation(s):
        s.model.student.evolvable_backbone.insert_block_identity(1)
        s.model.student.evolvable_backbone.remove_block(0)
        s.set_mode("trainable_backbone")
        s.model.extra = torch.nn.Linear(128, 128).cuda()
        migrate_adamw(s.model, s.optimizer)
        s.external["memory"]["fact"].append(9)
        s.external["controller"]["count"] += 1
        s.metadata["candidate"] = True
        s.step(*batch)

    assert not owner.candidate(mutation, lambda s: False)
    assert len(owner.model.student.base_model.blocks) == old_n
    assert not hasattr(owner.model, "extra")
    assert owner.external["memory"] == {"fact": [1]}
    assert owner.external["controller"] == {"count": 2}
    assert owner.external["root_refs"][0] is owner.model
    assert owner.external["root_refs"][1] is owner.optimizer
    assert owner.metadata == snapshot["roots"][4]
    for k, v in snapshot["roots"][0].state_dict().items():
        assert torch.equal(owner.model.state_dict()[k], v)
    # Canonical checkpoint contains the joint topology/optimizer graph.
    source, resumed = tmp_path / "state.pt", tmp_path / "next.pt"
    owner.save(source)
    import pytest

    with pytest.raises(FileExistsError):
        owner.save(source)
    expected_loss = owner.step(*batch)
    expected_rng = [
        random.random(),
        float(np.random.random()),
        torch.rand(2).tolist(),
        torch.rand(2, device="cuda").tolist(),
    ]
    expected = copy.deepcopy(owner.model.state_dict())
    expected_opt = copy.deepcopy(owner.optimizer.state_dict())
    env = os.environ.copy()
    subprocess.run(
        [sys.executable, "scripts/verify_foundation_restart.py", str(source), str(resumed)],
        check=True,
        env=env,
        timeout=90,
    )
    restored = LearningState.load(resumed)
    assert restored.external["root_refs"][0] is restored.model
    assert restored.external["root_refs"][1] is restored.optimizer
    assert restored.metadata["last_loss"] == expected_loss
    assert restored.metadata["rng_after_update"] == expected_rng
    for k, v in expected.items():
        assert torch.equal(restored.model.state_dict()[k], v)
    actual_opt = restored.optimizer.state_dict()
    assert actual_opt["param_groups"] == expected_opt["param_groups"]
    for key, value in expected_opt["state"].items():
        for name, tensor in value.items():
            assert torch.equal(actual_opt["state"][key][name], tensor)
    assert restored.scheduler.last_epoch == owner.scheduler.last_epoch


def test_optimizer_prefix_migration_and_freeze_transitions():
    require_remote_cuda()
    from adaptive_mini_gpt.adaptive.growth import grow_model_experts
    from adaptive_mini_gpt.model import MiniGPT, ModelConfig
    from adaptive_mini_gpt.training.learning_state import LearningState, migrate_adamw

    m = MiniGPT(
        ModelConfig(
            vocab_size=16,
            d_model=8,
            n_layers=1,
            n_heads=2,
            d_ff=16,
            context_length=8,
            dropout=0,
            adaptive_enabled=True,
            n_experts=2,
        )
    ).cuda()
    state = LearningState(m)
    x = torch.tensor([[1, 2, 3]], device="cuda")
    loss = m(x, x)[1]
    loss.backward()
    state.optimizer.step()
    state.optimizer.zero_grad(set_to_none=True)
    old = m.blocks[0].moe.router.gate.weight
    moments = state.optimizer.state[old]["exp_avg"].clone()
    before = m.eval()(x)[0].detach().clone()
    grow_model_experts(m, 2)
    new = m.blocks[0].moe.router.gate.weight
    migrate_adamw(m, state.optimizer, [(old, new)])
    assert torch.equal(state.optimizer.state[new]["exp_avg"][:2], moments)
    assert torch.count_nonzero(state.optimizer.state[new]["exp_avg"][2:]) == 0
    assert torch.equal(before, m(x)[0])
    owner, batch = make_owner()
    owner.step(*batch)
    parameter = owner.model.student.base_model.wte.weight
    saved = parameter.detach().clone()
    saved_moment = owner.optimizer.state[parameter]["exp_avg"].clone()
    owner.set_mode("frozen_backbone")
    owner.step(*batch)
    assert torch.equal(saved, parameter)
    assert torch.equal(saved_moment, owner.optimizer.state[parameter]["exp_avg"])
    owner.set_mode("selective_plasticity")
    assert parameter in owner.optimizer.state


def test_tied_language_output_drift_is_rejected_and_restored():
    require_remote_cuda()
    owner, batch = make_owner()
    owner.model.eval()
    with torch.no_grad():
        before = owner.model.student.forward_lm(batch[0]).softmax(-1).clone()
    observed = []

    def validate(s):
        s.model.eval()
        with torch.no_grad():
            after = s.model.student.forward_lm(batch[0]).softmax(-1)
        observed.append(not torch.equal(before, after))
        return torch.equal(before, after)

    assert not owner.candidate(lambda s: s.step(*batch), validate)
    assert observed == [True]
    with torch.no_grad():
        assert torch.equal(before, owner.model.student.forward_lm(batch[0]).softmax(-1))


def test_depth_insertion_preserves_dtype_and_adaptive_outputs():
    require_remote_cuda()
    from adaptive_mini_gpt.dynamic_transformer_evolution.dynamic_transformer_substrate import (
        DynamicEvolvableBackbone,
    )
    from adaptive_mini_gpt.model import MiniGPT, ModelConfig

    m = (
        MiniGPT(
            ModelConfig(
                vocab_size=8,
                d_model=8,
                n_heads=2,
                n_layers=1,
                d_ff=16,
                context_length=8,
                dropout=0,
                adaptive_enabled=True,
                n_experts=2,
            )
        )
        .cuda()
        .double()
        .eval()
    )
    dynamic = DynamicEvolvableBackbone(m).cuda().double().eval()
    x = torch.tensor([[1, 2, 3]], device="cuda")
    with torch.no_grad():
        before = dynamic.hidden_states(x).clone()
    dynamic.insert_block_identity(1)
    dynamic.eval()
    assert next(m.blocks[1].parameters()).dtype == torch.float64
    with torch.no_grad():
        assert torch.equal(before, dynamic.hidden_states(x))
    assert m.config.n_layers == 2


def test_rejection_restores_rng_flags_and_blocks_checkpoint_publication(tmp_path):
    require_remote_cuda()
    import pytest

    owner, batch = make_owner()
    owner.model.train()
    owner.model.student.base_model.blocks[0].eval()
    flags = {n: m.training for n, m in owner.model.named_modules()}
    masks = {n: p.requires_grad for n, p in owner.model.named_parameters()}
    snapshot = owner.snapshot()

    def draws():
        return [
            random.random(),
            float(np.random.random()),
            torch.rand(2).tolist(),
            torch.rand(2, device="cuda").tolist(),
        ]

    expected = draws()
    owner.restore(snapshot)
    protected = tmp_path / "checkpoint.pt"
    protected.write_bytes(b"immutable")

    def mutate(state):
        draws()
        state.model.eval()
        state.set_mode("trainable_backbone")
        state.external["replay"].append({"mutable": [1]})
        with pytest.raises(RuntimeError, match="cannot be published"):
            state.save(protected)

    assert not owner.candidate(mutate, lambda s: False)
    assert draws() == expected
    assert protected.read_bytes() == b"immutable"
    assert flags == {n: m.training for n, m in owner.model.named_modules()}
    assert masks == {n: p.requires_grad for n, p in owner.model.named_parameters()}
    assert owner.external["replay"] == []


def test_root_replacement_is_rejected_without_corrupting_live_owner():
    require_remote_cuda()
    import pytest

    owner, _ = make_owner()
    original = owner.model

    def replace(state):
        state.model = torch.nn.Linear(1, 1).cuda()

    with pytest.raises(RuntimeError, match="root handles"):
        owner.candidate(replace, lambda s: True)
    assert owner.model is original
    assert hasattr(owner.model, "student")
