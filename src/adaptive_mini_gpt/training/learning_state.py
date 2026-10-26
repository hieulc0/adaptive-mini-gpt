"""Canonical trusted experiment state, updates and reversible candidate boundary.

No domain solver and no automatic structural promotion. Roots are jointly copied
so module/optimizer aliases survive topology changes. Callers must access modules
through the owner after restore; external child/parameter references are invalid.
"""

from __future__ import annotations

import copy
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_mini_gpt.training.optimizer_contract import adamw_groups, optimizer_membership
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda


def migrate_adamw(model, optimizer, replacements=()):
    """Identity-preserving migration; explicitly mapped resized tensors use prefix copy.

    Unmapped replaced parameters are fresh. Arbitrary index selection/width rewiring
    is unsupported. Retain frozen live parameters/moments, clear frozen gradients.
    """
    if not isinstance(optimizer, torch.optim.AdamW) or len(optimizer.param_groups) != 2:
        raise ValueError("Canonical migration requires two-group AdamW")
    state = optimizer.state
    for old, new in replacements:
        if old.ndim != new.ndim or any(a > b for a, b in zip(old.shape, new.shape, strict=True)):
            raise ValueError("Only explicitly mapped prefix expansion is supported")
        if old in state:
            migrated = {}
            for key, value in state[old].items():
                if isinstance(value, torch.Tensor) and value.ndim > 0:
                    if value.shape != old.shape:
                        raise ValueError("Unsupported optimizer tensor state")
                    expanded = torch.zeros_like(new)
                    expanded[tuple(slice(0, n) for n in old.shape)] = value
                    migrated[key] = expanded
                else:
                    migrated[key] = copy.deepcopy(value)
            state[new] = migrated
    groups = adamw_groups(model, 0, include_frozen=True)
    old_decay = optimizer.param_groups[0]["weight_decay"]
    for i, group in enumerate(groups):
        optimizer.param_groups[i]["params"] = group["params"]
        optimizer.param_groups[i]["weight_decay"] = old_decay if i == 0 else 0.0
    live = set(model.parameters())
    for p in list(state):
        if p not in live:
            del state[p]
    for p in live:
        if not p.requires_grad:
            p.grad = None
    if any(optimizer_membership(model, optimizer).values()):
        raise RuntimeError("Invalid optimizer ownership after migration")


class LearningState:
    """One continuing model and all declared mutable learning state."""

    VERSION = 1

    def __init__(self, model, *, lr=0.002, weight_decay=0.0, external=None, metadata=None):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            adamw_groups(model, weight_decay, include_frozen=True), lr=lr
        )
        self.scheduler = None
        self.external = (
            external if external is not None else {"replay": [], "memory": {}, "controller": {}}
        )
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("steps", 0)
        self._in_candidate = False

    def snapshot(self):
        for module in self.model.modules():
            if module._forward_hooks or module._backward_hooks:
                raise NotImplementedError(
                    "Hook-bearing model needs an explicit serializable hook contract"
                )
        if any(p._backward_hooks for p in self.model.parameters()):
            raise NotImplementedError("Parameter hooks need an explicit serializable hook contract")
        return {
            "version": self.VERSION,
            "roots": copy.deepcopy(
                (self.model, self.optimizer, self.scheduler, self.external, self.metadata)
            ),
            "gradients": {
                n: p.grad.detach().clone()
                for n, p in self.model.named_parameters()
                if p.grad is not None
            },
            "rng": (
                random.getstate(),
                np.random.get_state(),
                torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            ),
        }

    def restore(self, snapshot):
        if snapshot["version"] != self.VERSION:
            raise ValueError("Unsupported learning-state version")
        model, optimizer, scheduler, external, metadata = copy.deepcopy(snapshot["roots"])
        # Preserve root handles; all child ownership comes from this joint graph.
        self.model.__dict__.clear()
        self.model.__dict__.update(model.__dict__)
        self.optimizer.__dict__.clear()
        self.optimizer.__dict__.update(optimizer.__dict__)
        if scheduler is not None:
            scheduler.optimizer = self.optimizer
        self.scheduler = scheduler
        self.external.clear()
        self.external.update(external)
        self.metadata = metadata
        for name, p in self.model.named_parameters():
            p.grad = snapshot["gradients"].get(name)
            if p.grad is not None:
                p.grad = p.grad.clone()
        py, np_rng, cpu, cuda = snapshot["rng"]
        random.setstate(py)
        np.random.set_state(np_rng)
        torch.set_rng_state(cpu.cpu())
        if cuda is not None:
            if len(cuda) != torch.cuda.device_count():
                raise RuntimeError("Exact resumed learning requires matching CUDA device count")
            torch.cuda.set_rng_state_all([s.cpu() for s in cuda])
        if any(optimizer_membership(self.model, self.optimizer).values()):
            raise RuntimeError("Restored optimizer ownership is invalid")

    def set_mode(self, mode):
        if mode not in ("frozen_backbone", "selective_plasticity", "trainable_backbone"):
            raise ValueError("Unsupported canonical plasticity mode")
        self.model.mode = mode
        self.model._configure_plasticity()
        migrate_adamw(self.model, self.optimizer)

    def step(self, tokens, actions, tags, states):
        require_remote_cuda(str(tokens.device))
        if any(optimizer_membership(self.model, self.optimizer).values()):
            raise RuntimeError("Optimizer ownership violation")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        state_logits, tag_logits, _, _ = self.model(tokens, actions)
        loss = F.cross_entropy(state_logits, states) + F.binary_cross_entropy_with_logits(
            tag_logits, tags
        )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss")
        loss.backward()
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()
        ):
            raise ValueError("Nonfinite gradient")
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.metadata["steps"] += 1
        return float(loss.detach())

    def candidate(self, modify, validate):
        if self._in_candidate:
            raise RuntimeError("Nested candidates are unsupported")
        before = self.snapshot()
        self._in_candidate = True
        try:
            modify(self)
            if any(optimizer_membership(self.model, self.optimizer).values()):
                raise RuntimeError("Candidate optimizer ownership violation")
            accepted = bool(validate(self))
            if not accepted:
                self.restore(before)
            return accepted
        except BaseException:
            self.restore(before)
            raise
        finally:
            self._in_candidate = False

    def save(self, path):
        if self._in_candidate:
            raise RuntimeError("Candidate checkpoints cannot be published")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
        try:
            with os.fdopen(fd, "wb") as f:
                torch.save(self.snapshot(), f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path):
        # Only same-code trusted artifacts produced by this owner are supported.
        snapshot = torch.load(path, weights_only=False)
        model = copy.deepcopy(snapshot["roots"][0])
        owner = cls(model)
        owner.restore(snapshot)
        return owner
