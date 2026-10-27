"""Canonical trusted experiment state, updates and reversible candidate boundary.

No domain solver and no automatic structural promotion. Roots are jointly copied
so module/optimizer aliases survive topology changes. Callers must access modules
through the owner after restore; external child/parameter references are invalid.
"""

from __future__ import annotations

import copy
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_mini_gpt.training.optimizer_contract import adamw_groups, optimizer_membership
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda


def validate_consistency_pairs(pairs, batch_size):
    """Explicit disjoint training pairs; never infer equivalence from labels alone."""
    pairs = tuple(tuple(p) for p in pairs)
    used = set()
    if not pairs:
        raise ValueError("Consistency pairs must be nonempty")
    for pair in pairs:
        if len(pair) != 2 or any(
            not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= batch_size for i in pair
        ):
            raise ValueError("Invalid consistency pair indices")
        if pair[0] == pair[1] or any(i in used for i in pair):
            raise ValueError("Consistency pairs must be disjoint and non-self")
        used.update(pair)
    return pairs


def paired_concept_loss(concepts, pairs):
    """Symmetric parameter-free alignment of existing concept vectors."""
    pairs = validate_consistency_pairs(pairs, concepts.shape[0])
    left, right = zip(*pairs, strict=True)
    return (1 - F.cosine_similarity(concepts[list(left)], concepts[list(right)], dim=-1)).mean()


def concept_supervision_loss(state_logits, tag_logits, tags, states=None, *, attribute_ids=None):
    """Explicit supervision: no invented state labels or absent-label negatives.

    Default joint supervision is unchanged. A declared attribute subset selects
    observed columns only; other outputs receive no loss gradient. This is a
    protocol choice, not a claim that historical fully annotated BCE was a bug.
    """
    if tags.shape != tag_logits.shape or tags.ndim != 2:
        raise ValueError("Attribute targets must match B x C logits")
    if attribute_ids is not None:
        ids = tuple(attribute_ids)
        if (
            not ids
            or len(set(ids)) != len(ids)
            or any(
                not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= tags.shape[1]
                for i in ids
            )
        ):
            raise ValueError("Attribute IDs must be unique, nonempty valid columns")
        tag_logits, tags = tag_logits[:, ids], tags[:, ids]
    if not torch.isfinite(tags).all() or not ((tags == 0) | (tags == 1)).all():
        raise ValueError("Observed attribute labels must be finite binary values")
    loss = F.binary_cross_entropy_with_logits(tag_logits, tags)
    if states is not None:
        loss = loss + F.cross_entropy(state_logits, states)
    return loss


def migrate_adamw(model, optimizer, replacements=()):
    """Identity-preserving migration; explicitly mapped resized tensors use prefix copy.

    Unmapped replaced parameters are fresh. Arbitrary index selection/width rewiring
    is unsupported. Retain frozen live parameters/moments, clear frozen gradients.
    """
    if not isinstance(optimizer, torch.optim.AdamW) or len(optimizer.param_groups) != 2:
        raise ValueError("Canonical migration requires two-group AdamW")
    state = optimizer.state
    for old, new in replacements:
        if old.dtype != new.dtype or old.device != new.device:
            raise ValueError("Optimizer migration cannot silently change device or dtype")
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
        model, optimizer, scheduler, external, metadata = snapshot["roots"]
        if type(model) is not type(self.model) or type(optimizer) is not type(self.optimizer):
            raise ValueError("Restore requires matching root types")
        # Root handles stay live; a shared memo also rebinds controller/self aliases.
        memo = {id(model): self.model, id(optimizer): self.optimizer, id(external): self.external}
        model_dict = copy.deepcopy(model.__dict__, memo)
        optimizer_dict = copy.deepcopy(optimizer.__dict__, memo)
        scheduler = copy.deepcopy(scheduler, memo)
        external_dict = copy.deepcopy(dict(external), memo)
        metadata = copy.deepcopy(metadata, memo)
        self.model.__dict__.clear()
        self.model.__dict__.update(model_dict)
        self.optimizer.__dict__.clear()
        self.optimizer.__dict__.update(optimizer_dict)
        self.scheduler = scheduler
        self.external.clear()
        self.external.update(external_dict)
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

    def step(
        self,
        tokens,
        actions,
        tags,
        states=None,
        *,
        attribute_ids=None,
        loss_scale=1.0,
        consistency_pairs=None,
        consistency_weight=0.0,
    ):
        require_remote_cuda(str(tokens.device))
        if not math.isfinite(loss_scale) or loss_scale <= 0:
            raise ValueError("Loss scale must be finite and positive")
        if not math.isfinite(consistency_weight) or consistency_weight < 0:
            raise ValueError("Consistency weight must be finite and nonnegative")
        if (consistency_pairs is None) != (consistency_weight == 0):
            raise ValueError("Positive consistency weight requires explicit pairs and vice versa")
        if consistency_pairs is not None:
            consistency_pairs = validate_consistency_pairs(consistency_pairs, tokens.shape[0])
            ids = tuple(attribute_ids) if attribute_ids is not None else tuple(range(tags.shape[1]))
            for left, right in consistency_pairs:
                if not torch.equal(tags[left, ids], tags[right, ids]) or (
                    states is not None and not torch.equal(states[left], states[right])
                ):
                    raise ValueError("Consistency pairs require matching observed targets")
        if any(optimizer_membership(self.model, self.optimizer).values()):
            raise RuntimeError("Optimizer ownership violation")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        state_logits, tag_logits, concepts, _ = self.model(tokens, actions)
        loss = concept_supervision_loss(
            state_logits, tag_logits, tags, states, attribute_ids=attribute_ids
        )
        if consistency_pairs is not None:
            loss = loss + consistency_weight * paired_concept_loss(concepts, consistency_pairs)
        if loss_scale != 1.0:
            loss = loss * loss_scale
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
        roots = (self.model, self.optimizer, self.external)

        def rollback():
            self.model, self.optimizer, self.external = roots
            self.restore(before)

        self._in_candidate = True
        try:
            modify(self)
            if any(
                current is not original
                for current, original in zip(
                    (self.model, self.optimizer, self.external), roots, strict=True
                )
            ):
                raise RuntimeError("Candidate cannot replace owned root handles")
            if any(optimizer_membership(self.model, self.optimizer).values()):
                raise RuntimeError("Candidate optimizer ownership violation")
            accepted = bool(validate(self))
            if not accepted:
                rollback()
            return accepted
        except BaseException:
            rollback()
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
            # Same-filesystem hard-link publication is atomic and cannot clobber
            # an existing immutable checkpoint, including a concurrent publisher.
            os.link(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
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
