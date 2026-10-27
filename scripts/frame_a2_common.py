"""Stage A.2 experimental state interventions, never automatic production policy."""

from __future__ import annotations

import copy
import random

from adaptive_mini_gpt.training.learning_state import LearningState
from adaptive_mini_gpt.training.reproducibility import seed_all

ROOT_HASH = "3e93d7238bfdb706ddfd49ee9903fed8b1e6f577529537f1e21cae12e905b262"
ROOT_TENSOR_HASH = "aed7480606a1a60e38ee7a95d22b5bcf14141f034f76cb45fb59680de277a33e"
OLD_DATA_HASH = "c2f7f254a540367f037f60f1832e1dc7c6a4fe9d5c3267f5f1d9fdaf7ff08ffb"
STRATEGIES = ("inherit", "moments", "age", "complete", "rebase")


def intervene_backbone(owner, strategy):
    if strategy not in STRATEGIES:
        raise ValueError("Unknown experimental optimizer strategy")
    if owner.scheduler is not None:
        raise ValueError("This diagnostic requires constant LR/no scheduler")
    affected = []
    if strategy in ("inherit", "rebase"):
        return affected
    for name, param in owner.model.named_parameters():
        if not name.startswith("student.") or param not in owner.optimizer.state:
            continue
        state = owner.optimizer.state[param]
        if strategy == "complete":
            del owner.optimizer.state[param]
        elif strategy == "moments":
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                if key in state:
                    state[key].zero_()
        else:
            state["step"].zero_()
        affected.append(name)
    return affected


def fork_owner(parent, seed):
    snapshot = parent.snapshot()
    owner = LearningState(copy.deepcopy(parent.model))
    owner.restore(snapshot)
    owner.external["frame_a2_sampler"] = random.Random(seed)
    seed_all(seed)
    return owner


def state_digest(value):
    """Typed nested research-state hash; not a serializer for arbitrary object graphs."""
    import hashlib

    import numpy as np
    import torch

    h = hashlib.sha256()

    def visit(v):
        h.update(type(v).__name__.encode() + b":")
        if isinstance(v, torch.Tensor):
            h.update(str((str(v.dtype), tuple(v.shape))).encode())
            h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, np.ndarray):
            h.update(str((str(v.dtype), v.shape)).encode())
            h.update(v.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=lambda k: (type(k).__name__, repr(k))):
                visit(k)
                visit(v[k])
        elif isinstance(v, (list, tuple)):
            for item in v:
                visit(item)
        elif isinstance(v, random.Random):
            visit(v.getstate())
        elif v is None or isinstance(v, (str, int, float, bool, bytes)):
            h.update(repr(v).encode())
        else:
            raise TypeError(f"Unsupported diagnostic hash type: {type(v).__name__}")
        h.update(b";")

    visit(value)
    return h.hexdigest()


def record_stage_provenance(
    owner, *, training_sha, publication_sha, parent_hash, data_hash, stage, comparison_hash=None
):
    """Preserve ancestor attribution; never mistake inherited metadata for producer."""
    previous = {k: owner.metadata.get(k) for k in ("git_sha", "data_hash", "stage", "steps")}
    owner.metadata.update(
        frame_a2_parent_provenance=previous,
        git_sha=training_sha,
        checkpoint_publication_sha=publication_sha,
        parent_checkpoint_sha256=parent_hash,
        comparison_checkpoint_sha256=comparison_hash,
        data_hash=data_hash,
        stage=stage,
        promoted=False,
    )
