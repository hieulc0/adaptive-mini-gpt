"""Parameter ownership and decay contracts shared by audited entry points."""

from __future__ import annotations

import torch.nn as nn


def adamw_groups(model, weight_decay: float, *, include_frozen: bool = False):
    embedding_ids = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    parameters = {id(p): p for p in model.parameters() if p.requires_grad or include_frozen}
    return [
        {
            "params": [
                p for p in parameters.values() if p.dim() >= 2 and id(p) not in embedding_ids
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for p in parameters.values() if p.dim() < 2 or id(p) in embedding_ids],
            "weight_decay": 0.0,
        },
    ]


def optimizer_membership(model, optimizer):
    expected = {id(p): name for name, p in model.named_parameters() if p.requires_grad or include_frozen}
    registered = [id(p) for group in optimizer.param_groups for p in group["params"]]
    live = {id(p) for p in model.parameters()}
    return {
        "missing_trainable": [name for key, name in expected.items() if key not in registered],
        "duplicate_entries": len(registered) - len(set(registered)),
        "stale_entries": len(set(registered) - live),
    }
