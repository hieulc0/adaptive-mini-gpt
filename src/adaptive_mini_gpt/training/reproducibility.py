"""Explicit independent RNG streams for reproducible experimental controls."""

from __future__ import annotations

import hashlib
import random
from contextlib import contextmanager

import numpy as np
import torch


def stream_seed(seed: int, name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:4], "big")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def isolated_rng(seed: int, name: str):
    """Restore the caller's RNGs; arm initialization cannot depend on arm order."""
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        try:
            seed_all(stream_seed(seed, name))
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
