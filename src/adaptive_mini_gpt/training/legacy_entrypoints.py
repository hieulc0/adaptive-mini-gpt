"""Fail-closed entry-point boundary for known unsafe historical experiments."""

from __future__ import annotations

import os

from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda


def require_historical_reproduction(name: str):
    if os.environ.get("AMG_ALLOW_UNSUPPORTED_HISTORICAL") != "1":
        raise RuntimeError(
            f"{name} is unsupported for new learning/qualification: historical input, "
            "label or evaluation contracts are invalid. Use the canonical foundation path. "
            "Exact historical reproduction only: set AMG_ALLOW_UNSUPPORTED_HISTORICAL=1 "
            "inside a Kaggle/Colab CUDA job; results remain unqualified."
        )
    require_remote_cuda()
