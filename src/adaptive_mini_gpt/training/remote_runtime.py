"""Fail-closed runtime check for remote scientific entry points."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def require_remote_cuda(device: str = "cuda") -> str:
    """Require a Kaggle/Colab runtime and CUDA; never select a CPU fallback."""
    platform = (
        "kaggle"
        if Path("/kaggle/working").is_dir() and os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        else "colab"
        if Path("/content").is_dir() and os.environ.get("COLAB_RELEASE_TAG")
        else None
    )
    if platform is None or torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Kaggle or Colab with CUDA is required; local/CPU neural execution refused"
        )
    return platform
