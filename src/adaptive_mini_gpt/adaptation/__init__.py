"""Fast Neural Adaptation and Generalization Package (M30)."""

from __future__ import annotations

from adaptive_mini_gpt.adaptation.curriculum import (
    MultiScaleEpisode,
    build_multiscale_episode_a,
    build_multiscale_episode_b,
    build_multiscale_episode_c,
)

__all__ = [
    "MultiScaleEpisode",
    "build_multiscale_episode_a",
    "build_multiscale_episode_b",
    "build_multiscale_episode_c",
]
