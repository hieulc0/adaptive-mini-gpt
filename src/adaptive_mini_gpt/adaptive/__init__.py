"""Adaptive capacity, modular routing, and function-preserving growth modules."""

from adaptive_mini_gpt.adaptive.experts import AdaptiveMoELayer, BottleneckExpert
from adaptive_mini_gpt.adaptive.growth import grow_model_depth, grow_model_experts
from adaptive_mini_gpt.adaptive.router import TopKRouter

__all__ = [
    "AdaptiveMoELayer",
    "BottleneckExpert",
    "TopKRouter",
    "grow_model_depth",
    "grow_model_experts",
]
