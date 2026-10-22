"""Stratified Relational Replay Buffer.

Guarantees balanced class exposure (1/3 YES, 1/3 NO, 1/3 UNKNOWN) during continual
adaptation, preventing majority collapse to any single epistemic class.
"""
from __future__ import annotations

import collections
import random
from typing import Any

from adaptive_mini_gpt.uncertainty.epistemic_benchmark import EpistemicTask


class StratifiedRelationalReplayBuffer:
    """Class-stratified replay buffer for epistemic relational tasks."""

    CLASSES = ["YES", "NO", "UNKNOWN"]

    def __init__(self, capacity_per_class: int = 100, seed: int = 42) -> None:
        self.capacity_per_class = capacity_per_class
        self.buffers: dict[str, list[EpistemicTask]] = {c: [] for c in self.CLASSES}
        self.rng = random.Random(seed)

    def add(self, task: EpistemicTask) -> None:
        """Adds a verified task to its respective class buffer."""
        cls_name = task.expected_verdict
        if cls_name not in self.buffers:
            return

        buf = self.buffers[cls_name]
        if len(buf) >= self.capacity_per_class:
            # FIFO eviction or random replace
            buf.pop(0)
        buf.append(task)

    def add_batch(self, tasks: list[EpistemicTask]) -> None:
        """Adds a list of tasks into the replay buffer."""
        for t in tasks:
            self.add(t)

    def sample_stratified_batch(self, batch_size: int) -> list[EpistemicTask]:
        """Samples exactly balanced class proportions from the buffer.

        If batch_size is not divisible by 3, remainders are allocated round-robin.
        """
        if self.total_count() == 0:
            return []

        per_class = batch_size // 3
        rem = batch_size % 3

        quotas = {
            "YES": per_class + (1 if rem > 0 else 0),
            "NO": per_class + (1 if rem > 1 else 0),
            "UNKNOWN": per_class,
        }

        sampled: list[EpistemicTask] = []
        for c in self.CLASSES:
            buf = self.buffers[c]
            quota = quotas[c]
            if not buf or quota <= 0:
                continue

            if len(buf) >= quota:
                sampled.extend(self.rng.sample(buf, quota))
            else:
                # Sample with replacement if buffer has fewer items
                sampled.extend([self.rng.choice(buf) for _ in range(quota)])

        self.rng.shuffle(sampled)
        return sampled

    def class_counts(self) -> dict[str, int]:
        return {c: len(self.buffers[c]) for c in self.CLASSES}

    def total_count(self) -> int:
        return sum(len(b) for b in self.buffers.values())

    def clear(self) -> None:
        for c in self.CLASSES:
            self.buffers[c].clear()
