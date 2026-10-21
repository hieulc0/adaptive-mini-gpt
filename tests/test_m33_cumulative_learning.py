"""Unit tests for M33: Repeated Self-Learning and Cumulative Skill Acquisition."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.learning.cumulative_learner import (
    build_m33_arithmetic_episode,
    build_m33_language_episode,
    build_m33_logic_episode,
)
from adaptive_mini_gpt.learning.lineage_manager import (
    LineageManager,
    compute_model_hash,
)
from adaptive_mini_gpt.learning.strategy_selector import (
    LearningStrategy,
    StrategySelector,
)
from adaptive_mini_gpt.model.transformer import MiniGPT, ModelConfig


class TinyDummyModel(nn.Module):
    """Tiny neural model for testing lineage hashing and management."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_lineage_manager_hashing_and_promotion() -> None:
    """Test cryptographic parent hashing, promotion, rollback, and manifest integrity."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = Path(tmp_dir)
        mgr = LineageManager(seed=42, storage_dir=storage)

        model_v1 = TinyDummyModel()
        rec_v1 = mgr.register_root(model_v1, metrics={"init_acc": 0.5})

        assert rec_v1.version_id == "V1"
        assert rec_v1.status == "QUALIFIED"
        assert mgr.active_version_id == "V1"
        v1_hash = rec_v1.checkpoint_hash

        # Create Candidate V2
        model_v2 = TinyDummyModel()
        model_v2.fc.weight.data += 0.1  # mutate weights
        rec_cand_v2 = mgr.create_candidate(
            candidate_model=model_v2,
            episode_id="episode_a_arithmetic",
            domain="arithmetic",
            candidate_id="candidate_V2",
        )
        assert rec_cand_v2.parent_hash == v1_hash
        assert rec_cand_v2.checkpoint_hash != v1_hash
        assert rec_cand_v2.status == "CANDIDATE"

        # Promote Candidate V2 -> qualified_V2
        rec_qual_v2 = mgr.promote_candidate(
            candidate_id="candidate_V2",
            qualified_id="qualified_V2",
            metrics={"arithmetic_exact": 0.4},
            rationale="Passed 30% gate",
        )
        assert rec_qual_v2.status == "QUALIFIED"
        assert mgr.active_version_id == "qualified_V2"
        v2_hash = rec_qual_v2.checkpoint_hash

        # Create Candidate V3
        model_v3 = TinyDummyModel()
        model_v3.fc.weight.data += 0.2
        rec_cand_v3 = mgr.create_candidate(
            candidate_model=model_v3,
            episode_id="episode_b_logic",
            domain="logic",
            candidate_id="candidate_V3",
        )
        assert rec_cand_v3.parent_hash == v2_hash

        # Rollback Candidate V3
        rec_rb = mgr.rollback_candidate(
            candidate_id="candidate_V3",
            metrics={"logic_exact": 0.1},
            rationale="Failed logic gate",
        )
        assert rec_rb.status == "ROLLED_BACK"
        # Active version remains qualified_V2!
        assert mgr.active_version_id == "qualified_V2"
        assert mgr.get_active_record().checkpoint_hash == v2_hash

        # Export manifest and verify JSON serialization
        manifest_file = storage / "manifest.json"
        mgr.save_manifest_json(manifest_file)
        assert manifest_file.exists()


def test_strategy_selector_adaptive_selection() -> None:
    """Test adaptive strategy selection based on prior active skills."""
    selector = StrategySelector(mode="adaptive")

    # First episode: no prior skills
    d1 = selector.select_strategy(
        episode_id="episode_a",
        target_domain="arithmetic",
        diagnosed_error="col 1 carry failure",
        active_prior_skills=[],
    )
    assert d1.selected_strategy == LearningStrategy.FAILURE_TARGETED_WITH_REPLAY
    assert "elementary_carry" in d1.replay_domains_included

    # Second episode: with prior arithmetic skill
    d2 = selector.select_strategy(
        episode_id="episode_b",
        target_domain="logic",
        diagnosed_error="chained implication failure",
        active_prior_skills=["arithmetic"],
    )
    assert d2.selected_strategy == LearningStrategy.FAILURE_TARGETED_WITH_REPLAY
    assert "arithmetic" in d2.replay_domains_included
    assert "catastrophic forgetting" in d2.decision_rationale


def test_episode_builders_and_variable_disjointness() -> None:
    """Test that Episode specs are properly constructed with disjoint variable/prompt sets."""
    # Episode A: Arithmetic
    ep_a = build_m33_arithmetic_episode(seed=42)
    assert ep_a.episode_id == "episode_a_arithmetic"
    assert len(ep_a.practice_samples) > 0
    assert len(ep_a.fresh_test_samples) == 10
    for s in ep_a.fresh_test_samples:
        assert s.metadata["a"] > 0
        assert s.metadata["b"] > 0

    # Episode B: Logic
    ep_b = build_m33_logic_episode(seed=42)
    assert ep_b.episode_id == "episode_b_logic"
    assert len(ep_b.fresh_test_samples) == 8

    # Verify disjoint variables between train and test
    train_queries = {s.metadata.get("query_var") for s in ep_b.practice_samples}
    test_queries = {s.metadata.get("query_var") for s in ep_b.fresh_test_samples}
    assert train_queries.isdisjoint(test_queries), f"Leakage detected: {train_queries & test_queries}"

    # Episode C: Language
    ep_c = build_m33_language_episode(seed=42)
    assert ep_c.episode_id == "episode_c_language"
    assert len(ep_c.fresh_test_samples) == 10
    abstentions = [s for s in ep_c.fresh_test_samples if s.metadata.get("is_abstention", False)]
    assert len(abstentions) == 4
    for s in abstentions:
        assert s.expected_final_answer == "Insufficient information."
