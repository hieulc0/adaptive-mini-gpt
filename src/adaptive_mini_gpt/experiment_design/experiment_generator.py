"""Phase 3: Autonomous Experiment Generation and Controlled Selection Arms for M50."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import random
import time
import torch
import torch.nn as nn

from adaptive_mini_gpt.experiment_design.expanded_hypothesis import AutonomousEnvironment, EnvironmentQuery
from adaptive_mini_gpt.experiment_design.hypothesis_revision import HypothesisRevisionTracker
from adaptive_mini_gpt.experiment_design.syntax_validator import ExperimentSyntaxValidator, ExperimentValidatorStats
from adaptive_mini_gpt.model.transformer import MiniGPT


@dataclass
class GeneratedExperimentResult:
    """Outcome of an experiment selection/generation run."""
    arm_name: str
    query_budget: int
    selected_queries: list[tuple[str, str]]  # list of (source, target)
    observed_answers: list[str]
    cumulative_information_gain: float
    hypothesis_identified: bool
    queries_to_identify: int
    final_top_hypothesis: str
    final_hypothesis_confidence: float
    oracle_call_count: int
    duration_seconds: float
    validator_stats: ExperimentValidatorStats
    history: list[dict[str, Any]] = field(default_factory=list)


class AutonomousExperimentGenerator(nn.Module):
    """Neural policy network proposing candidate experiments from entity tokens."""

    def __init__(self, d_model: int = 64, vocab_size: int = 1024) -> None:
        super().__init__()
        self.d_model = d_model
        self.entity_encoder = nn.Embedding(vocab_size, d_model)
        self.proposal_scorer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def score_entity_pair(self, src_idx: int, tgt_idx: int) -> float:
        device = self.entity_encoder.weight.device
        s_tensor = torch.tensor([src_idx], dtype=torch.long, device=device)
        t_tensor = torch.tensor([tgt_idx], dtype=torch.long, device=device)
        s_emb = self.entity_encoder(s_tensor)
        t_emb = self.entity_encoder(t_tensor)
        pair_emb = torch.cat([s_emb, t_emb], dim=-1)
        score = self.proposal_scorer(pair_emb).item()
        return float(score)


class ExperimentGenerationRunner:
    """Executes Arms A, B, C, D across autonomous environments."""

    def __init__(
        self,
        model: MiniGPT,
        tokenizer: Any,
        device: torch.device,
        seed: int = 42,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.seed = seed
        self.rng = random.Random(seed)

        # Neural experiment generator
        cfg = getattr(model, "config", getattr(model, "cfg", None))
        d_model = getattr(cfg, "d_model", 64)
        vocab_size = getattr(cfg, "vocab_size", 1024)
        self.neural_generator = AutonomousExperimentGenerator(d_model, vocab_size).to(device)

    def run_arm_a_random_generation(
        self,
        env: AutonomousEnvironment,
        query_budget: int = 4,
    ) -> GeneratedExperimentResult:
        """Arm A: Uniform random valid experiment generation."""
        t0 = time.perf_counter()
        env.reset_oracle_budget()
        validator = ExperimentSyntaxValidator(env.entities, env.relation_name)
        tracker = HypothesisRevisionTracker(
            env.base_edges, env.entity_levels, env.blocked_nodes, env.negated_edges
        )

        selected_queries: list[tuple[str, str]] = []
        observed_answers: list[str] = []
        history: list[dict[str, Any]] = []
        queries_to_id = query_budget

        while len(selected_queries) < query_budget:
            # Generate random pair
            u = self.rng.choice(env.entities)
            v = self.rng.choice(env.entities)

            val = validator.validate_experiment(u, env.relation_name, v)
            if not val.is_valid:
                continue

            # Valid query -> Ask oracle
            ans = env.ask_oracle(u, v)
            gain = tracker.update_belief(u, v, ans, env.true_hypothesis.hypothesis_id)
            selected_queries.append((u, v))
            observed_answers.append(ans)

            top_h, conf = tracker.get_top_hypothesis()
            if top_h.hypothesis_id == env.true_hypothesis.hypothesis_id and conf > 0.90:
                if queries_to_id == query_budget:
                    queries_to_id = len(selected_queries)

            history.append({
                "source": u, "target": v, "answer": ans, "info_gain": gain,
                "top_hyp": top_h.hypothesis_id, "confidence": conf,
            })

        dt = time.perf_counter() - t0
        top_h, conf = tracker.get_top_hypothesis()
        is_id = (top_h.hypothesis_id == env.true_hypothesis.hypothesis_id and conf > 0.90)

        return GeneratedExperimentResult(
            arm_name="ARM_A_RANDOM_GENERATION",
            query_budget=query_budget,
            selected_queries=selected_queries,
            observed_answers=observed_answers,
            cumulative_information_gain=sum(h["info_gain"] for h in history),
            hypothesis_identified=is_id,
            queries_to_identify=queries_to_id if is_id else query_budget,
            final_top_hypothesis=top_h.hypothesis_id,
            final_hypothesis_confidence=conf,
            oracle_call_count=env.oracle_call_count,
            duration_seconds=dt,
            validator_stats=validator.stats,
            history=history,
        )

    def run_arm_b_info_gain_selection(
        self,
        env: AutonomousEnvironment,
        query_budget: int = 4,
    ) -> GeneratedExperimentResult:
        """Arm B: Deterministic greedy Expected Information Gain (EIG) selection."""
        t0 = time.perf_counter()
        env.reset_oracle_budget()
        validator = ExperimentSyntaxValidator(env.entities, env.relation_name)
        tracker = HypothesisRevisionTracker(
            env.base_edges, env.entity_levels, env.blocked_nodes, env.negated_edges
        )

        selected_queries: list[tuple[str, str]] = []
        observed_answers: list[str] = []
        history: list[dict[str, Any]] = []
        queries_to_id = query_budget

        while len(selected_queries) < query_budget:
            # Enumerate all valid untried pairs and find max EIG
            best_pair = None
            best_eig = -1.0

            for u in env.entities:
                for v in env.entities:
                    if u == v or (u, v) in validator.seen_queries:
                        continue
                    eig = tracker.compute_expected_information_gain(u, v)
                    if eig > best_eig:
                        best_eig = eig
                        best_pair = (u, v)

            if best_pair is None:
                # Fallback to any remaining pair
                for u in env.entities:
                    for v in env.entities:
                        if u != v and (u, v) not in validator.seen_queries:
                            best_pair = (u, v)
                            break
                    if best_pair:
                        break

            if best_pair is None:
                break

            u, v = best_pair
            val = validator.validate_experiment(u, env.relation_name, v)
            ans = env.ask_oracle(u, v)
            gain = tracker.update_belief(u, v, ans, env.true_hypothesis.hypothesis_id)
            selected_queries.append((u, v))
            observed_answers.append(ans)

            top_h, conf = tracker.get_top_hypothesis()
            if top_h.hypothesis_id == env.true_hypothesis.hypothesis_id and conf > 0.90:
                if queries_to_id == query_budget:
                    queries_to_id = len(selected_queries)

            history.append({
                "source": u, "target": v, "answer": ans, "info_gain": gain,
                "top_hyp": top_h.hypothesis_id, "confidence": conf,
            })

        dt = time.perf_counter() - t0
        top_h, conf = tracker.get_top_hypothesis()
        is_id = (top_h.hypothesis_id == env.true_hypothesis.hypothesis_id and conf > 0.90)

        return GeneratedExperimentResult(
            arm_name="ARM_B_DETERMINISTIC_INFO_GAIN",
            query_budget=query_budget,
            selected_queries=selected_queries,
            observed_answers=observed_answers,
            cumulative_information_gain=sum(h["info_gain"] for h in history),
            hypothesis_identified=is_id,
            queries_to_identify=queries_to_id if is_id else query_budget,
            final_top_hypothesis=top_h.hypothesis_id,
            final_hypothesis_confidence=conf,
            oracle_call_count=env.oracle_call_count,
            duration_seconds=dt,
            validator_stats=validator.stats,
            history=history,
        )

    def run_arm_c_neural_selection(
        self,
        env: AutonomousEnvironment,
        query_budget: int = 4,
    ) -> GeneratedExperimentResult:
        """Arm C: Neural query policy scoring candidates."""
        t0 = time.perf_counter()
        env.reset_oracle_budget()
        validator = ExperimentSyntaxValidator(env.entities, env.relation_name)
        tracker = HypothesisRevisionTracker(
            env.base_edges, env.entity_levels, env.blocked_nodes, env.negated_edges
        )

        selected_queries: list[tuple[str, str]] = []
        observed_answers: list[str] = []
        history: list[dict[str, Any]] = []
        queries_to_id = query_budget

        # Candidate pool: pairwise Cartesian product
        candidates = []
        for u in env.entities:
            for v in env.entities:
                if u != v:
                    candidates.append((u, v))

        while len(selected_queries) < query_budget and candidates:
            # Score untried candidates
            scored = []
            for u, v in candidates:
                if (u, v) in validator.seen_queries:
                    continue
                # Neural score based on structural probe heuristic + neural embed
                num_emb = self.neural_generator.entity_encoder.num_embeddings
                u_tok = hash(u) % num_emb
                v_tok = hash(v) % num_emb
                score = self.neural_generator.score_entity_pair(u_tok, v_tok)
                # Boost structural reverse and multi-hop pairs
                if (v, u) in env.base_edges:
                    score += 1.5  # Reverse probe
                scored.append((score, (u, v)))

            if not scored:
                break

            scored.sort(key=lambda x: x[0], reverse=True)
            best_pair = scored[0][1]
            u, v = best_pair

            val = validator.validate_experiment(u, env.relation_name, v)
            ans = env.ask_oracle(u, v)
            gain = tracker.update_belief(u, v, ans, env.true_hypothesis.hypothesis_id)
            selected_queries.append((u, v))
            observed_answers.append(ans)

            top_h, conf = tracker.get_top_hypothesis()
            if top_h.hypothesis_id == env.true_hypothesis.hypothesis_id and conf > 0.90:
                if queries_to_id == query_budget:
                    queries_to_id = len(selected_queries)

            history.append({
                "source": u, "target": v, "answer": ans, "info_gain": gain,
                "top_hyp": top_h.hypothesis_id, "confidence": conf,
            })

        dt = time.perf_counter() - t0
        top_h, conf = tracker.get_top_hypothesis()
        is_id = (top_h.hypothesis_id == env.true_hypothesis.hypothesis_id and conf > 0.90)

        return GeneratedExperimentResult(
            arm_name="ARM_C_NEURAL_SELECTION",
            query_budget=query_budget,
            selected_queries=selected_queries,
            observed_answers=observed_answers,
            cumulative_information_gain=sum(h["info_gain"] for h in history),
            hypothesis_identified=is_id,
            queries_to_identify=queries_to_id if is_id else query_budget,
            final_top_hypothesis=top_h.hypothesis_id,
            final_hypothesis_confidence=conf,
            oracle_call_count=env.oracle_call_count,
            duration_seconds=dt,
            validator_stats=validator.stats,
            history=history,
        )

    def run_arm_d_autonomous_generation(
        self,
        env: AutonomousEnvironment,
        query_budget: int = 4,
    ) -> GeneratedExperimentResult:
        """Arm D: Autonomous Neural Experiment Generation with syntax and safety validation."""
        t0 = time.perf_counter()
        env.reset_oracle_budget()
        validator = ExperimentSyntaxValidator(env.entities, env.relation_name)
        tracker = HypothesisRevisionTracker(
            env.base_edges, env.entity_levels, env.blocked_nodes, env.negated_edges
        )

        selected_queries: list[tuple[str, str]] = []
        observed_answers: list[str] = []
        history: list[dict[str, Any]] = []
        queries_to_id = query_budget

        max_attempts = query_budget * 10
        attempts = 0

        while len(selected_queries) < query_budget and attempts < max_attempts:
            attempts += 1

            # Propose candidate experiment dynamically:
            # Sample source entity weighted by connectivity
            src = self.rng.choice(env.entities)
            
            # Neural candidate proposal for target
            tgt_scores = []
            for tgt in env.entities:
                if tgt == src:
                    continue
                num_emb = self.neural_generator.entity_encoder.num_embeddings
                s_tok = hash(src) % num_emb
                t_tok = hash(tgt) % num_emb
                score = self.neural_generator.score_entity_pair(s_tok, t_tok)
                # If reverse of an existing edge, prioritize
                if (tgt, src) in env.base_edges:
                    score += 2.0
                tgt_scores.append((score, tgt))

            tgt_scores.sort(key=lambda x: x[0], reverse=True)
            chosen_tgt = tgt_scores[0][1] if tgt_scores else self.rng.choice(env.entities)

            # Validate syntax, safety, and non-redundancy
            val = validator.validate_experiment(src, env.relation_name, chosen_tgt)
            if not val.is_valid:
                continue

            # Query oracle
            ans = env.ask_oracle(src, chosen_tgt)
            gain = tracker.update_belief(src, chosen_tgt, ans, env.true_hypothesis.hypothesis_id)
            selected_queries.append((src, chosen_tgt))
            observed_answers.append(ans)

            top_h, conf = tracker.get_top_hypothesis()
            if top_h.hypothesis_id == env.true_hypothesis.hypothesis_id and conf > 0.90:
                if queries_to_id == query_budget:
                    queries_to_id = len(selected_queries)

            history.append({
                "source": src, "target": chosen_tgt, "answer": ans, "info_gain": gain,
                "top_hyp": top_h.hypothesis_id, "confidence": conf,
            })

        dt = time.perf_counter() - t0
        top_h, conf = tracker.get_top_hypothesis()
        is_id = (top_h.hypothesis_id == env.true_hypothesis.hypothesis_id and conf > 0.90)

        return GeneratedExperimentResult(
            arm_name="ARM_D_AUTONOMOUS_GENERATION",
            query_budget=query_budget,
            selected_queries=selected_queries,
            observed_answers=observed_answers,
            cumulative_information_gain=sum(h["info_gain"] for h in history),
            hypothesis_identified=is_id,
            queries_to_identify=queries_to_id if is_id else query_budget,
            final_top_hypothesis=top_h.hypothesis_id,
            final_hypothesis_confidence=conf,
            oracle_call_count=env.oracle_call_count,
            duration_seconds=dt,
            validator_stats=validator.stats,
            history=history,
        )
