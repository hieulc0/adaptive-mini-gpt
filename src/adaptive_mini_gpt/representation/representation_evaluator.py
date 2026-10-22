"""Phase 5: Independent Representation Evaluation.

Measures whether latent representations encode graph structure:
  - Schema Classification (CHAIN vs TREE vs DIAMOND vs FORK).
  - Connected vs Disconnected Classification.
  - Directional Relation Discrimination (Forward vs Reverse).
  - Pairwise Embedding Separation (Cosine Similarity of congruent vs incongruent pairs).
  - Comparison against Non-Neural Structural Heuristic Baselines (Entity Jaccard / Path length).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.data.bpe_scaling import BPETokenizer
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.representation.structural_benchmark import RelationalGraphDataset


@dataclass
class RepresentationMetrics:
    topology_classification_accuracy: float
    connectivity_classification_accuracy: float
    direction_discrimination_accuracy: float
    mean_positive_pair_cosine_similarity: float
    mean_negative_pair_cosine_similarity: float
    embedding_separation_margin: float  # (positive_sim - negative_sim)
    non_neural_heuristic_accuracy: float  # Jaccard overlap heuristic reference

    def to_dict(self) -> dict[str, Any]:
        return {
            "topology_classification_accuracy": round(self.topology_classification_accuracy, 4),
            "connectivity_classification_accuracy": round(self.connectivity_classification_accuracy, 4),
            "direction_discrimination_accuracy": round(self.direction_discrimination_accuracy, 4),
            "mean_positive_pair_cosine_similarity": round(self.mean_positive_pair_cosine_similarity, 4),
            "mean_negative_pair_cosine_similarity": round(self.mean_negative_pair_cosine_similarity, 4),
            "embedding_separation_margin": round(self.embedding_separation_margin, 4),
            "non_neural_heuristic_accuracy": round(self.non_neural_heuristic_accuracy, 4),
        }


class IndependentRepresentationEvaluator:
    """Evaluates geometric and semantic properties of learned neural hidden states."""

    TOPOLOGIES = ["CHAIN", "TREE", "DIAMOND", "FORK"]
    TOPOLOGY_TO_IDX = {name: i for i, name in enumerate(TOPOLOGIES)}

    def __init__(
        self,
        model: MiniGPT,
        tokenizer: BPETokenizer,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def extract_representation(self, premises: list[str], question: str) -> torch.Tensor:
        """Extracts the final hidden state vector for a given prompt."""
        prompt = f"[PREMISES] {' '.join(premises)} [QUESTION] {question} [ANSWER]"
        ids = self.tokenizer.encode(prompt)
        if len(ids) > self.model.config.context_length:
            ids = ids[-self.model.config.context_length:]
        inp = torch.tensor([ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            b, t = inp.size()
            tok_emb = self.model.wte(inp)
            pos = torch.arange(0, t, dtype=torch.long, device=inp.device)
            pos_emb = self.model.wpe(pos)
            x = self.model.drop(tok_emb + pos_emb)
            for block in self.model.blocks:
                x, _ = block(x)
            hidden = self.model.ln_f(x)
            # Last token representation
            return hidden[0, -1, :].detach().cpu()

    def evaluate_representation_quality(
        self,
        worlds: list[RelationalGraphDataset],
    ) -> RepresentationMetrics:
        """Evaluates representation geometry and graph property encoding across worlds."""
        # 1. Topology Classification (Chain vs Tree vs Diamond vs Fork)
        topo_embeddings: list[torch.Tensor] = []
        topo_labels: list[int] = []

        for w in worlds:
            if w.topology in self.TOPOLOGY_TO_IDX:
                for t in w.held_out_test_pool[:6]:
                    emb = self.extract_representation(t.premises, t.question)
                    topo_embeddings.append(emb)
                    topo_labels.append(self.TOPOLOGY_TO_IDX[w.topology])

        topo_acc = 0.25  # Default chance
        if len(topo_embeddings) >= 8 and len(set(topo_labels)) >= 2:
            X = torch.stack(topo_embeddings)
            y = torch.tensor(topo_labels, dtype=torch.long)
            # Nearest centroid classifier on half-split
            split_idx = len(X) // 2
            train_X, test_X = X[:split_idx], X[split_idx:]
            train_y, test_y = y[:split_idx], y[split_idx:]

            centroids = {}
            for label in set(train_y.tolist()):
                centroids[label] = train_X[train_y == label].mean(dim=0)

            correct = 0
            for i in range(len(test_X)):
                pred_label = max(
                    centroids.keys(),
                    key=lambda l: F.cosine_similarity(test_X[i : i + 1], centroids[l].unsqueeze(0)).item(),
                )
                if pred_label == test_y[i].item():
                    correct += 1
            topo_acc = correct / max(1, len(test_X))

        # 2. Connectivity & Directional Discrimination & Pairwise Separation
        pos_sims: list[float] = []
        neg_sims: list[float] = []
        conn_correct = 0
        conn_total = 0
        dir_correct = 0
        dir_total = 0
        heuristic_correct = 0
        heuristic_total = 0

        for w in worlds:
            yes_tasks = [t for t in w.held_out_test_pool if t.expected_verdict == "YES"]
            no_tasks = [t for t in w.held_out_test_pool if t.expected_verdict == "NO"]
            unk_tasks = [t for t in w.held_out_test_pool if t.expected_verdict == "UNKNOWN"]

            # Compute pairwise cosine similarities
            if yes_tasks and unk_tasks:
                yes_emb_1 = self.extract_representation(yes_tasks[0].premises, yes_tasks[0].question)
                yes_emb_2 = self.extract_representation(
                    yes_tasks[min(1, len(yes_tasks) - 1)].premises,
                    yes_tasks[min(1, len(yes_tasks) - 1)].question,
                )
                pos_sim = F.cosine_similarity(yes_emb_1.unsqueeze(0), yes_emb_2.unsqueeze(0)).item()
                pos_sims.append(pos_sim)

                unk_emb = self.extract_representation(unk_tasks[0].premises, unk_tasks[0].question)
                neg_sim = F.cosine_similarity(yes_emb_1.unsqueeze(0), unk_emb.unsqueeze(0)).item()
                neg_sims.append(neg_sim)

                # Connectivity separation: connected pairs should have higher similarity than ungrounded
                if pos_sim > neg_sim:
                    conn_correct += 1
                conn_total += 1

            if yes_tasks and no_tasks:
                y_emb = self.extract_representation(yes_tasks[0].premises, yes_tasks[0].question)
                n_emb = self.extract_representation(no_tasks[0].premises, no_tasks[0].question)
                # Direction discrimination: distance between forward and reverse queries
                dir_sim = F.cosine_similarity(y_emb.unsqueeze(0), n_emb.unsqueeze(0)).item()
                if dir_sim < 0.95:  # Representations distinguish direction
                    dir_correct += 1
                dir_total += 1

            # Non-neural Jaccard heuristic reference:
            for t in w.held_out_test_pool:
                ents = t.entities_involved
                if len(ents) >= 2:
                    u, v = ents[0], ents[1]
                    found_fwd = any(f"{u} {w.semantics.name} {v}" in p for p in t.premises)
                    found_rev = any(f"{v} {w.semantics.name} {u}" in p for p in t.premises)
                    if found_fwd:
                        h_pred = "YES"
                    elif found_rev and not w.semantics.is_symmetric:
                        h_pred = "NO"
                    else:
                        h_pred = "UNKNOWN"
                    if h_pred == t.expected_verdict:
                        heuristic_correct += 1
                    heuristic_total += 1

        mean_pos = float(np.mean(pos_sims)) if pos_sims else 0.0
        mean_neg = float(np.mean(neg_sims)) if neg_sims else 0.0
        margin = mean_pos - mean_neg
        conn_acc = conn_correct / max(1, conn_total)
        dir_acc = dir_correct / max(1, dir_total)
        h_acc = heuristic_correct / max(1, heuristic_total)

        return RepresentationMetrics(
            topology_classification_accuracy=topo_acc,
            connectivity_classification_accuracy=conn_acc,
            direction_discrimination_accuracy=dir_acc,
            mean_positive_pair_cosine_similarity=mean_pos,
            mean_negative_pair_cosine_similarity=mean_neg,
            embedding_separation_margin=margin,
            non_neural_heuristic_accuracy=h_acc,
        )
