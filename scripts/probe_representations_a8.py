"""Detached diagnostic frozen representation probes for Stage A.8 (Task 1).

Evaluates:
- Arm P1: Deployed Head
- Arm P2: Cross-validated Linear Ridge Probe on h_pool (128), z_concept (32), h_tokens_stat (256)
- Arm P3: Bounded Nonlinear MLP Probe (d_in -> 64 -> GELU -> 2, wd=1e-2, 300 steps AdamW)
- Arm P4: Lexical Controls (BoW and Word 1-3 gram linear) on the exact same 378 training set.

Strictly detached; never backpropagates to student or alters model state.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from annotate_frame_a2_diagnostics import signature
from run_attribute_learning import encode, evaluate, file_hash, scores, words
from run_stable_attributes import enrich

from adaptive_mini_gpt.data.probe_a8 import (
    coverage_matched_probe_indices,
    extract_question_token_bounds,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103
from adaptive_mini_gpt.training.learning_state import LearningState
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda

SPEC_PATH = Path("data/specs/probe_a8_v1.json")


def extract_frozen_representations(
    model: Any,
    batch: tuple[torch.Tensor, ...],
    rows: list[Any],
) -> tuple[dict[str, list[list[float]]], list[dict[str, Any]]]:
    """Extracts h_pool (128), z_concept (32), and h_tokens_stat (256) per example."""
    model.eval()
    representations: dict[str, list[list[float]]] = {
        "h_pool": [],
        "z_concept": [],
        "h_tokens_stat": [],
    }
    bounds_log: list[dict[str, Any]] = []

    with torch.no_grad():
        for start in range(0, len(rows), 32):
            tok = batch[0][start : start + 32]
            hidden = model.extract_sequence_hidden_states(tok)
            pooled, _ = model.pool_representations(hidden, tok)

            # Extract z_concept through low-rank projection
            if hasattr(model, "bottleneck") and model.bottleneck is not None:
                # First linear of concept_proj maps d_model (128) -> bottleneck_dim (32)
                first_linear = model.bottleneck.concept_proj[0]
                first_norm = model.bottleneck.concept_proj[1]
                z_concept = first_norm(first_linear(pooled))
            else:
                z_concept = pooled[:, :32]

            for j, row in enumerate(rows[start : start + 32]):
                tok_ids = tok[j].tolist()
                q_pos, a_pos = extract_question_token_bounds(tok_ids, row.question)
                bounds_log.append(dict(q_pos=q_pos, a_pos=a_pos))

                # Question hidden states for h_tokens_stat
                q_hidden = hidden[j, q_pos:a_pos]
                if q_hidden.shape[0] == 0:
                    q_hidden = hidden[j, q_pos : q_pos + 1]

                mean_pool = q_hidden.mean(dim=0)
                max_pool = q_hidden.max(dim=0)[0]
                h_stat = torch.cat([mean_pool, max_pool], dim=-1)

                representations["h_pool"].append(pooled[j].tolist())
                representations["z_concept"].append(z_concept[j].tolist())
                representations["h_tokens_stat"].append(h_stat.tolist())

    return representations, bounds_log


def solve_ridge(
    x: torch.Tensor,
    y: torch.Tensor,
    train_indices: list[int],
    regularization: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Dual ridge regression with training-only mean/std standardization."""
    x = x.double()
    y_target = 2 * y.double() - 1  # [-1, 1] signed targets
    train_ix = torch.tensor(train_indices, device=x.device, dtype=torch.long)

    mu = x[train_ix].mean(dim=0)
    sd = x[train_ix].std(dim=0, unbiased=False).clamp_min(1e-6)
    z = (x - mu) / sd

    center = y_target[train_ix].mean(dim=0)
    n = len(train_indices)
    k_mat = z[train_ix] @ z[train_ix].T + regularization * torch.eye(
        n, device=x.device, dtype=torch.float64
    )
    alpha = torch.linalg.solve(k_mat, y_target[train_ix] - center)
    coef = z[train_ix].T @ alpha
    predictions = z @ coef + center

    residual = float((k_mat @ alpha - (y_target[train_ix] - center)).abs().max())
    if not torch.isfinite(predictions).all() or residual > 1e-4:
        raise RuntimeError(
            f"Unreliable ridge solve with lambda={regularization}, residual={residual}"
        )

    fit_info = dict(
        lambda_=regularization,
        max_solve_residual=residual,
        features=int(x.shape[1]),
        training_cases=n,
    )
    return predictions.float(), fit_info


def cross_validated_ridge(
    x: torch.Tensor,
    y: torch.Tensor,
    train_indices: list[int],
    lambdas: list[float],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """5-fold cross-validation over lambda on probe train set."""
    n = len(train_indices)
    k = 5
    fold_size = n // k
    best_lambda = lambdas[0]
    best_val_acc = -1.0

    for lam in lambdas:
        val_accs = []
        for fold in range(k):
            val_split = train_indices[fold * fold_size : (fold + 1) * fold_size]
            tr_split = [i for i in train_indices if i not in set(val_split)]
            preds, _ = solve_ridge(x, y, tr_split, lam)
            val_ix = torch.tensor(val_split, device=x.device, dtype=torch.long)
            binary_preds = (preds[val_ix] > 0).long()
            acc = float((binary_preds == y[val_ix]).all(dim=-1).float().mean())
            val_accs.append(acc)
        mean_acc = sum(val_accs) / len(val_accs)
        if mean_acc > best_val_acc:
            best_val_acc = mean_acc
            best_lambda = lam

    # Final fit on entire train split with best lambda
    final_preds, fit_info = solve_ridge(x, y, train_indices, best_lambda)
    fit_info["cv_best_lambda"] = best_lambda
    fit_info["cv_val_accuracy"] = best_val_acc
    return final_preds, fit_info


class MLPProbe(nn.Module):
    """2-layer bounded nonlinear MLP probe (d_in -> 64 -> GELU -> 2)."""

    def __init__(self, d_in: int, d_hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_mlp_probe(
    x: torch.Tensor,
    y: torch.Tensor,
    train_indices: list[int],
    d_hidden: int = 64,
    steps: int = 300,
    lr: float = 0.001,
    weight_decay: float = 0.01,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Trains 2-layer bounded MLP probe on train_indices using AdamW."""
    train_ix = torch.tensor(train_indices, device=x.device, dtype=torch.long)
    mu = x[train_ix].mean(dim=0)
    sd = x[train_ix].std(dim=0, unbiased=False).clamp_min(1e-6)
    z = (x - mu) / sd

    model = MLPProbe(x.shape[1], d_hidden).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    target_float = y[train_ix].float()
    loss = torch.tensor(0.0, device=x.device)
    for _ in range(steps):
        optimizer.zero_grad()
        logits = model(z[train_ix])
        loss = F.binary_cross_entropy_with_logits(logits, target_float)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        all_logits = model(z)

    fit_info = dict(
        d_hidden=d_hidden,
        steps=steps,
        lr=lr,
        weight_decay=weight_decay,
        final_train_loss=float(loss.item()),
    )
    return all_logits, fit_info


def calculate_state_recall(
    predictions_binary: torch.Tensor,
    targets: torch.Tensor,
    rows: list[Any],
) -> dict[str, float]:
    """Calculates recall for positive, negated, and absent attribute states."""
    recalls = {}
    for state_name in ("positive", "negated", "absent"):
        correct_count = 0
        total_count = 0
        for i, row in enumerate(rows):
            for attr_idx, expected_status in enumerate(row.statuses):
                if expected_status == state_name:
                    total_count += 1
                    target_val = int(targets[i, attr_idx].item())
                    pred_val = int(predictions_binary[i, attr_idx].item())
                    if pred_val == target_val:
                        correct_count += 1
        recalls[state_name] = (correct_count / total_count) if total_count > 0 else 1.0
    return recalls


def run_probe_diagnosis(
    paths: dict[str, Path],
    training_rows: list[Any],
    shared_rows: dict[str, list[Any]],
    out_dir: Path,
) -> dict[str, Any]:
    """Runs Stage A.8 Frozen-Feature Representation Diagnosis on all target checkpoints."""
    require_remote_cuda()
    spec = json.loads(SPEC_PATH.read_text())

    # 1. Coverage-Matched Diagnostic Train Split
    train_ix, fam_val_ix = coverage_matched_probe_indices(training_rows, max_coverage=756)
    all_rows = training_rows[:756] + sum(shared_rows.values(), [])

    # Index partitions
    ranges: dict[str, list[int]] = {
        "probe_train": train_ix,
        "familiar_val": fam_val_ix,
    }
    offset = 756
    for group_name, r_list in shared_rows.items():
        ranges[group_name] = list(range(offset, offset + len(r_list)))
        offset += len(r_list)

    y_tensor = torch.tensor([r.labels for r in all_rows], device="cuda", dtype=torch.long)
    tokenizer = ByteTextTokenizerM103()
    context_length = 256

    result: dict[str, Any] = {
        "scope": "Stage A.8 Frozen-Feature Representation Diagnosis",
        "probe_train_cases": len(train_ix),
        "familiar_val_cases": len(fam_val_ix),
        "models": {},
    }

    # Lexical Controls (Arm P4) on all_rows
    bow_features: list[list[float]] = []
    ngram_features: list[list[float]] = []
    bow_vocab = sorted({w for i in train_ix for w in words(all_rows[i], "bow_linear")})
    ngram_vocab = sorted({w for i in train_ix for w in words(all_rows[i], "ngram_linear")})

    for r in all_rows:
        bow_counts = Counter(words(r, "bow_linear"))
        bow_features.append([float(bow_counts[w]) for w in bow_vocab])
        ngram_counts = Counter(words(r, "ngram_linear"))
        ngram_features.append([float(ngram_counts[w]) for w in ngram_vocab])

    bow_tensor = torch.tensor(bow_features, device="cuda", dtype=torch.float32)
    ngram_tensor = torch.tensor(ngram_features, device="cuda", dtype=torch.float32)

    lexical_probes: dict[str, Any] = {}
    for name, feat in (("bow_linear", bow_tensor), ("ngram_linear", ngram_tensor)):
        preds, fit = cross_validated_ridge(feat, y_tensor, train_ix, spec["ridge_lambdas"])
        binary_preds = (preds > 0).long()
        lexical_probes[name] = {
            "fit": fit,
            "metrics": {
                grp: {
                    **enrich(scores(preds[ix], y_tensor[ix], [all_rows[i] for i in ix])),
                    "state_recall": calculate_state_recall(
                        binary_preds[ix], y_tensor[ix], [all_rows[i] for i in ix]
                    ),
                }
                for grp, ix in ranges.items()
            },
        }

    for model_name, ckpt_path in paths.items():
        if not ckpt_path.exists():
            continue
        source_hash = file_hash(ckpt_path)
        owner = LearningState.load(ckpt_path)
        snap = owner.snapshot()
        before_sig = signature(owner)

        batch, _ = encode(all_rows, tokenizer, context_length)
        features, bounds = extract_frozen_representations(owner.model, batch, all_rows)
        deployed_eval = evaluate(owner.model, batch, all_rows)

        model_report: dict[str, Any] = {
            "checkpoint_sha256": source_hash,
            "deployed_predictions": {},
            "probes": {},
        }

        # Arm P1: Deployed Head
        dep_logits = torch.tensor(deployed_eval["logits"], device="cuda")
        dep_binary = (dep_logits > 0).long()
        for grp, ix in ranges.items():
            model_report["deployed_predictions"][grp] = {
                **enrich(scores(dep_logits[ix], y_tensor[ix], [all_rows[i] for i in ix])),
                "state_recall": calculate_state_recall(
                    dep_binary[ix], y_tensor[ix], [all_rows[i] for i in ix]
                ),
            }

        # Arm P2: Cross-Validated Ridge Probe on h_pool, z_concept, h_tokens_stat
        for rep_name in ("h_pool", "z_concept", "h_tokens_stat"):
            rep_tensor = torch.tensor(features[rep_name], device="cuda", dtype=torch.float32)
            preds, fit = cross_validated_ridge(
                rep_tensor, y_tensor, train_ix, spec["ridge_lambdas"]
            )
            binary_preds = (preds > 0).long()
            model_report["probes"][f"ridge_{rep_name}"] = {
                "fit": fit,
                "metrics": {
                    grp: {
                        **enrich(scores(preds[ix], y_tensor[ix], [all_rows[i] for i in ix])),
                        "state_recall": calculate_state_recall(
                            binary_preds[ix], y_tensor[ix], [all_rows[i] for i in ix]
                        ),
                    }
                    for grp, ix in ranges.items()
                },
            }

        # Arm P3: Bounded Nonlinear MLP Probe on h_pool and h_tokens_stat
        for rep_name in ("h_pool", "h_tokens_stat"):
            rep_tensor = torch.tensor(features[rep_name], device="cuda", dtype=torch.float32)
            mlp_preds, mlp_fit = train_mlp_probe(
                rep_tensor,
                y_tensor,
                train_ix,
                d_hidden=spec["mlp"]["d_hidden"],
                steps=spec["mlp"]["steps"],
                lr=spec["mlp"]["lr"],
                weight_decay=spec["mlp"]["weight_decay"],
            )
            binary_preds = (mlp_preds > 0).long()
            model_report["probes"][f"mlp_{rep_name}"] = {
                "fit": mlp_fit,
                "metrics": {
                    grp: {
                        **enrich(scores(mlp_preds[ix], y_tensor[ix], [all_rows[i] for i in ix])),
                        "state_recall": calculate_state_recall(
                            binary_preds[ix], y_tensor[ix], [all_rows[i] for i in ix]
                        ),
                    }
                    for grp, ix in ranges.items()
                },
            }

        # Copy lexical controls into model report
        model_report["probes"]["bow_linear"] = lexical_probes["bow_linear"]
        model_report["probes"]["ngram_linear"] = lexical_probes["ngram_linear"]

        # Restore and verify immutability
        owner.restore(snap)
        if signature(owner) != before_sig or file_hash(ckpt_path) != source_hash:
            raise RuntimeError(f"Probe diagnosis altered checkpoint {ckpt_path}")

        result["models"][model_name] = model_report

    (out_dir / "probe_results_a8.json").write_text(json.dumps(result, indent=2))
    return result
