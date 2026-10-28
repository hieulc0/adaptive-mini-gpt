"""Detached diagnostic ridge decoding only. Never substitutes for student inference."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any

import torch
from annotate_frame_a2_diagnostics import signature
from run_attribute_learning import encode, evaluate, file_hash, scores, words
from run_stable_attributes import enrich

from adaptive_mini_gpt.data.exposure_a7 import probe_training_indices
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103
from adaptive_mini_gpt.training.learning_state import LearningState
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda


def feature_trace(model, batch, rows):
    model.eval()
    output = {
        k: []
        for k in (
            "embedding",
            "embedding_position",
            "question_mean",
            "content_DIAGNOSTIC",
            "question_last",
            "concept",
        )
    }
    boundaries = []
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            tok = batch[0][start : start + 32]
            base = model.student.base_model
            emb = base.wte(tok)
            pos = base.wpe(torch.arange(tok.shape[1], device="cuda"))
            hidden = model.extract_sequence_hidden_states(tok)
            pooled, _ = model.pool_representations(hidden, tok)
            concept = model.bottleneck.concept_proj(pooled)
            for j, row in enumerate(rows[start : start + 32]):
                ids = tok[j].tolist()
                q, a = ids.index(261), ids.index(262)
                matches = list(
                    re.finditer(r"\b(?:(?:not|without) )?(?:flammable|conductive)\b", row.question)
                )
                char_start, char_end = (
                    (matches[0].start(), matches[-1].end())
                    if matches
                    else (row.question.index("present"), row.question.index("present") + 7)
                )
                left = q + 1 + len(row.question[:char_start].encode())
                right = q + 1 + len(row.question[:char_end].encode())
                boundaries.append(dict(pool=[q, a], content=[left, right]))
                vectors = dict(
                    embedding=emb[j, q:a].mean(0),
                    embedding_position=(emb[j, q:a] + pos[q:a]).mean(0),
                    question_mean=pooled[j],
                    content_DIAGNOSTIC=hidden[j, left:right].mean(0),
                    question_last=hidden[j, a - 1],
                    concept=concept[j],
                )
                for key, v in vectors.items():
                    output[key].append(v.tolist())
    return output, boundaries


def ridge(x, truth, train, regularization):
    """Training-only centering/scaling; dual solve avoids dimension-dependent optimization."""
    x = x.double()
    y = 2 * truth.double() - 1
    mu = x[train].mean(0)
    sd = x[train].std(0, unbiased=False).clamp_min(1e-6)
    z = (x - mu) / sd
    center = y[train].mean(0)
    a = z[train] @ z[train].T + regularization * torch.eye(
        len(train), device="cuda", dtype=torch.float64
    )
    alpha = torch.linalg.solve(a, y[train] - center)
    coef = z[train].T @ alpha
    prediction = z @ coef + center
    if not torch.isfinite(prediction).all():
        raise RuntimeError("Nonfinite diagnostic probe")
    residual = float((a @ alpha - (y[train] - center)).abs().max())
    if residual > 1e-6:
        raise RuntimeError("Unreliable diagnostic ridge solve")
    return prediction, dict(
        max_solve_residual=residual,
        features=x.shape[1],
        fitted_coefficients=2 * (x.shape[1] + 1),
        training_cases=len(train),
        lambda_=regularization,
        feature_mean=mu.tolist(),
        feature_scale=sd.tolist(),
        coefficients=coef.tolist(),
        intercept=center.tolist(),
    )


def run(paths, training, shared, out):
    require_remote_cuda()
    train = probe_training_indices(training)
    remaining = [i for i in range(len(training)) if i not in set(train)]
    rows = training + sum(shared.values(), [])
    ranges = dict(probe_training=train, student_seen_probe_unseen=remaining)
    offset = len(training)
    for key, rs in shared.items():
        ranges[key] = list(range(offset, offset + len(rs)))
        offset += len(rs)
    y = torch.tensor([r.labels for r in rows], device="cuda")
    result: dict[str, Any] = dict(
        scope="Frozen-weight supervised decodability; not deployed learner competence",
        training_indices=train,
        student_updates=0,
        models={},
    )
    for name, path in paths.items():
        source_hash = file_hash(path)
        owner = LearningState.load(path)
        snap = owner.snapshot()
        before = signature(owner)
        batch, tokens = encode(
            rows, ByteTextTokenizerM103(), owner.model.student.base_model.config.context_length
        )
        features, bounds = feature_trace(owner.model, batch, rows)
        deployed = evaluate(owner.model, batch, rows)
        model_result: dict[str, Any] = dict(
            checkpoint_sha256=source_hash,
            coverage_support_previously_seen_by_student=(name != "parent"),
            rows=len(rows),
            feature_hidden_traversals=math.ceil(len(rows) / 32),
            deployed_forward_batches=math.ceil(len(rows) / 32),
            deployed={
                k: enrich(
                    scores(
                        torch.tensor(deployed["logits"], device="cuda")[ix],
                        y[ix],
                        [rows[i] for i in ix],
                    )
                )
                for k, ix in ranges.items()
            },
            probes={},
        )
        for kind in ("bow_linear", "ngram_linear"):
            vocab = sorted({w for i in train for w in words(rows[i], kind)})
            counts = [Counter(words(row, kind)) for row in rows]
            features[kind] = [[float(c[w]) for w in vocab] for c in counts]
        for kind, values in features.items():
            predictions, fit = ridge(torch.tensor(values, device="cuda"), y, train, 1.0)
            model_result["probes"][kind] = dict(
                fit=fit,
                metrics={
                    k: enrich(scores(predictions[ix], y[ix], [rows[i] for i in ix]))
                    for k, ix in ranges.items()
                },
            )
        # Never select a replacement solely on content-span/oracle decoding.
        candidates = {}
        for kind in ("question_mean", "question_last"):
            metrics = model_result["probes"][kind]["metrics"]
            supported = (
                metrics["probe_training"]["exact"] >= 0.99
                and metrics["student_seen_probe_unseen"]["exact"] >= 0.9
            )
            for group in ("development_familiar", "development_novel"):
                m = metrics[group]
                supported = (
                    supported
                    and m["exact"] >= 0.9
                    and min(m["family_exact"].values()) >= 0.85
                    and min(m["status_accuracy"].values()) >= 0.95
                    and m["exact"] - model_result["deployed"][group]["exact"] >= 0.2
                )
            candidates[kind] = bool(supported)
        model_result["positive_readout_diagnosis"] = candidates
        (out / f"probe_features_{name}.json").write_text(
            json.dumps(dict(features=features, bounds=bounds, tokens=tokens))
        )
        owner.restore(snap)
        if signature(owner) != before or file_hash(path) != source_hash:
            raise RuntimeError("Diagnostic altered owner state")
        model_result["unchanged_signature"] = before
        result["models"][name] = model_result
        (out / "probes.json").write_text(json.dumps(result, indent=2))
    return result
