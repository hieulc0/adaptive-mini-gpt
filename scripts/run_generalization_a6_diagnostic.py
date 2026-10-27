"""REMOTE ONLY: frozen A6.1 localization; deliberately contains no optimizer updates."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from annotate_frame_a2_diagnostics import signature
from frame_a2_common import ROOT_HASH
from run_a4_localization import ensure_compatible, hybrid, trace
from run_attribute_learning import encode, evaluate, file_hash, scores
from run_rehearsal_a5 import prepare
from run_stable_attributes import enrich

from adaptive_mini_gpt.data.attribute_learning import TAG_IDS
from adaptive_mini_gpt.data.generalization_a6 import diagnostic, matched_indices
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103
from adaptive_mini_gpt.training.learning_state import (
    LearningState,
    deterministic_numerical_policy,
    require_numerical_policy,
)
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda
from adaptive_mini_gpt.training.reproducibility import seed_all

SPEC = Path("data/specs/generalization_a6_diagnostic_v1.json")


def subset_metrics(logits, batch, rows):
    groups = {"all": list(range(len(rows)))}
    for attr in ("name", "frame", "prefix"):
        for val in sorted({getattr(r, attr) for r in rows}):
            groups[f"{attr}={val!r}"] = [i for i, r in enumerate(rows) if getattr(r, attr) == val]
    return {
        k: enrich(scores(logits[ix], batch[2][ix][:, TAG_IDS], [rows[i] for i in ix]))
        for k, ix in groups.items()
    }


def hidden_diagnostics(model, batch, rows):
    t = trace(model, batch)
    content = []
    alternatives: dict[str, list] = {k: [] for k in ("content_mean", "question_last", "no_marker")}
    bounds = []
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            tokens = batch[0][start : start + 32]
            h = model.extract_sequence_hidden_states(tokens)
            for j, row in enumerate(rows[start : start + 32]):
                ids = tokens[j].tolist()
                q, a = ids.index(261), ids.index(262)
                begin = q + 1 + len(row.question[: row.question.index(row.content)].encode())
                end = begin + len(row.content.encode())
                if end > a or end <= begin:
                    raise ValueError("Invalid observable content boundaries")
                content.append(h[j, begin:end].mean(0).tolist())
                bounds.append(dict(question=q, answer=a, content=[begin, end]))
                for key, v in (
                    ("content_mean", h[j, begin:end].mean(0)),
                    ("question_last", h[j, a - 1]),
                    ("no_marker", h[j, q + 1 : a].mean(0)),
                ):
                    z = model.bottleneck.concept_proj(v)
                    alternatives[key].append(
                        model.bottleneck.attribute_head(z)[list(TAG_IDS)].tolist()
                    )
    t["content_hidden_mean"] = content
    t["bounds"] = bounds
    metrics = {
        key: subset_metrics(torch.tensor(values, device="cuda"), batch, rows)["all"]
        for key, values in alternatives.items()
    }
    pair_report = {}
    for key, pairs in matched_indices(rows).items():
        left, right = zip(*pairs, strict=True)
        result = {}
        for name in (
            "token_embedding_mean",
            "embedding_plus_position_mean",
            "content_hidden_mean",
            "transformer_pooled",
            "concept",
            "logits",
        ):
            v = torch.tensor(t[name], device="cuda")
            result[name] = dict(
                cosine_mean=float(
                    F.cosine_similarity(v[list(left)], v[list(right)], dim=-1).mean()
                ),
                mean_abs_change=float((v[list(left)] - v[list(right)]).abs().mean()),
            )
        v = torch.tensor(t["logits"], device="cuda")
        result["prediction_disagreement"] = float(
            ((v[list(left)] >= 0) != (v[list(right)] >= 0)).any(-1).float().mean()
        )
        pair_report[key] = dict(count=len(pairs), metrics=result)
    return t, dict(pairs=pair_report, pooling_DIAGNOSTIC_ONLY=metrics)


def run(root, a5, out):
    require_remote_cuda()
    spec = json.loads(SPEC.read_text())
    manifest = json.loads(Path("data/manifests/a5_rehearsal_v2.json").read_text())
    if file_hash(root) != ROOT_HASH:
        raise ValueError("Wrong immutable parent")
    paths = {"parent": root, **{k: a5 / f"{k}.pt" for k in ("none", "synthetic_6")}}
    for name in ("none", "synthetic_6"):
        if file_hash(paths[name]) != manifest["sha256"][f"{name}.pt"]:
            raise ValueError("Wrong A5 immutable checkpoint")
    _, sets, _, _ = prepare()
    rows = diagnostic(spec)
    tokenizer = ByteTextTokenizerM103()
    owners = {k: LearningState.load(v) for k, v in paths.items()}
    context = owners["parent"].model.student.base_model.config.context_length
    batch, tokens = encode(rows, tokenizer, context)
    batches = {k: encode(v, tokenizer, context)[0] for k, v in sets.items()}
    # Export the exact reconstructed historical evaluation cases and complete tokens too.
    (out / "cases.json").write_text(
        json.dumps(
            dict(
                diagnostic=[asdict(r) for r in rows],
                a5={k: [asdict(r) for r in v] for k, v in sets.items()},
            )
        )
    )
    (out / "tokens.json").write_text(
        json.dumps(
            dict(
                diagnostic=tokens, a5={k: encode(v, tokenizer, context)[1] for k, v in sets.items()}
            )
        )
    )
    result: dict[str, Any] = dict(
        provenance=dict(
            git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            platform=require_remote_cuda(),
            python=platform.python_version(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),
        ),
        protocol=spec,
        data_sha256=file_hash(out / "cases.json"),
        models={},
        hybrids={},
        optimizer_updates=0,
        sealed_generated=False,
        promoted=False,
    )
    torch.cuda.synchronize()
    start = time.perf_counter()
    for name, owner in owners.items():
        snap = owner.snapshot()
        before = signature(owner)
        counters = {
            n: int(owner.optimizer.state[p]["step"])
            for n, p in owner.model.named_parameters()
            if p in owner.optimizer.state
        }
        model = owner.model
        model.eval()
        original = {k: enrich(evaluate(model, batches[k], v)) for k, v in sets.items()}
        saved = manifest["initial"] if name == "parent" else manifest["arms"][name]["final"]
        # Compact manifest omits full logits; exact, per-property and status scores still reproduce.
        for k in sets:
            for field in ("exact", "status_accuracy", "family_exact"):
                if original[k][field] != saved[k][field]:
                    raise RuntimeError(f"A5 endpoint did not reproduce: {name}/{k}/{field}")
        t, details = hidden_diagnostics(model, batch, rows)
        logits = torch.tensor(t["logits"], device="cuda")
        if logits.tolist() != evaluate(model, batch, rows)["logits"]:
            raise RuntimeError("Manual representation path differs from deployed head")
        shifted = {}
        for offset in (8, 24):
            if batch[0].shape[1] + offset > context:
                raise ValueError("Position offset exceeds actual configured capacity")
            hook = model.student.base_model.wpe.register_forward_pre_hook(
                lambda module, args, offset=offset: (args[0] + offset,)
            )
            try:
                shifted[str(offset)] = enrich(evaluate(model, batch, rows))
            finally:
                hook.remove()
        (out / f"representations_{name}.json").write_text(json.dumps(t))
        result["models"][name] = dict(
            a5_reproduction=original,
            diagnostic=subset_metrics(logits, batch, rows),
            representation=details,
            shifted_position_DIAGNOSTIC_ONLY=shifted,
            optimizer_counters=counters,
            metadata_steps=owner.metadata["steps"],
        )
        owner.restore(snap)
        if signature(owner) != before or file_hash(paths[name]) != (
            ROOT_HASH if name == "parent" else manifest["sha256"][f"{name}.pt"]
        ):
            raise RuntimeError("Inference changed preserved learning state")
        result["models"][name]["unchanged_signature"] = before
        (out / "results.json").write_text(json.dumps(result, indent=2))
    # Compatible isolated combinations only. Neither checkpoint owner is modified.
    pre, post = owners["parent"].model, owners["synthetic_6"].model
    ensure_compatible(pre, post)
    for part in spec["hybrid_components"]:
        for direction, recipient, donor in (
            ("post_with_pre", post, pre),
            ("pre_with_post", pre, post),
        ):
            model = hybrid(recipient, donor, part)
            result["hybrids"][f"{direction}_{part}"] = {
                k: enrich(evaluate(model, batches[k], sets[k]))
                for k in ("new", "old_known", "old_renamed", "development")
            }
            del model
    torch.cuda.synchronize()
    result.update(
        synchronized_diagnostic_seconds=time.perf_counter() - start,
        complete_sequences=len(rows) + sum(map(len, sets.values())),
        max_tokens=max(map(len, tokens)),
        context=context,
        state_unchanged=True,
        input_hashes={k: file_hash(v) for k, v in paths.items()},
    )
    (out / "results.json").write_text(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--a5-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    require_remote_cuda()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Strict CUDA workspace required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    require_numerical_policy(deterministic_numerical_policy())
    seed_all(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run(args.root, args.a5_dir, args.output_dir)


if __name__ == "__main__":
    main()
