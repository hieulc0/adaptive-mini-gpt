"""REMOTE-ONLY inference ablations of compatible frozen checkpoint copies."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from run_attribute_learning import encode, evaluate, file_hash
from run_foundational_audit import digest
from run_stable_attributes import enrich

from adaptive_mini_gpt.data.attribute_diagnostic import factorial
from adaptive_mini_gpt.data.stable_attributes import corpus
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103
from adaptive_mini_gpt.training.learning_state import (
    LearningState,
    deterministic_numerical_policy,
    require_numerical_policy,
)
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda

PRE = "3e93d7238bfdb706ddfd49ee9903fed8b1e6f577529537f1e21cae12e905b262"
POST = "02dcbe3feedd975df34222b2a5a38e0481bedc7a52fe56dff7d655bb14c7adf3"
PARTS = (
    "wte",
    "wpe",
    "lower_blocks",
    "top_block",
    "ln_f",
    "concept_proj",
    "attribute_head",
    "backbone",
    "bottleneck",
)


def components(model, name):
    base = model.student.base_model
    if name == "wte":
        return [base.wte]
    if name == "wpe":
        return [base.wpe]
    if name == "lower_blocks":
        return list(base.blocks[:-1])
    if name == "top_block":
        return [base.blocks[-1]]
    if name == "ln_f":
        return [base.ln_f]
    if name == "concept_proj":
        return [model.bottleneck.concept_proj]
    if name == "attribute_head":
        return [model.bottleneck.attribute_head]
    if name == "backbone":
        return [base]
    if name == "bottleneck":
        return [model.bottleneck]
    raise ValueError("Unknown isolated component")


def ensure_compatible(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    if sa.keys() != sb.keys() or any(
        sa[k].shape != sb[k].shape or sa[k].dtype != sb[k].dtype for k in sa
    ):
        raise ValueError("Checkpoint topologies or tensor types differ")
    for model in (a, b):
        if model.student.base_model.wte.weight is not model.student.base_model.lm_head.weight:
            raise ValueError("Expected tied embedding alias is absent")
        if model.default_strategy != "question_only":
            raise ValueError("Unexpected inference path")


def hybrid(recipient, donor, part):
    result = copy.deepcopy(recipient)
    for target, source in zip(components(result, part), components(donor, part), strict=True):
        target.load_state_dict(source.state_dict(), strict=True)
    if result.student.base_model.wte.weight is not result.student.base_model.lm_head.weight:
        raise ValueError("Swap damaged tied alias")
    return result


def trace(model, batch):
    model.eval()
    out = {
        k: []
        for k in (
            "token_embedding_mean",
            "embedding_plus_position_mean",
            "transformer_pooled",
            "concept",
            "logits",
            "pool_bounds",
        )
    }
    with torch.no_grad():
        for start in range(0, len(batch[0]), 32):
            tok = batch[0][start : start + 32]
            base = model.student.base_model
            emb = base.wte(tok)
            pos = base.wpe(torch.arange(tok.shape[1], device=tok.device))
            hidden = model.extract_sequence_hidden_states(tok)
            pooled, _ = model.pool_representations(hidden, tok)
            concept = model.bottleneck.concept_proj(pooled)
            logits = model.bottleneck.attribute_head(concept)[:, (10, 15)]
            for j, ids in enumerate(tok.tolist()):
                q, a = ids.index(261), ids.index(262)
                out["pool_bounds"].append([q, a])
                out["token_embedding_mean"].append(emb[j, q:a].mean(0).tolist())
                out["embedding_plus_position_mean"].append(
                    (emb[j, q:a] + pos[q:a]).mean(0).tolist()
                )
            out["transformer_pooled"].extend(pooled.tolist())
            out["concept"].extend(concept.tolist())
            out["logits"].extend(logits.tolist())
    return out


def run(pre_path, post_path, out):
    if file_hash(pre_path) != PRE or file_hash(post_path) != POST:
        raise ValueError("Wrong immutable checkpoint")
    pre_owner = LearningState.load(pre_path)
    post_owner = LearningState.load(post_path)
    pre, post = pre_owner.model, post_owner.model
    ensure_compatible(pre, post)
    start_pre, start_post = digest(pre.state_dict()), digest(post.state_dict())
    old = factorial()
    sets = {
        "old_known": old["familiar_familiar"],
        "old_renamed": old["novel_familiar"],
        "new_training": corpus(),
    }
    tokenizer = ByteTextTokenizerM103()
    context = pre.student.base_model.config.context_length
    batches = {}
    seqs = {}
    for name, rows in sets.items():
        batches[name], seqs[name] = encode(rows, tokenizer, context)
    dataset = {n: [asdict(x) for x in rows] for n, rows in sets.items()}
    data_hash = hashlib.sha256(json.dumps(dataset, sort_keys=True).encode()).hexdigest()
    (out / "cases.json").write_text(json.dumps(dataset))
    (out / "tokens.json").write_text(json.dumps(seqs))
    r: dict[str, Any] = dict(
        provenance=dict(
            git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            platform=require_remote_cuda(),
            python=platform.python_version(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),
        ),
        pre_file_sha256=PRE,
        post_file_sha256=POST,
        data_sha256=data_hash,
        components={},
        baseline={},
        hybrids={},
        scope="Inference-only development diagnostics; incompatible hybrids are not deployable; no training or promotion",
    )
    torch.cuda.synchronize()
    start = time.perf_counter()
    for name, model in (("pre", pre), ("post", post)):
        r["baseline"][name] = {
            s: enrich(evaluate(model, batches[s], rows)) for s, rows in sets.items()
        }
        for s, b in batches.items():
            t = trace(model, b)
            assert t["logits"] == r["baseline"][name][s]["logits"]
            (out / f"{name}_{s}_representations.json").write_text(json.dumps(t))
    assert (
        r["baseline"]["pre"]["old_known"]["exact"] == 1
        and r["baseline"]["pre"]["old_renamed"]["exact"] == 1
    )
    assert abs(r["baseline"]["post"]["old_known"]["exact"] - 2 / 3) < 1e-6
    assert r["baseline"]["post"]["new_training"]["exact"] == 1
    for name in PARTS:
        pmods, qmods = components(pre, name), components(post, name)
        changes = []
        for p, q in zip(pmods, qmods, strict=True):
            for key, value in p.state_dict().items():
                other = q.state_dict()[key]
                if value.is_floating_point():
                    diff = (other - value).float()
                    changes.append(
                        dict(
                            name=key,
                            l2=float(diff.norm()),
                            max_abs=float(diff.abs().max()),
                            numel=diff.numel(),
                        )
                    )
        r["components"][name] = changes
        for direction, recipient, donor in (
            ("post_with_pre", post, pre),
            ("pre_with_post", pre, post),
        ):
            candidate = hybrid(recipient, donor, name)
            r["hybrids"][name + "_" + direction] = {
                s: enrich(evaluate(candidate, batches[s], rows)) for s, rows in sets.items()
            }
            del candidate
    torch.cuda.synchronize()
    r["gpu_wall_seconds"] = time.perf_counter() - start
    assert digest(pre.state_dict()) == start_pre and digest(post.state_dict()) == start_post
    assert file_hash(pre_path) == PRE and file_hash(post_path) == POST
    r.update(
        model_tensors_unchanged=True,
        checkpoint_files_unchanged=True,
        optimizer_steps=0,
        sealed_generated=False,
        promoted=False,
    )
    (out / "results.json").write_text(json.dumps(r, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pre", required=True)
    p.add_argument("--post", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    require_remote_cuda()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    require_numerical_policy(deterministic_numerical_policy())
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run(Path(a.pre), Path(a.post), out)


if __name__ == "__main__":
    main()
