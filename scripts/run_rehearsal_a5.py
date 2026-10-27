"""REMOTE-ONLY conditional rehearsal coverage, preserving the continuing parent."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import run_joint_acquisition_a5 as joint
import torch
import torch.nn.functional as F
from annotate_frame_a2_diagnostics import signature
from frame_a2_common import (
    ROOT_HASH,
    ROOT_TENSOR_HASH,
    fork_owner,
    record_stage_provenance,
    state_digest,
)
from run_a4_replay import prepare as historical_prepare
from run_attribute_learning import encode, evaluate, file_hash, scores, words
from run_foundational_audit import digest
from run_stable_attributes import compact, enrich

from adaptive_mini_gpt.data.attribute_learning import TAG_IDS
from adaptive_mini_gpt.data.joint_acquisition_a5 import validate_disjoint
from adaptive_mini_gpt.data.rehearsal_a5 import ARMS, plans, stable_new, stable_old, synthetic_old
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103
from adaptive_mini_gpt.training.learning_state import (
    LearningState,
    deterministic_numerical_policy,
    require_numerical_policy,
)
from adaptive_mini_gpt.training.optimizer_contract import optimizer_membership
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda
from adaptive_mini_gpt.training.reproducibility import seed_all

SPEC = Path("data/specs/rehearsal_a5_v1.json")


def prepare():
    reference = json.loads(Path("data/manifests/a5_joint_v1.json").read_text())
    if (
        reference["status"] != "COMPLETE"
        or not reference["joint_acquisition_admission"]
        or reference["provenance"]["git_sha"] != "a79c74471f812067d45e6079c4e5d3e544ccd53c"
    ):
        raise ValueError("Joint acquisition prerequisite missing")
    all_joint, original, previous, dev, audit = joint.prepare()
    payload = dict(training=[asdict(r) for r in all_joint], indices=original, audit=audit)
    if (
        hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        != reference["data_sha256"]
    ):
        raise ValueError("Phase1 data changed")
    _, historical, _, _, _, hist_audit = historical_prepare()
    new = [r for r in all_joint if r.group == "new"]
    synthetic = synthetic_old(all_joint)
    protected = [
        r.question
        for sets in (previous, joint.generate(), joint.evaluation_sets())
        for rows in sets.values()
        for r in rows
    ]
    protected += [
        r.question
        for r in all_joint + joint.prior_corpus() + joint.minimal_pairs() + dev + historical
    ]
    protected += [
        r["question"]
        for r in json.loads(Path("data/fixtures/stage_a_prior_state_tiny.json").read_text())[
            "cases"
        ]
    ]
    validate_disjoint(synthetic, protected)
    if len(new) != 126 or len(synthetic) != 168 or len(historical) != 9:
        raise ValueError("Wrong rehearsal support")
    spec = json.loads(SPEC.read_text())
    streams = plans(new, historical, synthetic, spec["seed"], spec["updates"])
    rows = new + historical + synthetic
    exposure = {}
    for arm, indices in streams.items():
        counts = Counter(i for ix in indices for i in ix)
        statuses = Counter(tuple(rows[i].statuses) for ix in indices for i in ix)
        if set(statuses.values()) != {2400}:
            raise ValueError("Unequal annotation exposure")
        old = sum(n for i, n in counts.items() if i >= len(new))
        if old != (0 if arm == "none" else 10800 if arm == "balanced_9" else 7200):
            raise ValueError("Incorrect replay ratio")
        if any(i not in counts for i in range(len(new))):
            raise ValueError("Undelivered new case")
        for base, ix in zip(streams["none"], indices, strict=True):
            if [tuple(rows[i].statuses) for i in base] != [tuple(rows[i].statuses) for i in ix]:
                raise ValueError("Changed status per slot")
        exposure[arm] = dict(
            new_presentations=21600 - old,
            old_presentations=old,
            unique_new=sum(i < len(new) for i in counts),
            unique_historical=sum(len(new) <= i < len(new) + len(historical) for i in counts),
            unique_synthetic=sum(i >= len(new) + len(historical) for i in counts),
            counts_by_case=[counts[i] for i in range(len(rows))],
        )
    sets = dict(
        new=new,
        historical=historical,
        synthetic=synthetic,
        prior_joint_old=[r for r in all_joint if r.group == "old"],
        old_known=previous["familiar_familiar"],
        old_renamed=previous["novel_familiar"],
        development=dev,
    )
    data = dict(
        rows=[asdict(r) for r in rows],
        streams=streams,
        historical_audit=hist_audit,
        exposure=exposure,
    )
    return rows, sets, streams, data


def advance(owner, batch):
    cursor = owner.external["a5r_cursor"]
    ix = owner.external["a5r_indices"][cursor % len(owner.external["a5r_indices"])]
    owner.external["a5r_cursor"] = cursor + 1
    return owner.step(*[v[ix] for v in batch], attribute_ids=TAG_IDS), ix


def resume(path, out):
    owner = LearningState.load(path)
    rows, _, _, _ = prepare()
    batch, _ = encode(
        rows, ByteTextTokenizerM103(), owner.model.student.base_model.config.context_length
    )
    pred = evaluate(owner.model, batch, rows)
    loss, ix = advance(owner, batch)
    out.write_text(
        json.dumps(
            dict(
                predictions=pred,
                loss=loss,
                indices=ix,
                signature=signature(owner),
                steps=owner.metadata["steps"],
            )
        )
    )


def run(parent_path, out):
    if file_hash(parent_path) != ROOT_HASH:
        raise ValueError("Wrong continuing parent")
    parent = LearningState.load(parent_path)
    if digest(parent.model.state_dict()) != ROOT_TENSOR_HASH or parent.scheduler is not None:
        raise ValueError("Incorrect root state")
    rows, sets, streams, data = prepare()
    spec = json.loads(SPEC.read_text())
    data_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    tokenizer = ByteTextTokenizerM103()
    context = parent.model.student.base_model.config.context_length
    batch, tokens = encode(rows, tokenizer, context)
    seqs = {"training_all": tokens}
    batches = {}
    counts = (len(sets["new"]), len(sets["historical"]), len(sets["synthetic"]))
    offset = 0
    for name, count in zip(("new", "historical", "synthetic"), counts, strict=True):
        batches[name] = tuple(v[offset : offset + count] for v in batch)
        offset += count
    for name in ("prior_joint_old", "old_known", "old_renamed", "development"):
        batches[name], seqs[name] = encode(sets[name], tokenizer, context)
    (out / "dataset.json").write_text(json.dumps(data))
    (out / "evaluation_cases.json").write_text(
        json.dumps({s: [asdict(r) for r in values] for s, values in sets.items()})
    )
    (out / "tokens.json").write_text(json.dumps(seqs))
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    initial = {s: enrich(evaluate(parent.model, batches[s], values)) for s, values in sets.items()}
    if min(initial[s]["exact"] for s in ("old_known", "old_renamed")) != 1:
        raise ValueError("Old baseline competence absent")
    r: dict[str, Any] = dict(
        provenance=dict(
            git_sha=source,
            platform=require_remote_cuda(),
            python=platform.python_version(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),
        ),
        protocol=spec,
        parent_sha256=ROOT_HASH,
        data_sha256=data_hash,
        initial=initial,
        initial_evaluation_rows=sum(map(len, sets.values())),
        arms={},
        baselines={},
        sealed_generated=False,
        promoted=False,
    )

    def store():
        (out / "results.json").write_text(json.dumps(r, indent=2))

    store()
    for arm in ARMS:
        owner = fork_owner(parent, spec["seed"])
        if owner.model.default_strategy != "question_only" or any(
            optimizer_membership(owner.model, owner.optimizer).values()
        ):
            raise RuntimeError("Incorrect active path/optimizer membership")
        initial_opt = state_digest(owner.optimizer.state_dict())
        if initial_opt != state_digest(parent.optimizer.state_dict()) or any(
            g["lr"] != 0.002 or g["weight_decay"] != 0 for g in owner.optimizer.param_groups
        ):
            raise RuntimeError("Optimizer changed before trial")
        record_stage_provenance(
            owner,
            training_sha=source,
            publication_sha=source,
            parent_hash=ROOT_HASH,
            comparison_hash=ROOT_HASH,
            data_hash=data_hash,
            stage="A.5_rehearsal",
        )
        owner.metadata.update(
            numerical_policy=deterministic_numerical_policy(),
            seed=spec["seed"],
            a5r_arm=arm,
            supervision_schema=dict(attribute_ids=TAG_IDS, state_targets=None),
        )
        owner.external.update(
            a5r_indices=streams[arm],
            a5r_cursor=0,
            a5r_sources=dict(
                historical=[asdict(v) for v in sets["historical"]],
                synthetic=[asdict(v) for v in sets["synthetic"]],
                data_sha256=data_hash,
            ),
        )
        a: dict[str, Any] = dict(
            initial_hash=digest(owner.model.state_dict()),
            initial_optimizer_hash=initial_opt,
            stored_parameters=sum(p.numel() for p in owner.model.parameters()),
            trainable_parameters=sum(
                p.numel() for p in owner.model.parameters() if p.requires_grad
            ),
            updates=0,
            examples=0,
            tokens=0,
            new_presentations=0,
            old_presentations=0,
            curves=[],
            evaluation_rows=0,
            evaluation_forward_batches=0,
        )
        final = initial
        torch.cuda.synchronize()
        start = time.perf_counter()
        for step in range(spec["updates"]):
            loss, ix = advance(owner, batch)
            a["updates"] += 1
            a["examples"] += len(ix)
            a["tokens"] += int((batch[0][ix] != 256).sum())
            a["new_presentations"] += sum(i < counts[0] for i in ix)
            a["old_presentations"] += sum(i >= counts[0] for i in ix)
            if step == 0:
                a["first_update_active_parameters"] = sum(
                    p.numel() for p in owner.model.parameters() if p.grad is not None
                )
            if (step + 1) % 100 == 0:
                final = {
                    s: enrich(evaluate(owner.model, batches[s], values))
                    for s, values in sets.items()
                }
                a["evaluation_rows"] += sum(map(len, sets.values()))
                a["evaluation_forward_batches"] += sum(
                    math.ceil(len(v) / 32) for v in sets.values()
                )
                a["curves"].append(
                    dict(
                        updates=step + 1,
                        examples=a["examples"],
                        last_batch_loss=loss,
                        training=compact(final["new"]),
                        historical=compact(final["historical"]),
                        synthetic=compact(final["synthetic"]),
                        prior_joint_old=compact(final["prior_joint_old"]),
                        retention={s: compact(final[s]) for s in ("old_known", "old_renamed")},
                        development=compact(final["development"]),
                    )
                )
        a["final"] = final
        a["stable_new_acquisition"] = stable_new(a["curves"])
        a["stable_old_observations"] = stable_old(a["curves"])
        m = final["development"]
        a["development_admission"] = (
            m["exact"] >= 0.9
            and min(m["family_exact"].values()) >= 0.85
            and min(m["status_accuracy"].values()) >= 0.95
        )
        a["final_hash"] = digest(owner.model.state_dict())
        a["final_optimizer_hash"] = state_digest(owner.optimizer.state_dict())
        a["lifetime_updates"] = owner.metadata["steps"]
        a["adam_counters"] = {
            n: int(owner.optimizer.state[p]["step"])
            for n, p in owner.model.named_parameters()
            if p in owner.optimizer.state
        }
        owner.save(out / f"{arm}.pt")
        a["checkpoint_sha256"] = file_hash(out / f"{arm}.pt")
        torch.cuda.synchronize()
        a["synchronized_wall_seconds"] = time.perf_counter() - start
        r["arms"][arm] = a
        store()
        del owner
    allrows = sum(sets.values(), [])
    ranges = {}
    offset = 0
    for s, values in sets.items():
        ranges[s] = list(range(offset, offset + len(values)))
        offset += len(values)
    # allrows begins with new/historical/synthetic in exactly the update index order.
    if [row.question for row in allrows[: len(rows)]] != [row.question for row in rows]:
        raise RuntimeError("Baseline source indexing differs")
    y = torch.tensor([row.labels for row in allrows], device="cuda", dtype=torch.float32)
    for kind in ("bow_linear", "ngram_linear"):
        for arm in ARMS:
            delivered = {i for ix in streams[arm] for i in ix}
            vocab = sorted({w for i in delivered for w in words(rows[i], kind)})
            counters = [Counter(words(row, kind)) for row in allrows]
            x = torch.tensor([[float(c[w]) for w in vocab] for c in counters], device="cuda")
            seed_all(spec["seed"])
            model = torch.nn.Linear(len(vocab), 2).cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for ix in streams[arm]:
                opt.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(model(x[ix]), y[ix])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            with torch.no_grad():
                logits = model(x)
            metrics = {s: enrich(scores(logits[ix], y[ix], sets[s])) for s, ix in ranges.items()}
            torch.cuda.synchronize()
            r["baselines"][f"{kind}_{arm}"] = dict(
                metrics=metrics,
                parameters=sum(p.numel() for p in model.parameters()),
                updates=spec["updates"],
                examples=21600,
                synchronized_wall_seconds=time.perf_counter() - start,
            )
            store()
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--resume-check",
            str(out / "synthetic_6.pt"),
            "--output-dir",
            str(out),
        ],
        check=True,
        timeout=120,
    )
    actual = json.loads((out / "resume_child.json").read_text())
    owner = LearningState.load(out / "synthetic_6.pt")
    pred = evaluate(owner.model, batch, rows)
    loss, ix = advance(owner, batch)
    if actual != dict(
        predictions=pred,
        loss=loss,
        indices=ix,
        signature=signature(owner),
        steps=owner.metadata["steps"],
    ):
        raise RuntimeError("Fresh-process continuation differs")
    r["fresh_process_continuation"] = dict(
        exact=True,
        disposable_extra_updates=2,
        candidate_unchanged=file_hash(out / "synthetic_6.pt")
        == r["arms"]["synthetic_6"]["checkpoint_sha256"],
    )
    a = r["arms"]["synthetic_6"]
    r["primary_preservation_admission"] = (
        a["stable_new_acquisition"] and a["stable_old_observations"]
    )
    r["qualification_admission"] = (
        r["primary_preservation_admission"] and a["development_admission"]
    )
    r["verdict"] = (
        "READY_FOR_SEPARATE_QUALIFICATION_PROTOCOL"
        if r["qualification_admission"]
        else "PRESERVATION_WORKS_LANGUAGE_GATE_FAIL_STOP"
        if r["primary_preservation_admission"]
        else "REHEARSAL_PRESERVATION_FAIL"
    )
    r["input"] = dict(
        new=len(sets["new"]),
        historical=len(sets["historical"]),
        synthetic=len(sets["synthetic"]),
        context=context,
        max_tokens=max(len(ids) for seq in seqs.values() for ids in seq),
    )
    if digest(parent.model.state_dict()) != ROOT_TENSOR_HASH or file_hash(parent_path) != ROOT_HASH:
        raise RuntimeError("Parent changed")
    r["parent_unchanged"] = True
    store()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root")
    p.add_argument("--resume-check")
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    require_remote_cuda()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Deterministic workspace missing")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    require_numerical_policy(deterministic_numerical_policy())
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.resume_check:
        resume(Path(a.resume_check), out / "resume_child.json")
    elif a.root:
        run(Path(a.root), out)
    else:
        p.error("--root required")


if __name__ == "__main__":
    main()
