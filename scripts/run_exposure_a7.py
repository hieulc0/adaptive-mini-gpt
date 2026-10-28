"""REMOTE ONLY: fixed exposure contrasts then frozen representation probes."""

from __future__ import annotations

import argparse
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

import torch
from annotate_frame_a2_diagnostics import signature
from frame_a2_common import (
    ROOT_HASH,
    ROOT_TENSOR_HASH,
    fork_owner,
    record_stage_provenance,
    state_digest,
)
from probe_representations_a7 import run as probe
from run_attribute_learning import encode, evaluate, file_hash
from run_foundational_audit import digest
from run_generalization_a6_training import dev_gate
from run_generalization_a6_training import prepare as a6_prepare
from run_stable_attributes import compact, enrich

from adaptive_mini_gpt.data.attribute_learning import TAG_IDS
from adaptive_mini_gpt.data.exposure_a7 import factor_subset, repeated_plan, window_pass
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103
from adaptive_mini_gpt.training.learning_state import (
    LearningState,
    deterministic_numerical_policy,
    require_numerical_policy,
)
from adaptive_mini_gpt.training.optimizer_contract import optimizer_membership
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda

SPEC = Path("data/specs/exposure_a7_v1.json")


def prepare():
    spec = json.loads(SPEC.read_text())
    training, newcounts, blocks, shared, _ = a6_prepare()
    small, n, mapping = factor_subset(training["coverage"], newcounts["coverage"])
    training["factor_small"] = small
    newcounts["factor_small"] = n
    blocks["factor_small"] = [[mapping[i] for i in ix] for ix in blocks["coverage"]]
    if n != 378 or len(small) != 546:
        raise ValueError("Wrong factor-matched support")
    for full, small_ix in zip(blocks["coverage"], blocks["factor_small"], strict=True):
        for i, j in zip(full, small_ix, strict=True):
            a, b = training["coverage"][i], small[j]
            if i >= 756:
                if a.question != b.question:
                    raise ValueError("Rehearsal changed")
            elif (a.statuses, a.content, a.frame, a.prefix, len(a.question.encode())) != (
                b.statuses,
                b.content,
                b.frame,
                b.prefix,
                len(b.question.encode()),
            ):
                raise ValueError("Non-name factors changed")
    plans = {k: repeated_plan(blocks[k], steps) for k, steps in spec["arms"].items()}
    exposure = {}
    for arm, plan in plans.items():
        rows = training[arm]
        n = newcounts[arm]
        counts = Counter(i for ix in plan for i in ix)
        if set(counts) != set(range(len(rows))):
            raise ValueError("Missing training exposure")
        if sum(counts[i] for i in range(n)) != 12 * len(plan):
            raise ValueError("New dose mismatch")
        if any(sum(i >= n for i in ix) != 6 for ix in plan):
            raise ValueError("Rehearsal changed per slot")
        exposure[arm] = dict(
            updates=len(plan),
            examples=18 * len(plan),
            new_presentations=12 * len(plan),
            old_presentations=6 * len(plan),
            unique_new=n,
            unique_old=len(rows) - n,
            new_presentations_per_unique=12 * len(plan) / n,
            counts_by_case=[counts[i] for i in range(len(rows))],
            new_min=min(counts[i] for i in range(n)),
            new_max=max(counts[i] for i in range(n)),
        )
    return training, newcounts, plans, shared, exposure


def advance(owner, batch):
    cursor = owner.external["a7_cursor"]
    ix = owner.external["a7_indices"][cursor % len(owner.external["a7_indices"])]
    owner.external["a7_cursor"] = cursor + 1
    return owner.step(*[v[ix] for v in batch], attribute_ids=TAG_IDS), ix


def label_losses(m):
    result = []
    for j in range(2):
        values = [
            (max(v[j], 0) - v[j] * t[j] + math.log1p(math.exp(-abs(v[j]))), t[j])
            for v, t in zip(m["logits"], m["truth"], strict=True)
        ]
        result.append(
            dict(
                mean=sum(x for x, _ in values) / len(values),
                positive=sum(x for x, t in values if t) / max(1, sum(t for _, t in values)),
                negative=sum(x for x, t in values if not t) / max(1, sum(not t for _, t in values)),
            )
        )
    return result


def run(root, out):
    if file_hash(root) != ROOT_HASH:
        raise ValueError("Wrong immutable parent")
    parent = LearningState.load(root)
    if digest(parent.model.state_dict()) != ROOT_TENSOR_HASH or parent.scheduler is not None:
        raise ValueError("Wrong parent state")
    training, newcounts, plans, shared, exposure = prepare()
    spec = json.loads(SPEC.read_text())
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    data = dict(
        training={k: [asdict(x) for x in v] for k, v in training.items()},
        shared={k: [asdict(x) for x in v] for k, v in shared.items()},
        streams=plans,
        exposure=exposure,
    )
    (out / "dataset.json").write_text(json.dumps(data, sort_keys=True))
    datahash = file_hash(out / "dataset.json")
    tokenizer = ByteTextTokenizerM103()
    context = parent.model.student.base_model.config.context_length
    batches = {}
    sequences = {}
    shared_batches = {}
    for k, rs in training.items():
        batches[k], sequences[k] = encode(rs, tokenizer, context)
    for k, rs in shared.items():
        shared_batches[k], sequences[k] = encode(rs, tokenizer, context)
    (out / "tokens.json").write_text(json.dumps(sequences))
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
        data_sha256=datahash,
        exposure=exposure,
        arms={},
        sealed_generated=False,
        promoted=False,
    )

    def store():
        (out / "results.json").write_text(json.dumps(r, indent=2))

    a6 = json.loads(Path("data/manifests/a6_training_v1.json").read_text())
    for arm, steps in spec["arms"].items():
        owner = fork_owner(parent, 42)
        if any(optimizer_membership(owner.model, owner.optimizer).values()):
            raise RuntimeError("Unregistered parameter")
        if state_digest(owner.optimizer.state_dict()) != state_digest(
            parent.optimizer.state_dict()
        ) or any(g["lr"] != 0.002 or g["weight_decay"] != 0 for g in owner.optimizer.param_groups):
            raise RuntimeError("Changed inherited optimizer")
        record_stage_provenance(
            owner,
            training_sha=source,
            publication_sha=source,
            parent_hash=ROOT_HASH,
            comparison_hash=ROOT_HASH,
            data_hash=datahash,
            stage="A.7_exposure",
        )
        owner.metadata.update(
            numerical_policy=deterministic_numerical_policy(),
            seed=42,
            a7_arm=arm,
            supervision_schema=dict(attribute_ids=TAG_IDS, state_targets=None),
        )
        owner.external.update(a7_cursor=0, a7_indices=plans[arm], a7_training_hash=datahash)
        rows = training[arm]
        n = newcounts[arm]
        batch = batches[arm]
        sets = dict(training=rows[:n], synthetic=rows[n:], **shared)
        eval_batches = dict(
            training=tuple(v[:n] for v in batch),
            synthetic=tuple(v[n:] for v in batch),
            **shared_batches,
        )
        a: dict[str, Any] = dict(
            initial_hash=digest(owner.model.state_dict()),
            initial_optimizer_hash=state_digest(owner.optimizer.state_dict()),
            stored_parameters=sum(p.numel() for p in owner.model.parameters()),
            trainable_parameters=sum(
                p.numel() for p in owner.model.parameters() if p.requires_grad
            ),
            updates=0,
            examples=0,
            tokens=0,
            curves=[],
            checkpoints={},
            milestone_hashes={},
            evaluation_rows=0,
            evaluation_forward_batches=0,
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss = None
        for update in range(steps + 1):
            if update % 100 == 0:
                metrics = {
                    k: enrich(evaluate(owner.model, eval_batches[k], rs)) for k, rs in sets.items()
                }
                a["evaluation_rows"] += sum(map(len, sets.values()))
                a["evaluation_forward_batches"] += sum(
                    math.ceil(len(v) / 32) for v in sets.values()
                )
                curve = dict(
                    updates=update,
                    examples=a["examples"],
                    training=compact(metrics["training"]),
                    synthetic=compact(metrics["synthetic"]),
                    retention={k: compact(metrics[k]) for k in ("old_known", "old_renamed")},
                    development={
                        k: compact(metrics[k]) for k in shared if k.startswith("development")
                    },
                    last_batch_loss=loss,
                    per_label_training_loss=label_losses(metrics["training"]),
                    gradient_norms={
                        name: float(p.grad.norm())
                        for name, p in owner.model.named_parameters()
                        if p.grad is not None
                    },
                )
                a["curves"].append(curve)
                a["milestone_hashes"][str(update)] = digest(owner.model.state_dict())
                if update == 0 and any(
                    metrics[k]["exact"] != 1 for k in ("old_known", "old_renamed")
                ):
                    raise RuntimeError("Old competence absent")
                if update == 0 or update in spec["snapshot_updates"]:
                    (out / f"predictions_{arm}_{update:04d}.json").write_text(json.dumps(metrics))
                if update in spec["snapshot_updates"]:
                    path = out / f"{arm}_{update}.pt"
                    owner.save(path)
                    a["checkpoints"][str(update)] = file_hash(path)
                if (
                    update == 1200
                    and arm in a6["arms"]
                    and a["milestone_hashes"][str(update)] != a6["arms"][arm]["final_hash"]
                ):
                    raise RuntimeError("A6 trajectory endpoint did not reproduce")
                a["final"] = metrics
                r["arms"][arm] = a
                store()
            if update == steps:
                break
            loss, ix = advance(owner, batch)
            a["updates"] += 1
            a["examples"] += len(ix)
            a["tokens"] += int((batch[0][ix] != 256).sum())
            if update == 0:
                a["first_update_active_parameters"] = sum(
                    p.numel() for p in owner.model.parameters() if p.grad is not None
                )
        torch.cuda.synchronize()
        a["synchronized_wall_seconds"] = time.perf_counter() - start
        a["lifetime_steps"] = owner.metadata["steps"]
        a["adam_counters"] = {
            name: int(owner.optimizer.state[p]["step"])
            for name, p in owner.model.named_parameters()
            if p in owner.optimizer.state
        }
        a["windows"] = {
            str(end): window_pass(a["curves"], end, {x.family for x in rows[:n]})
            for end in (1200, 2400, 7200)
            if end <= steps
        }
        a["development_admission"] = all(
            dev_gate(a["final"][k]) for k in shared if k.startswith("development")
        )
        store()
        del owner
    paths = dict(
        parent=root, coverage_1200=out / "coverage_1200.pt", coverage_7200=out / "coverage_7200.pt"
    )
    torch.cuda.synchronize()
    start = time.perf_counter()
    r["representation_probes"] = probe(paths, training["coverage"][:756], shared, out)
    torch.cuda.synchronize()
    r["probe_wall_seconds"] = time.perf_counter() - start
    store()
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--resume-check",
            str(out / "coverage_7200.pt"),
            "--output-dir",
            str(out),
        ],
        check=True,
        timeout=120,
    )
    expected = json.loads((out / "resume_child.json").read_text())
    owner = LearningState.load(out / "coverage_7200.pt")
    pred = evaluate(owner.model, batches["coverage"], training["coverage"])
    loss, ix = advance(owner, batches["coverage"])
    if expected != dict(
        predictions=pred,
        loss=loss,
        indices=ix,
        signature=signature(owner),
        steps=owner.metadata["steps"],
    ):
        raise RuntimeError("Fresh-process continuation differs")
    r["fresh_process_continuation"] = dict(
        exact=True,
        disposable_updates=2,
        candidate_unchanged=file_hash(out / "coverage_7200.pt")
        == r["arms"]["coverage"]["checkpoints"]["7200"],
    )
    r["parent_unchanged"] = (
        file_hash(root) == ROOT_HASH and digest(parent.model.state_dict()) == ROOT_TENSOR_HASH
    )
    if not r["parent_unchanged"]:
        raise RuntimeError("Immutable parent changed")
    primary = r["arms"]["coverage"]
    w = primary["windows"]["7200"]
    r["learning_prerequisites"] = (
        w["acquisition"] and w["preservation"] and primary["development_admission"]
    )
    r["positive_readout_diagnosis"] = any(
        any(m["positive_readout_diagnosis"].values())
        for m in r["representation_probes"]["models"].values()
    )
    r["verdict"] = (
        "DIAGNOSIS_COMPLETE_REVIEW_CONDITIONAL_INTERVENTION"
        if r["positive_readout_diagnosis"]
        else "NO_QUALIFIED_READOUT_CORRECTION_STOP"
    )
    r["input"] = dict(
        context=context,
        max_tokens=max(len(ids) for seq in sequences.values() for ids in seq),
        complete_sequences=sum(map(len, sequences.values())),
    )
    store()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path)
    p.add_argument("--resume-check", type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    require_remote_cuda()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Strict CUDA workspace required")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    require_numerical_policy(deterministic_numerical_policy())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_check:
        owner = LearningState.load(args.resume_check)
        training, _, _, _, _ = prepare()
        batch, _ = encode(
            training["coverage"],
            ByteTextTokenizerM103(),
            owner.model.student.base_model.config.context_length,
        )
        pred = evaluate(owner.model, batch, training["coverage"])
        loss, ix = advance(owner, batch)
        (args.output_dir / "resume_child.json").write_text(
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
    elif args.root:
        run(args.root, args.output_dir)
    else:
        p.error("--root required")


if __name__ == "__main__":
    main()
