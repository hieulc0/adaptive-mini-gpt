"""REMOTE ONLY: frozen A6 coverage ablation; no new modules or promotion."""

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
import torch.nn.functional as F
from annotate_frame_a2_diagnostics import signature
from frame_a2_common import (
    ROOT_HASH,
    ROOT_TENSOR_HASH,
    fork_owner,
    record_stage_provenance,
    state_digest,
)
from run_attribute_learning import encode, evaluate, file_hash, scores, words
from run_foundational_audit import digest
from run_generalization_a6_diagnostic import SPEC as DIAG_SPEC
from run_joint_acquisition_a5 import evaluation_sets, generate, minimal_pairs, prior_corpus
from run_rehearsal_a5 import prepare as a5_prepare
from run_stable_attributes import compact, enrich

from adaptive_mini_gpt.data.attribute_learning import TAG_IDS
from adaptive_mini_gpt.data.generalization_a6 import (
    coverage_corpus,
    coverage_plan,
    development_corpus,
    diagnostic,
    stable_acquisition,
    structured_features,
)
from adaptive_mini_gpt.data.joint_acquisition_a5 import STATUSES, normalized, validate_disjoint
from adaptive_mini_gpt.data.rehearsal_a5 import stable_old
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103
from adaptive_mini_gpt.training.learning_state import (
    LearningState,
    deterministic_numerical_policy,
    require_numerical_policy,
)
from adaptive_mini_gpt.training.optimizer_contract import optimizer_membership
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda
from adaptive_mini_gpt.training.reproducibility import seed_all

SPEC = Path("data/specs/generalization_a6_training_v1.json")


def prepare():
    spec = json.loads(SPEC.read_text())
    original, oldsets, streams, _ = a5_prepare()
    ref, syn = oldsets["new"], oldsets["synthetic"]
    broad = coverage_corpus(ref)
    previous_allowed = {normalized(r.question) for r in ref + syn}
    protected = [
        r.question
        for groups in (generate(), evaluation_sets())
        for rs in groups.values()
        for r in rs
    ]
    protected += [r.question for rs in oldsets.values() for r in rs]
    protected += [
        r.question
        for r in prior_corpus() + minimal_pairs() + diagnostic(json.loads(DIAG_SPEC.read_text()))
    ]
    for fname in ("stage_a_original_tiny.json", "stage_a_prior_state_tiny.json"):
        protected += [
            r["question"] for r in json.loads(Path("data/fixtures", fname).read_text())["cases"]
        ]
    protected = [q for q in protected if normalized(q) not in previous_allowed]
    validate_disjoint(broad, protected)
    dev = {
        f"development_{kind}": development_corpus(ref, spec[f"development_{kind}_names"])
        for kind in ("familiar", "novel")
    }
    all_train = {normalized(r.question) for r in broad + ref + syn}
    if any(normalized(r.question) in all_train for rs in dev.values() for r in rs):
        raise ValueError("Development overlaps training")
    if len(broad) != 756 or any(len(rs) != 126 for rs in dev.values()):
        raise ValueError("Unexpected support")
    old_offset = len(ref) + len(oldsets["historical"])
    base = streams["synthetic_6"]
    plans = dict(
        reference=[
            [i if i < len(ref) else i - len(oldsets["historical"]) for i in ix] for ix in base
        ],
        coverage=coverage_plan(base, ref, broad, old_offset),
    )
    training = dict(reference=ref + syn, coverage=broad + syn)
    newcounts = dict(reference=len(ref), coverage=len(broad))
    exposures = {}
    for arm, indices in plans.items():
        rows = training[arm]
        counts = Counter(i for ix in indices for i in ix)
        if set(counts) != set(range(len(rows))):
            raise ValueError("Missing delivered training example")
        old_total = sum(n for i, n in counts.items() if i >= newcounts[arm])
        if old_total != 7200 or sum(counts.values()) != 21600:
            raise ValueError("Changed example budgets")
        for a, b in zip(plans["reference"], indices, strict=True):
            if Counter(tuple(rows[i].statuses) for i in b) != Counter({s: 2 for s in STATUSES}):
                raise ValueError("Unbalanced status batch")
            for j, i in enumerate(b):
                r = training["reference"][a[j]]
                if tuple(r.statuses) != tuple(rows[i].statuses):
                    raise ValueError("Annotation mismatch per slot")
                if (a[j] >= len(ref)) != (i >= newcounts[arm]):
                    raise ValueError("Old slot changed")
                if i >= newcounts[arm] and r.question != rows[i].question:
                    raise ValueError("Historical rehearsal identity changed")
        cells = (
            Counter(
                (tuple(rows[i].statuses), rows[i].frame, rows[i].prefix)
                for ix in indices
                for i in ix
                if i < newcounts[arm]
            )
            if arm == "coverage"
            else Counter()
        )
        if cells and (len(cells) != 162 or max(cells.values()) - min(cells.values()) > 1):
            raise ValueError("Confounded coverage grammar/prefix/status exposure")
        exposures[arm] = dict(
            counts_by_case=[counts[i] for i in range(len(rows))],
            new=14400,
            old=7200,
            unique=len(counts),
            cell_counts={str(k): v for k, v in cells.items()},
        )
    shared = {k: oldsets[k] for k in ("old_known", "old_renamed", "development")}
    shared.update(dev)
    return training, newcounts, plans, shared, exposures


def advance(owner, batch):
    cursor = owner.external["a6_cursor"]
    ix = owner.external["a6_indices"][cursor % len(owner.external["a6_indices"])]
    owner.external["a6_cursor"] = cursor + 1
    return owner.step(*[v[ix] for v in batch], attribute_ids=TAG_IDS), ix


def dev_gate(m):
    return (
        m["exact"] >= 0.9
        and min(m["family_exact"].values()) >= 0.85
        and min(m["status_accuracy"].values()) >= 0.95
    )


def controls(training, plans, newcounts, shared, out):
    result = {}
    for arm, train in training.items():
        sets = dict(training=train[: newcounts[arm]], synthetic=train[newcounts[arm] :], **shared)
        rows = train + sum(shared.values(), [])
        ranges, offset = {}, 0
        for role, rs in sets.items():
            ranges[role] = list(range(offset, offset + len(rs)))
            offset += len(rs)
        y = torch.tensor([r.labels for r in rows], device="cuda", dtype=torch.float32)
        for kind in ("bow_linear", "ngram_linear", "structured_diagnostic_linear"):
            if kind.startswith("structured"):
                x = torch.tensor(structured_features(rows), device="cuda")
            else:
                vocab = sorted({w for r in train for w in words(r, kind)})
                counters = [Counter(words(r, kind)) for r in rows]
                x = torch.tensor([[float(c[w]) for w in vocab] for c in counters], device="cuda")
            seed_all(42)
            model = torch.nn.Linear(x.shape[1], 2).cuda()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for ix in plans[arm]:
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(model(x[ix]), y[ix])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            with torch.no_grad():
                logits = model(x)
            torch.cuda.synchronize()
            result[f"{kind}_{arm}"] = dict(
                metrics={k: enrich(scores(logits[ix], y[ix], sets[k])) for k, ix in ranges.items()},
                parameters=sum(p.numel() for p in model.parameters()),
                updates=1200,
                examples=21600,
                synchronized_wall_seconds=time.perf_counter() - start,
                privileged_DIAGNOSTIC_ONLY=kind.startswith("structured"),
            )
            (out / "controls.json").write_text(json.dumps(result))
    return result


def run(root, out):
    spec = json.loads(SPEC.read_text())
    diagnostic_manifest = json.loads(Path("data/manifests/a6_diagnostic_v1.json").read_text())
    if diagnostic_manifest["status"] != "COMPLETE" or not diagnostic_manifest["state_unchanged"]:
        raise ValueError("Inference inspection must complete first")
    if file_hash(root) != ROOT_HASH:
        raise ValueError("Wrong immutable continuing student")
    parent = LearningState.load(root)
    if digest(parent.model.state_dict()) != ROOT_TENSOR_HASH or parent.scheduler is not None:
        raise ValueError("Parent state mismatch")
    training, newcounts, plans, shared, exposures = prepare()
    data = dict(
        training={k: [asdict(v) for v in rs] for k, rs in training.items()},
        shared={k: [asdict(v) for v in rs] for k, rs in shared.items()},
        streams=plans,
        exposures=exposures,
    )
    (out / "dataset.json").write_text(json.dumps(data, sort_keys=True))
    data_hash = file_hash(out / "dataset.json")
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    context = parent.model.student.base_model.config.context_length
    tokenizer = ByteTextTokenizerM103()
    batches, sequences = {}, {}
    for arm, rows in training.items():
        batches[arm], sequences[arm] = encode(rows, tokenizer, context)
    shared_batches = {}
    for role, rows in shared.items():
        shared_batches[role], sequences[role] = encode(rows, tokenizer, context)
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
        data_sha256=data_hash,
        arms={},
        optimizer_reset=False,
        new_parameters=False,
        sealed_generated=False,
        promoted=False,
    )

    def store():
        (out / "results.json").write_text(json.dumps(r, indent=2))

    for arm, rows in training.items():
        owner = fork_owner(parent, 42)
        if any(optimizer_membership(owner.model, owner.optimizer).values()):
            raise RuntimeError("Optimizer membership mismatch")
        if state_digest(owner.optimizer.state_dict()) != state_digest(
            parent.optimizer.state_dict()
        ) or any(g["lr"] != 0.002 or g["weight_decay"] != 0 for g in owner.optimizer.param_groups):
            raise RuntimeError("Optimizer policy changed")
        record_stage_provenance(
            owner,
            training_sha=source,
            publication_sha=source,
            parent_hash=ROOT_HASH,
            comparison_hash=ROOT_HASH,
            data_hash=data_hash,
            stage="A.6_coverage",
        )
        owner.metadata.update(
            numerical_policy=deterministic_numerical_policy(),
            seed=42,
            a6_arm=arm,
            supervision_schema=dict(attribute_ids=TAG_IDS, state_targets=None),
        )
        owner.external.update(a6_cursor=0, a6_indices=plans[arm], a6_training_data_hash=data_hash)
        batch = batches[arm]
        sets = dict(training=rows[: newcounts[arm]], synthetic=rows[newcounts[arm] :], **shared)
        eval_batches = dict(
            training=tuple(v[: newcounts[arm]] for v in batch),
            synthetic=tuple(v[newcounts[arm] :] for v in batch),
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
            evaluation_rows=0,
            evaluation_forward_batches=0,
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss = None
        for update in range(1201):
            if update % 100 == 0:
                metrics = {
                    k: enrich(evaluate(owner.model, eval_batches[k], rs)) for k, rs in sets.items()
                }
                (out / f"predictions_{arm}_{update:04d}.json").write_text(json.dumps(metrics))
                a["evaluation_rows"] += sum(map(len, sets.values()))
                a["evaluation_forward_batches"] += sum(
                    math.ceil(len(rs) / 32) for rs in sets.values()
                )
                a["curves"].append(
                    dict(
                        updates=update,
                        examples=a["examples"],
                        training=compact(metrics["training"]),
                        synthetic=compact(metrics["synthetic"]),
                        retention={k: compact(metrics[k]) for k in ("old_known", "old_renamed")},
                        development={
                            k: compact(metrics[k]) for k in shared if k.startswith("development")
                        },
                        last_batch_loss=loss if update else None,
                    )
                )
                if update == 0 and any(
                    metrics[k]["exact"] != 1 for k in ("old_known", "old_renamed")
                ):
                    raise RuntimeError("Missing initial old competence")
                if update in spec["stable_updates"]:
                    path = out / f"{arm}_{update}.pt"
                    owner.save(path)
                    a["checkpoints"][str(update)] = file_hash(path)
                a["final"] = metrics
                r["arms"][arm] = a
                store()
            if update == 1200:
                break
            loss, ix = advance(owner, batch)
            a["updates"] += 1
            a["examples"] += len(ix)
            a["tokens"] += int((batch[0][ix] != 256).sum())
            if update == 0:
                a["first_update_active_parameters"] = sum(
                    p.numel() for p in owner.model.parameters() if p.grad is not None
                )
                a["gradient_norms"] = {
                    n: float(p.grad.norm())
                    for n, p in owner.model.named_parameters()
                    if p.grad is not None
                }
        torch.cuda.synchronize()
        a["synchronized_wall_seconds"] = time.perf_counter() - start
        a["final_hash"] = digest(owner.model.state_dict())
        a["lifetime_steps"] = owner.metadata["steps"]
        a["adam_counters"] = {
            n: int(owner.optimizer.state[p]["step"])
            for n, p in owner.model.named_parameters()
            if p in owner.optimizer.state
        }
        a["stable_acquisition"] = stable_acquisition(
            a["curves"], {row.family for row in sets["training"]}
        )
        a["stable_old_observations"] = stable_old(a["curves"])
        a["development_admission"] = all(
            dev_gate(a["final"][k]) for k in shared if k.startswith("development")
        )
        if (
            arm == "reference"
            and a["final_hash"]
            != json.loads(Path("data/manifests/a5_rehearsal_v2.json").read_text())["arms"][
                "synthetic_6"
            ]["final_hash"]
        ):
            raise RuntimeError("Reference no longer reproduces historical A5 learning")
        store()
        del owner
    r["baselines"] = controls(training, plans, newcounts, shared, out)
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--resume-check",
            str(out / "coverage_1200.pt"),
            "--output-dir",
            str(out),
        ],
        check=True,
        timeout=120,
    )
    expected = json.loads((out / "resume_child.json").read_text())
    owner = LearningState.load(out / "coverage_1200.pt")
    pred = evaluate(owner.model, batches["coverage"], training["coverage"])
    loss, ix = advance(owner, batches["coverage"])
    if expected != dict(
        predictions=pred,
        loss=loss,
        indices=ix,
        signature=signature(owner),
        steps=owner.metadata["steps"],
    ):
        raise RuntimeError("Fresh process continuation differs")
    r["fresh_process_continuation"] = dict(
        exact=True,
        disposable_updates=2,
        candidate_unchanged=file_hash(out / "coverage_1200.pt")
        == r["arms"]["coverage"]["checkpoints"]["1200"],
    )
    primary = r["arms"]["coverage"]
    r["qualification_admission"] = all(
        primary[k]
        for k in ("stable_acquisition", "stable_old_observations", "development_admission")
    )
    r["verdict"] = (
        "DEVELOPMENT_PREREQUISITES_PASS_SEPARATE_QUALIFICATION_REQUIRED"
        if r["qualification_admission"]
        else "PREREQUISITE_FAIL_STOP"
    )
    r["parent_unchanged"] = (
        file_hash(root) == ROOT_HASH and digest(parent.model.state_dict()) == ROOT_TENSOR_HASH
    )
    if not r["parent_unchanged"]:
        raise RuntimeError("Parent changed")
    r["input"] = dict(
        context=context,
        complete_sequences=sum(map(len, sequences.values())),
        max_tokens=max(len(ids) for seq in sequences.values() for ids in seq),
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
