"""REMOTE-ONLY Stage A.5 gate1; stop before rehearsal if joint acquisition fails."""

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
from run_concept_binding import language_model
from run_foundational_audit import digest
from run_stable_attributes import compact, enrich

from adaptive_mini_gpt.data.attribute_diagnostic import factorial
from adaptive_mini_gpt.data.attribute_learning import TAG_IDS, generate
from adaptive_mini_gpt.data.frame_curriculum import evaluation_sets
from adaptive_mini_gpt.data.joint_acquisition_a5 import (
    NAMES,
    corpus,
    plan,
    stable_joint,
    validate_disjoint,
)
from adaptive_mini_gpt.data.replay_a4 import development
from adaptive_mini_gpt.data.stable_attributes import corpus as prior_corpus
from adaptive_mini_gpt.data.stable_attributes import minimal_pairs
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103
from adaptive_mini_gpt.training.learning_state import (
    LearningState,
    deterministic_numerical_policy,
    require_numerical_policy,
)
from adaptive_mini_gpt.training.optimizer_contract import optimizer_membership
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda
from adaptive_mini_gpt.training.reproducibility import seed_all

SPEC = Path("data/specs/joint_acquisition_a5_v1.json")


def prepare():
    previous = factorial()
    older = generate()
    a2 = evaluation_sets()
    protected = [
        r.question for sets in (previous, older, a2) for rows in sets.values() for r in rows
    ]
    protected += [r.question for r in prior_corpus() + minimal_pairs() + development()]
    for path in ("stage_a_original_tiny.json", "stage_a_prior_state_tiny.json"):
        protected += [
            r["question"] for r in json.loads(Path("data/fixtures", path).read_text())["cases"]
        ]
    rows = corpus()
    validate_disjoint(rows, protected)
    if len(rows) != 210 or {r.name for r in rows} != set(NAMES):
        raise ValueError("Unexpected joint acquisition support")
    protocol = json.loads(SPEC.read_text())
    indices = plan(rows, protocol["seed"], protocol["updates"])
    exposure = Counter(i for batch in indices for i in batch)
    if set(exposure) != set(range(len(rows))):
        raise ValueError("A training case is never delivered")
    status_counts = Counter(
        (rows[i].group, tuple(rows[i].statuses)) for batch in indices for i in batch
    )
    frame_counts = Counter(
        (rows[i].frame, tuple(rows[i].statuses)) for batch in indices for i in batch
    )
    if set(status_counts.values()) != {1200} or any(
        n != (600 if f < 2 else 400) for (f, _), n in frame_counts.items()
    ):
        raise ValueError("Unmatched status/grammar exposure")
    audit = dict(
        protected_count=len(set(protected)),
        protected_sha256=hashlib.sha256(json.dumps(sorted(set(protected))).encode()).hexdigest(),
        unique_examples=len(exposure),
        exposure_by_case=[exposure[i] for i in range(len(rows))],
        status_counts={str(k): v for k, v in status_counts.items()},
        frame_counts={str(k): v for k, v in frame_counts.items()},
        all_labels_independently_verified=True,
        historical_replay=False,
    )
    return rows, indices, previous, development(), audit


def next_update(owner, batch):
    cursor = owner.external["a5_cursor"]
    indices = owner.external["a5_indices"]
    ix = indices[cursor % len(indices)]
    owner.external["a5_cursor"] = cursor + 1
    return owner.step(*[v[ix] for v in batch], attribute_ids=TAG_IDS), ix


def gradient_probe(owner, batch, first_indices):
    """Training-only, eval-mode gradients; no optimizer step or buffer/RNG change."""
    model = owner.model
    before = signature(owner)
    modes = {n: m.training for n, m in model.named_modules()}
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grads, losses = [], []
    try:
        model.eval()
        for indices in (first_indices[::2], first_indices[1::2]):
            logits = model(batch[0][indices], batch[1][indices])[1][:, TAG_IDS]
            per_label = F.binary_cross_entropy_with_logits(
                logits, batch[2][indices][:, TAG_IDS], reduction="none"
            ).mean(0)
            losses.append(per_label.detach().tolist())
            grads.append(
                torch.autograd.grad(per_label.mean(), [p for _, p in named], allow_unused=True)
            )
    finally:
        for n, m in model.named_modules():
            m.training = modes[n]
    if signature(owner) != before:
        raise RuntimeError("Gradient diagnostic changed learning state")

    def component(name):
        if name.startswith("student.base_model.wte"):
            return "token_embeddings"
        if name.startswith("student."):
            return "trainable_transformer"
        if "concept_proj" in name:
            return "concept_projection"
        if "attribute_head" in name:
            return "attribute_head"
        return "other"

    result = {}
    for group in sorted({component(n) for n, _ in named}):
        norm_a = norm_b = dot = 0.0
        active = []
        for (n, _p), a, b in zip(named, *grads, strict=True):
            if component(n) != group:
                continue
            if a is not None:
                norm_a += float(a.square().sum())
            if b is not None:
                norm_b += float(b.square().sum())
            if a is not None and b is not None:
                dot += float((a * b).sum())
            if a is not None or b is not None:
                active.append(n)
        denominator = math.sqrt(norm_a * norm_b)
        result[group] = dict(
            old_norm=math.sqrt(norm_a),
            new_norm=math.sqrt(norm_b),
            dot=dot,
            cosine=dot / denominator if denominator else None,
            active_names=active,
        )
    return dict(
        old_per_label_bce=losses[0],
        new_per_label_bce=losses[1],
        components=result,
        state_unchanged=True,
    )


def resume(path, out):
    owner = LearningState.load(path)
    rows, _, _, _, _ = prepare()
    batch, _ = encode(
        rows, ByteTextTokenizerM103(), owner.model.student.base_model.config.context_length
    )
    predictions = evaluate(owner.model, batch, rows)
    loss, indices = next_update(owner, batch)
    out.write_text(
        json.dumps(
            dict(
                predictions=predictions,
                loss=loss,
                indices=indices,
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
        raise ValueError("Wrong parent learning state")
    rows, indices, previous, dev, audit = prepare()
    protocol = json.loads(SPEC.read_text())
    data = dict(training=[asdict(r) for r in rows], indices=indices, audit=audit)
    data_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    (out / "dataset.json").write_text(json.dumps(data))
    tokenizer = ByteTextTokenizerM103()
    context = parent.model.student.base_model.config.context_length
    batch, seqs = encode(rows, tokenizer, context)
    groups = {g: [i for i, r in enumerate(rows) if r.group == g] for g in ("old", "new")}
    sets = {g: [rows[i] for i in ix] for g, ix in groups.items()}
    batches = {g: tuple(v[ix] for v in batch) for g, ix in groups.items()}
    sets.update(
        old_known=previous["familiar_familiar"],
        old_renamed=previous["novel_familiar"],
        development=dev,
    )
    tokens = {"training": seqs}
    for name in ("old_known", "old_renamed", "development"):
        batches[name], tokens[name] = encode(sets[name], tokenizer, context)
    (out / "evaluation_cases.json").write_text(
        json.dumps({k: [asdict(r) for r in v] for k, v in sets.items()})
    )
    (out / "tokens.json").write_text(json.dumps(tokens))
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result: dict[str, Any] = dict(
        provenance=dict(
            git_sha=source,
            platform=require_remote_cuda(),
            python=platform.python_version(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),
        ),
        protocol=protocol,
        parent_sha256=ROOT_HASH,
        data_sha256=data_hash,
        audit=audit,
        arms={},
        baselines={},
        sealed_generated=False,
        promoted=False,
    )

    def store():
        (out / "results.json").write_text(json.dumps(result, indent=2))

    parent_schema = [
        (n, tuple(p.shape), str(p.dtype), p.requires_grad)
        for n, p in parent.model.named_parameters()
    ]
    for arm in ("continuing", "fresh"):
        if arm == "continuing":
            owner = fork_owner(parent, protocol["seed"])
        else:
            seed_all(protocol["seed"])
            owner = LearningState(language_model("selective_plasticity"))
            seed_all(protocol["seed"])
        if [
            (n, tuple(p.shape), str(p.dtype), p.requires_grad)
            for n, p in owner.model.named_parameters()
        ] != parent_schema:
            raise RuntimeError("Control architecture/trainability differs")
        if owner.model.default_strategy != "question_only" or any(
            optimizer_membership(owner.model, owner.optimizer).values()
        ):
            raise RuntimeError("Wrong path/optimizer membership")
        if any(
            g["lr"] != protocol["learning_rate"] or g["weight_decay"] != 0
            for g in owner.optimizer.param_groups
        ):
            raise RuntimeError("Budget mismatch")
        initial_hash = digest(owner.model.state_dict())
        initial_optimizer = state_digest(owner.optimizer.state_dict())
        if arm == "continuing" and initial_optimizer != state_digest(parent.optimizer.state_dict()):
            raise RuntimeError("Inherited Adam changed")
        record_stage_provenance(
            owner,
            training_sha=source,
            publication_sha=source,
            parent_hash=ROOT_HASH if arm == "continuing" else None,
            comparison_hash=ROOT_HASH,
            data_hash=data_hash,
            stage="A.5_joint",
        )
        owner.metadata.update(
            numerical_policy=deterministic_numerical_policy(),
            a5_arm=arm,
            supervision_schema=dict(attribute_ids=TAG_IDS, state_targets=None),
            seed=protocol["seed"],
        )
        owner.external.update(
            a5_indices=indices,
            a5_cursor=0,
            a5_training_manifest=dict(data_sha256=data_hash, audit=audit),
        )
        a: dict[str, Any] = dict(
            initial_hash=initial_hash,
            initial_optimizer_hash=initial_optimizer,
            initial_lifetime_updates=owner.metadata["steps"],
            stored_parameters=sum(p.numel() for p in owner.model.parameters()),
            trainable_parameters=sum(
                p.numel() for p in owner.model.parameters() if p.requires_grad
            ),
            updates=0,
            examples=0,
            tokens=0,
            old_presentations=0,
            new_presentations=0,
            evaluation_rows=0,
            evaluation_forward_batches=0,
            curves=[],
            gradients={},
            checkpoints={},
        )

        def observe(owner=owner, a=a):
            metrics = {
                k: enrich(evaluate(owner.model, batches[k], values)) for k, values in sets.items()
            }
            a["evaluation_rows"] += sum(map(len, sets.values()))
            a["evaluation_forward_batches"] += sum(math.ceil(len(v) / 32) for v in sets.values())
            return metrics

        a["initial"] = observe()
        final = a["initial"]
        if (
            arm == "continuing"
            and min(a["initial"][k]["exact"] for k in ("old_known", "old_renamed")) != 1
        ):
            raise RuntimeError("Prior competence missing")
        torch.cuda.synchronize()
        start = time.perf_counter()
        a["gradients"]["0"] = gradient_probe(owner, batch, indices[0])
        for step in range(protocol["updates"]):
            loss, ix = next_update(owner, batch)
            a["updates"] += 1
            a["examples"] += len(ix)
            a["tokens"] += int((batch[0][ix] != tokenizer.PAD_ID).sum())
            a["old_presentations"] += sum(rows[i].group == "old" for i in ix)
            a["new_presentations"] += sum(rows[i].group == "new" for i in ix)
            if step == 0:
                a["first_update_active_parameters"] = sum(
                    p.numel() for p in owner.model.parameters() if p.grad is not None
                )
            if step + 1 in protocol["gradient_diagnostics"]:
                a["gradients"][str(step + 1)] = gradient_probe(owner, batch, indices[0])
            if (step + 1) % 100 == 0:
                final = observe()
                a["curves"].append(
                    dict(
                        updates=step + 1,
                        examples=a["examples"],
                        last_batch_loss=loss,
                        training={k: compact(final[k]) for k in ("old", "new")},
                        retention={k: compact(final[k]) for k in ("old_known", "old_renamed")},
                        development=compact(final["development"]),
                    )
                )
            if step + 1 in protocol["checkpoint_updates"]:
                file = (
                    f"{arm}.pt" if step + 1 == protocol["updates"] else f"{arm}_step_{step + 1}.pt"
                )
                owner.save(out / file)
                a["checkpoints"][str(step + 1)] = dict(file=file, sha256=file_hash(out / file))
        a["final"] = final
        a["stable_joint_acquisition"] = stable_joint(a["curves"])
        a["old_observation_admission"] = (
            min(final[k]["exact"] for k in ("old_known", "old_renamed")) >= 0.95
        )
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
        torch.cuda.synchronize()
        a["synchronized_wall_seconds"] = time.perf_counter() - start
        result["arms"][arm] = a
        store()
        del owner
    # Text controls are diagnostic; no inherited history or oracle features.
    allrows = rows + sets["old_known"] + sets["old_renamed"] + dev
    ranges = {g: ix for g, ix in groups.items()}
    offset = len(rows)
    for k in ("old_known", "old_renamed", "development"):
        ranges[k] = list(range(offset, offset + len(sets[k])))
        offset += len(sets[k])
    y = torch.tensor([r.labels for r in allrows], device="cuda", dtype=torch.float32)
    for kind in ("majority", "bow_linear", "ngram_linear"):
        seed_all(protocol["seed"])
        torch.cuda.synchronize()
        start = time.perf_counter()
        if kind == "majority":
            delivered = [i for ix in indices for i in ix]
            rate = y[delivered].mean(0).clamp(1e-6, 1 - 1e-6)
            logits = torch.logit(rate).expand_as(y)
            parameters = 0
        else:
            vocab = sorted({w for row in rows for w in words(row, kind)})
            counters = [Counter(words(row, kind)) for row in allrows]
            x = torch.tensor([[float(c[w]) for w in vocab] for c in counters], device="cuda")
            model = torch.nn.Linear(len(vocab), 2).cuda()
            opt = torch.optim.AdamW(
                model.parameters(), lr=protocol["learning_rate"], weight_decay=0
            )
            for ix in indices:
                opt.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(model(x[ix]), y[ix])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            with torch.no_grad():
                logits = model(x)
            parameters = sum(p.numel() for p in model.parameters())
        metrics = {k: enrich(scores(logits[ix], y[ix], sets[k])) for k, ix in ranges.items()}
        torch.cuda.synchronize()
        result["baselines"][kind] = dict(
            metrics=metrics,
            parameters=parameters,
            updates=0 if kind == "majority" else protocol["updates"],
            examples=protocol["presentations"],
            synchronized_wall_seconds=time.perf_counter() - start,
        )
        store()
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--resume-check",
            str(out / "continuing.pt"),
            "--output-dir",
            str(out),
        ],
        check=True,
        timeout=120,
    )
    child = json.loads((out / "resume_child.json").read_text())
    owner = LearningState.load(out / "continuing.pt")
    pred = evaluate(owner.model, batch, rows)
    loss, ix = next_update(owner, batch)
    if child != dict(
        predictions=pred,
        loss=loss,
        indices=ix,
        signature=signature(owner),
        steps=owner.metadata["steps"],
    ):
        raise RuntimeError("Fresh-process continuation mismatch")
    result["fresh_process_continuation"] = dict(
        exact=True,
        disposable_extra_updates=2,
        candidate_unchanged=file_hash(out / "continuing.pt")
        == result["arms"]["continuing"]["checkpoints"]["1200"]["sha256"],
    )
    result["joint_acquisition_admission"] = result["arms"]["continuing"]["stable_joint_acquisition"]
    result["verdict"] = (
        "JOINT_ACQUISITION_READY_FOR_SEPARATE_REHEARSAL_PROTOCOL"
        if result["joint_acquisition_admission"]
        else "JOINT_ACQUISITION_FAIL_STOP"
    )
    result["input"] = dict(
        training=len(rows),
        old_training=len(sets["old"]),
        new_training=len(sets["new"]),
        context=context,
        max_tokens=max(map(len, seqs)),
    )
    if digest(parent.model.state_dict()) != ROOT_TENSOR_HASH or file_hash(parent_path) != ROOT_HASH:
        raise RuntimeError("Parent changed")
    result["parent_unchanged"] = True
    store()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root")
    p.add_argument("--resume-check")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    require_remote_cuda()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Configure deterministic workspace before process start")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    require_numerical_policy(deterministic_numerical_policy())
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.resume_check:
        resume(Path(args.resume_check), out / "resume_child.json")
    elif args.root:
        run(Path(args.root), out)
    else:
        p.error("--root required")


if __name__ == "__main__":
    main()
