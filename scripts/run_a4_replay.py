"""REMOTE-ONLY matched old-training replay feasibility. No sealed evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from annotate_frame_a2_diagnostics import signature
from frame_a2_common import (
    OLD_DATA_HASH,
    ROOT_HASH,
    ROOT_TENSOR_HASH,
    fork_owner,
    record_stage_provenance,
    state_digest,
)
from run_attribute_learning import encode, evaluate, file_hash, scores, words
from run_foundational_audit import digest
from run_stable_attributes import compact, enrich, stable

from adaptive_mini_gpt.data.attribute_diagnostic import factorial
from adaptive_mini_gpt.data.attribute_learning import TAG_IDS, AttributeCase, generate
from adaptive_mini_gpt.data.frame_curriculum import evaluation_sets
from adaptive_mini_gpt.data.replay_a4 import (
    OLD_FIXTURE_SHA,
    development,
    normalize,
    replace_tail,
    validate_replay,
)
from adaptive_mini_gpt.data.stable_attributes import corpus, minimal_pairs, streams
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103
from adaptive_mini_gpt.training.learning_state import (
    LearningState,
    deterministic_numerical_policy,
    require_numerical_policy,
)
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda

STEPS = 1200
SEED = 42
COUNTS = (0, 3, 6)


def prepare():
    fixture = Path("data/fixtures/stage_a_original_tiny.json")
    if file_hash(fixture) != OLD_FIXTURE_SHA:
        raise ValueError("Wrong historical replay fixture")
    raw = json.loads(fixture.read_text())
    if raw["source"] != "216c3d5 Stage A training subset only":
        raise ValueError("Replay lineage mismatch")
    old = [AttributeCase(**v) for v in raw["cases"]]
    new = corpus()
    dev = development()
    previous = factorial()
    digest_old = hashlib.sha256(
        json.dumps(
            {s: [asdict(v) for v in rows] for s, rows in previous.items()}, sort_keys=True
        ).encode()
    ).hexdigest()
    if digest_old != OLD_DATA_HASH:
        raise ValueError("Previously viewed probe dataset changed")
    blocked = [r.question for values in previous.values() for r in values]
    blocked += [r.question for r in minimal_pairs()]
    blocked += [r.question for r in new]
    blocked += [r.question for r in dev]
    blocked += [
        v["question"]
        for v in json.loads(Path("data/fixtures/stage_a_prior_state_tiny.json").read_text())[
            "cases"
        ]
    ]
    blocked += [r.question for values in evaluation_sets().values() for r in values]
    older = generate()
    blocked += [
        r.question for k, values in older.items() if k not in ("train", "tiny") for r in values
    ]
    validate_replay(old, blocked)
    # Fresh development cases cannot overlap either training support or previously viewed cells.
    old_questions = {normalize(r.question) for r in old}
    new_questions = {normalize(r.question) for r in new}
    dev_questions = {normalize(r.question) for r in dev}
    old_probes = {normalize(r.question) for values in previous.values() for r in values}
    old_probes.update(
        normalize(r.question) for values in evaluation_sets().values() for r in values
    )
    old_probes.update(
        normalize(v["question"])
        for v in json.loads(Path("data/fixtures/stage_a_prior_state_tiny.json").read_text())[
            "cases"
        ]
    )
    old_probes.update(
        normalize(r.question)
        for k, values in older.items()
        if k not in ("train", "tiny")
        for r in values
    )
    if dev_questions & (old_questions | new_questions | old_probes):
        raise ValueError("Development overlap")
    original = streams(new, SEED, STEPS)["augmented"]
    lookup = {tuple(r.statuses): i for i, r in enumerate(old)}
    plans = {
        str(n): [replace_tail(ix, new, lookup, len(new), n) for ix in original] for n in COUNTS
    }
    for step in range(STEPS):
        expected = [tuple(new[i].statuses) for i in original[step]]
        for n in COUNTS:
            observed = [tuple((new + old)[i].statuses) for i in plans[str(n)][step]]
            if observed != expected:
                raise ValueError("Replay changed label/status distribution")
    return (
        new,
        old,
        dev,
        previous,
        plans,
        dict(
            fixture_sha256=OLD_FIXTURE_SHA,
            blocked_question_sha256=hashlib.sha256(
                json.dumps(sorted(set(map(normalize, blocked)))).encode()
            ).hexdigest(),
            blocked_count=len(set(map(normalize, blocked))),
            historical_old_training=True,
        ),
    )


def advance(owner, batch):
    cursor = owner.external["a4_cursor"]
    plan = owner.external["a4_indices"]
    ix = plan[cursor % len(plan)]
    owner.external["a4_cursor"] = cursor + 1
    return owner.step(*[v[ix] for v in batch], attribute_ids=TAG_IDS), ix


def resume(path, out):
    owner = LearningState.load(path)
    new, old, _, _, _, _ = prepare()
    batch, _ = encode(
        new + old, ByteTextTokenizerM103(), owner.model.student.base_model.config.context_length
    )
    pred = evaluate(owner.model, tuple(v[: len(new)] for v in batch), new)
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
        raise ValueError("Wrong immutable parent")
    inspection = json.loads(Path("data/manifests/a4_localization_v1.json").read_text())
    if (
        inspection["status"] != "COMPLETE"
        or inspection["optimizer_steps"] != 0
        or inspection["pre_file_sha256"] != ROOT_HASH
        or inspection["post_file_sha256"]
        != "02dcbe3feedd975df34222b2a5a38e0481bedc7a52fe56dff7d655bb14c7adf3"
        or not inspection["checkpoint_files_unchanged"]
    ):
        raise ValueError("Complete inference-only localization is required first")
    parent = LearningState.load(parent_path)
    if digest(parent.model.state_dict()) != ROOT_TENSOR_HASH or parent.scheduler is not None:
        raise ValueError("Wrong parent state")
    new, old, dev, previous, plans, audit = prepare()
    tokenizer = ByteTextTokenizerM103()
    context = parent.model.student.base_model.config.context_length
    batch, seqs = encode(new + old, tokenizer, context)
    new_batch, new_seqs = encode(new, tokenizer, context)
    old_training_batch = tuple(v[len(new) :] for v in batch)
    if any(not torch.equal(a[: len(new)], b) for a, b in zip(batch, new_batch, strict=True)):
        raise ValueError("Combined padding changed the control stream")
    dev_batch, dev_seqs = encode(dev, tokenizer, context)
    oldsets = {
        "old_known": previous["familiar_familiar"],
        "old_renamed": previous["novel_familiar"],
    }
    old_batches = {s: encode(rows, tokenizer, context)[0] for s, rows in oldsets.items()}
    initial_old = {
        s: enrich(evaluate(parent.model, old_batches[s], rows)) for s, rows in oldsets.items()
    }
    if min(v["exact"] for v in initial_old.values()) != 1:
        raise ValueError("Old competence not established")
    dataset = dict(
        new=[asdict(r) for r in new],
        old_training=[asdict(r) for r in old],
        development=[asdict(r) for r in dev],
        plans=plans,
        audit=audit,
    )
    data_hash = hashlib.sha256(json.dumps(dataset, sort_keys=True).encode()).hexdigest()
    (out / "dataset.json").write_text(json.dumps(dataset))
    (out / "tokens.json").write_text(
        json.dumps(dict(new_old=seqs, development=dev_seqs, new_only=new_seqs))
    )
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    r: dict[str, Any] = dict(
        provenance=dict(
            git_sha=source,
            platform=require_remote_cuda(),
            python=platform.python_version(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),
        ),
        data_sha256=data_hash,
        parent_sha256=ROOT_HASH,
        replay_audit=audit,
        initial_old=initial_old,
        initial_old_training=enrich(evaluate(parent.model, old_training_batch, old)),
        initial_new=enrich(evaluate(parent.model, new_batch, new)),
        initial_development=enrich(evaluate(parent.model, dev_batch, dev)),
        arms={},
        baselines={},
        protocol=json.loads(Path("data/specs/replay_a4_v1.json").read_text()),
        sealed_generated=False,
        promoted=False,
    )

    def store():
        (out / "results.json").write_text(json.dumps(r, indent=2))

    store()
    for count in COUNTS:
        owner = fork_owner(parent, SEED)
        initial_optimizer = state_digest(owner.optimizer.state_dict())
        record_stage_provenance(
            owner,
            training_sha=source,
            publication_sha=source,
            parent_hash=ROOT_HASH,
            data_hash=data_hash,
            stage="A.4_replay",
        )
        owner.metadata.update(
            numerical_policy=deterministic_numerical_policy(),
            a4_replay_count=count,
            supervision_schema=dict(attribute_ids=TAG_IDS, state_targets=None),
            seed=SEED,
        )
        owner.external.update(
            a4_indices=plans[str(count)],
            a4_cursor=0,
            a4_replay_training=[asdict(v) for v in old] if count else [],
            a4_replay_audit=audit,
        )
        a: dict[str, Any] = dict(
            initial_hash=digest(owner.model.state_dict()),
            initial_optimizer_hash=initial_optimizer,
            stored_parameters=sum(p.numel() for p in owner.model.parameters()),
            trainable_parameters=sum(
                p.numel() for p in owner.model.parameters() if p.requires_grad
            ),
            updates=0,
            examples=0,
            new_presentations=0,
            replay_presentations=0,
            tokens=0,
            curves=[],
        )
        torch.cuda.synchronize()
        start = time.perf_counter()
        for step in range(STEPS):
            loss, ix = advance(owner, batch)
            a["updates"] += 1
            a["examples"] += len(ix)
            a["replay_presentations"] += sum(i >= len(new) for i in ix)
            a["new_presentations"] += sum(i < len(new) for i in ix)
            a["tokens"] += int((batch[0][ix] != 256).sum())
            if (step + 1) % 100 == 0:
                a["curves"].append(
                    dict(
                        updates=step + 1,
                        examples=a["examples"],
                        loss=loss,
                        training=compact(enrich(evaluate(owner.model, new_batch, new))),
                        old_training=compact(
                            enrich(evaluate(owner.model, old_training_batch, old))
                        ),
                        development=compact(enrich(evaluate(owner.model, dev_batch, dev))),
                        retention={
                            s: compact(enrich(evaluate(owner.model, old_batches[s], rows)))
                            for s, rows in oldsets.items()
                        },
                    )
                )
        a["final_new"] = enrich(evaluate(owner.model, new_batch, new))
        a["final_old_training"] = enrich(evaluate(owner.model, old_training_batch, old))
        a["final_development"] = enrich(evaluate(owner.model, dev_batch, dev))
        a["retention"] = {
            s: enrich(evaluate(owner.model, old_batches[s], rows)) for s, rows in oldsets.items()
        }
        a["stable_acquisition"] = stable(a)
        a["retention_admission"] = min(v["exact"] for v in a["retention"].values()) >= 0.95
        m = a["final_development"]
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
        owner.save(out / f"replay_{count}.pt")
        a["checkpoint_sha256"] = file_hash(out / f"replay_{count}.pt")
        torch.cuda.synchronize()
        a["gpu_wall_seconds"] = time.perf_counter() - start
        r["arms"][str(count)] = a
        store()
        del owner
    a3 = json.loads(Path("data/manifests/stable_attributes_v2.json").read_text())["arms"][
        "augmented"
    ]["final_hash"]
    r["no_replay_matches_a3_weight_hash"] = r["arms"]["0"]["final_hash"] == a3
    if not r["no_replay_matches_a3_weight_hash"]:
        raise AssertionError("No-replay control did not reproduce A.3; comparisons invalid")
    for kind in ("bow_linear", "ngram_linear"):
        for count in COUNTS:
            delivered = [(new + old)[i] for ix in plans[str(count)] for i in ix]
            vocab = sorted({w for row in delivered for w in words(row, kind)})
            allrows = new + old + dev + sum(oldsets.values(), [])
            counters = [__import__("collections").Counter(words(row, kind)) for row in allrows]
            x = torch.tensor([[float(c[w]) for w in vocab] for c in counters], device="cuda")
            y = torch.tensor([row.labels for row in allrows], device="cuda", dtype=torch.float32)
            torch.manual_seed(SEED)
            model = torch.nn.Linear(len(vocab), 2).cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for ix in plans[str(count)]:
                opt.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(model(x[ix]), y[ix])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
            metrics = {}
            offset = len(new) + len(old)
            for name, rows in [("development", dev), *oldsets.items()]:
                sl = slice(offset, offset + len(rows))
                metrics[name] = enrich(scores(model(x[sl]).detach(), y[sl], rows))
                offset += len(rows)
            metrics["new_training"] = enrich(
                scores(model(x[: len(new)]).detach(), y[: len(new)], new)
            )
            torch.cuda.synchronize()
            r["baselines"][f"{kind}_{count}"] = dict(
                evaluation=metrics,
                parameters=sum(v.numel() for v in model.parameters()),
                updates=STEPS,
                examples=STEPS * 18,
                gpu_wall_seconds=time.perf_counter() - start,
            )
            store()
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--resume-check",
            str(out / "replay_6.pt"),
            "--output-dir",
            str(out),
        ],
        check=True,
        timeout=120,
    )
    actual = json.loads((out / "resume_child.json").read_text())
    owner = LearningState.load(out / "replay_6.pt")
    pred = evaluate(owner.model, new_batch, new)
    loss, ix = advance(owner, batch)
    assert actual == dict(
        predictions=pred,
        loss=loss,
        indices=ix,
        signature=signature(owner),
        steps=owner.metadata["steps"],
    )
    r["fresh_process_continuation"] = dict(
        exact=True,
        disposable_extra_updates=2,
        candidate_unchanged=file_hash(out / "replay_6.pt") == r["arms"]["6"]["checkpoint_sha256"],
    )
    primary = r["arms"]["6"]
    r["qualification_admission"] = all(
        primary[k] for k in ("stable_acquisition", "retention_admission", "development_admission")
    )
    r["verdict"] = (
        "DEVELOPMENT_READY_FOR_SEPARATE_SEALED_PROTOCOL"
        if r["qualification_admission"]
        else "REPLAY_GATE_FAIL_STOP"
    )
    r["input"] = dict(
        new_training=len(new),
        old_training=len(old),
        development=len(dev),
        max_tokens=max(map(len, seqs + dev_seqs)),
        context=context,
    )
    assert digest(parent.model.state_dict()) == ROOT_TENSOR_HASH
    store()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root")
    p.add_argument("--resume-check")
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    require_remote_cuda()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Workspace must be set before process start")
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
