"""Remote bounded Stage A.3 tiny acquisition; no sealed cases or promotion."""

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
    intervene_backbone,
    record_stage_provenance,
    state_digest,
)
from run_attribute_learning import encode, evaluate, file_hash, scores, words
from run_foundational_audit import digest
from run_frame_a2_diagnosis import position_evaluation, ridge_readout
from run_stage_a_diagnostic import gradient_audit

from adaptive_mini_gpt.data.attribute_diagnostic import factorial
from adaptive_mini_gpt.data.attribute_learning import TAG_IDS
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
ARMS = (
    "fixed",
    "augmented",
    "paired",
    "backbone_moments",
    "backbone_age",
    "backbone_complete",
    "head_moments",
    "rebase",
)
PAIRS = tuple((i, i + 1) for i in range(0, 18, 2))


def compact(m):
    return {k: v for k, v in m.items() if k not in ("predictions", "truth", "logits")}


def enrich(m):
    m["per_property"] = []
    for j in range(2):
        tp = sum(t[j] == p[j] == 1 for t, p in zip(m["truth"], m["predictions"], strict=True))
        tn = sum(t[j] == p[j] == 0 for t, p in zip(m["truth"], m["predictions"], strict=True))
        fp = sum(t[j] == 0 and p[j] == 1 for t, p in zip(m["truth"], m["predictions"], strict=True))
        fn = sum(t[j] == 1 and p[j] == 0 for t, p in zip(m["truth"], m["predictions"], strict=True))
        m["per_property"].append(
            dict(
                tp=tp,
                tn=tn,
                fp=fp,
                fn=fn,
                precision=tp / max(1, tp + fp),
                recall=tp / max(1, tp + fn),
                negative_recall=tn / max(1, tn + fp),
            )
        )
    m["bce"] = sum(
        max(x, 0) - x * y + math.log1p(math.exp(-abs(x)))
        for logit, t in zip(m["logits"], m["truth"], strict=True)
        for x, y in zip(logit, t, strict=True)
    ) / (2 * len(m["truth"]))
    return m


def inspect(model, batch, rows):
    model.eval()
    result: dict[str, Any] = {
        k: [] for k in ("pooled", "concept", "last_observation", "logits", "pool_masks")
    }
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            tokens = batch[0][start : start + 32]
            hidden = model.extract_sequence_hidden_states(tokens)
            pooled, _ = model.pool_representations(hidden, tokens)
            concept = model.bottleneck.concept_proj(pooled)
            logits = model.bottleneck.attribute_head(concept)[:, TAG_IDS]
            result["pooled"].extend(pooled.tolist())
            result["concept"].extend(concept.tolist())
            result["logits"].extend(logits.tolist())
            for j, ids in enumerate(tokens.tolist()):
                q, a = ids.index(261), ids.index(262)
                result["last_observation"].append(hidden[j, a - 1].tolist())
                result["pool_masks"].append(dict(start=q, end=a, last_observation=a - 1))
    for kind in ("pooled", "concept", "last_observation"):
        z = F.normalize(torch.tensor(result[kind], device="cuda"), dim=-1)
        similarities = (z @ z.T).cpu().tolist()
        same = []
        different = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                value = similarities[i][j]
                (same if rows[i].statuses == rows[j].statuses else different).append(value)
        result[kind + "_cosine"] = dict(
            same_status_mean=sum(same) / len(same),
            different_status_mean=sum(different) / len(different),
        )
    return result


def kwargs(arm):
    return dict(
        attribute_ids=TAG_IDS,
        **(dict(consistency_pairs=PAIRS, consistency_weight=0.1) if arm == "paired" else {}),
    )


def next_update(owner, batch):
    cursor = owner.external["a3_cursor"]
    sequence = owner.external["a3_indices"]
    ix = sequence[cursor % len(sequence)]
    owner.external["a3_cursor"] = cursor + 1
    loss = owner.step(*[v[ix] for v in batch], **kwargs(owner.metadata["a3_arm"]))
    return loss, ix


def stable(a):
    return all(
        c["training"]["exact"] >= 0.99
        and min(c["training"]["family_exact"].values()) >= 0.99
        and min(c["training"]["status_accuracy"].values()) >= 0.98
        for c in a["curves"][-3:]
    )


def resume_check(path, output):
    owner = LearningState.load(path)
    rows = corpus()
    batch, _ = encode(
        rows, ByteTextTokenizerM103(), owner.model.student.base_model.config.context_length
    )
    predictions = evaluate(owner.model, batch, rows)
    loss, ix = next_update(owner, batch)
    output.write_text(
        json.dumps(
            dict(
                predictions=predictions,
                loss=loss,
                indices=ix,
                signature=signature(owner),
                steps=owner.metadata["steps"],
            )
        )
    )


def run(out, parent_path):
    if file_hash(parent_path) != ROOT_HASH:
        raise ValueError("Wrong persistent parent; no fresh fallback")
    parent = LearningState.load(parent_path)
    if digest(parent.model.state_dict()) != ROOT_TENSOR_HASH or parent.scheduler is not None:
        raise ValueError("Wrong weight/scheduler contract")
    rows = corpus()
    plans = streams(rows, SEED, STEPS)
    mini = minimal_pairs()
    tokenizer = ByteTextTokenizerM103()
    context = parent.model.student.base_model.config.context_length
    batch, seqs = encode(rows, tokenizer, context)
    mbatch, mseqs = encode(mini, tokenizer, context)
    old = factorial()
    retention = {k: old[k] for k in ("familiar_familiar", "novel_familiar")}
    rb = {k: encode(v, tokenizer, context)[0] for k, v in retention.items()}
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dataset = dict(
        cases=[asdict(r) for r in rows], minimal_pairs=[asdict(r) for r in mini], streams=plans
    )
    data_hash = hashlib.sha256(json.dumps(dataset, sort_keys=True).encode()).hexdigest()
    numerical_protocol = json.loads(
        Path("data/specs/stable_attributes_numerics_v1.json").read_text()
    )
    if data_hash != numerical_protocol["unchanged_data_sha256"]:
        raise ValueError("Numerical repair must not change study data")
    (out / "dataset.json").write_text(json.dumps(dataset))
    (out / "tokens.json").write_text(json.dumps(dict(training=seqs, minimal_pairs=mseqs)))
    initial_retention = {s: evaluate(parent.model, rb[s], v) for s, v in retention.items()}
    assert min(m["exact"] for m in initial_retention.values()) == 1
    r = dict(
        provenance=dict(
            git_sha=source,
            platform=require_remote_cuda(),
            python=platform.python_version(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),
        ),
        parent_sha256=ROOT_HASH,
        numerical_policy=deterministic_numerical_policy(),
        numerical_protocol=numerical_protocol,
        data_sha256=data_hash,
        protocol=json.loads(Path("data/specs/stable_attributes_v1.json").read_text()),
        initial_retention=initial_retention,
        initial=enrich(evaluate(parent.model, batch, rows)),
        minimal_pairs=enrich(evaluate(parent.model, mbatch, mini)),
        positions={
            str(k): enrich(position_evaluation(parent.model, mbatch, mini, k)) for k in (0, 4, 16)
        },
        arms={},
        baselines={},
        sealed_generated=False,
        promoted=False,
    )
    before = inspect(parent.model, batch, rows)
    (out / "initial_representations.json").write_text(json.dumps(before))
    (out / "minimal_pair_representations.json").write_text(
        json.dumps(inspect(parent.model, mbatch, mini))
    )
    r["initial_ridge_full_development_fit"] = {
        kind: enrich(
            scores(
                ridge_readout(before[kind], batch[2], dict(training=before[kind]))["training"],
                batch[2][:, TAG_IDS],
                rows,
            )
        )
        for kind in ("pooled", "concept", "last_observation")
    }

    def store():
        (out / "results.json").write_text(json.dumps(r, indent=2))

    store()
    for arm in ARMS:
        owner = fork_owner(parent, SEED)
        initial_optimizer = state_digest(owner.optimizer.state_dict())
        affected = []
        if arm.startswith("backbone_"):
            affected = intervene_backbone(owner, arm.removeprefix("backbone_"))
        if arm == "head_moments":
            for n, p in owner.model.named_parameters():
                if (
                    n.startswith(("bottleneck.concept_proj.", "bottleneck.attribute_head."))
                    and p in owner.optimizer.state
                ):
                    affected.append(n)
                    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        if key in owner.optimizer.state[p]:
                            owner.optimizer.state[p][key].zero_()
        record_stage_provenance(
            owner,
            training_sha=source,
            publication_sha=source,
            parent_hash=ROOT_HASH,
            data_hash=data_hash,
            stage="A.3_development",
        )
        owner.metadata.update(
            a3_arm=arm,
            numerical_policy=deterministic_numerical_policy(),
            supervision_schema=dict(
                attribute_ids=TAG_IDS,
                state_targets=None,
                consistency_weight=0.1 if arm == "paired" else 0,
                consistency_pairs=PAIRS if arm == "paired" else None,
            ),
            seed=SEED,
        )
        owner.external.update(
            a3_indices=plans["fixed" if arm == "fixed" else "augmented"], a3_cursor=0
        )
        a: dict[str, Any] = dict(
            initial_hash=digest(owner.model.state_dict()),
            initial_optimizer_hash=initial_optimizer,
            post_intervention_optimizer_hash=state_digest(owner.optimizer.state_dict()),
            affected_parameters=affected,
            stored_parameters=sum(p.numel() for p in owner.model.parameters()),
            trainable_parameters=sum(
                p.numel() for p in owner.model.parameters() if p.requires_grad
            ),
            curves=[],
            updates=0,
            examples=0,
            tokens=0,
        )
        support = sorted({i for ix in owner.external["a3_indices"] for i in ix})
        support_rows = [rows[i] for i in support]
        support_batch = tuple(v[support] for v in batch)
        a["unique_delivered_training_cases"] = len(support)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for step in range(STEPS):
            loss, ix = next_update(owner, batch)
            a["updates"] += 1
            a["examples"] += len(ix)
            a["tokens"] += int((batch[0][ix] != 256).sum())
            if (step + 1) % 100 == 0:
                a["curves"].append(
                    dict(
                        updates=step + 1,
                        examples=a["examples"],
                        batch_total_loss=loss,
                        training=compact(enrich(evaluate(owner.model, batch, rows))),
                        retention={
                            s: compact(enrich(evaluate(owner.model, rb[s], v)))
                            for s, v in retention.items()
                        },
                    )
                )
        a["final"] = enrich(evaluate(owner.model, batch, rows))
        a["delivered_training"] = enrich(evaluate(owner.model, support_batch, support_rows))
        a["retention"] = {s: enrich(evaluate(owner.model, rb[s], v)) for s, v in retention.items()}
        a["minimal_pairs"] = enrich(evaluate(owner.model, mbatch, mini))
        a["gradient"] = gradient_audit(owner, batch)
        features = inspect(owner.model, batch, rows)
        (out / f"{arm}_representations.json").write_text(json.dumps(features))
        a["feature_geometry"] = {k: v for k, v in features.items() if k.endswith("_cosine")}
        a["ridge_full_development_fit"] = {
            kind: enrich(
                scores(
                    ridge_readout(features[kind], batch[2], dict(training=features[kind]))[
                        "training"
                    ],
                    batch[2][:, TAG_IDS],
                    rows,
                )
            )
            for kind in ("pooled", "concept", "last_observation")
        }
        a["stable_acquisition"] = stable(a)
        a["retention_admission"] = min(m["exact"] for m in a["retention"].values()) >= 0.95
        a["final_hash"] = digest(owner.model.state_dict())
        a["lifetime_updates"] = owner.metadata["steps"]
        a["adam_counters"] = {
            n: int(owner.optimizer.state[p]["step"])
            for n, p in owner.model.named_parameters()
            if p in owner.optimizer.state
        }
        path = out / f"{arm}.pt"
        owner.save(path)
        a["checkpoint_sha256"] = file_hash(path)
        torch.cuda.synchronize()
        a["gpu_wall_seconds"] = time.perf_counter() - start
        r["arms"][arm] = a
        store()
        del owner
    assert r["arms"]["augmented"]["final_hash"] == r["arms"]["rebase"]["final_hash"]
    r["rebase_exact"] = True
    for kind in ("bow_linear", "ngram_linear"):
        for surface in ("fixed", "augmented"):
            indices = plans[surface]
            delivered = [rows[i] for ix in indices for i in ix]
            vocab = sorted({w for row in delivered for w in words(row, kind)})
            counts = [__import__("collections").Counter(words(row, kind)) for row in rows]
            x = torch.tensor([[float(c[w]) for w in vocab] for c in counts], device="cuda")
            y = batch[2][:, TAG_IDS]
            torch.manual_seed(SEED)
            control = torch.nn.Linear(len(vocab), 2).cuda()
            opt = torch.optim.AdamW(control.parameters(), lr=0.002, weight_decay=0)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for ix in indices:
                opt.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(control(x[ix]), y[ix])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(control.parameters(), 1)
                opt.step()
            with torch.no_grad():
                metric = enrich(scores(control(x), y, rows))
            torch.cuda.synchronize()
            r["baselines"][kind + "_" + surface] = dict(
                evaluation=metric,
                parameters=sum(p.numel() for p in control.parameters()),
                updates=STEPS,
                examples=STEPS * 18,
                gpu_wall_seconds=time.perf_counter() - start,
            )
            store()
    child = out / "resume_child.json"
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--resume-check",
            str(out / "paired.pt"),
            "--output-dir",
            str(out),
        ],
        check=True,
        timeout=120,
    )
    actual = json.loads(child.read_text())
    owner = LearningState.load(out / "paired.pt")
    expected_predictions = evaluate(owner.model, batch, rows)
    loss, ix = next_update(owner, batch)
    assert actual == dict(
        predictions=expected_predictions,
        loss=loss,
        indices=ix,
        signature=signature(owner),
        steps=owner.metadata["steps"],
    )
    r["fresh_process_continuation"] = dict(
        exact=True,
        disposable_extra_updates=2,
        candidate_checkpoint_unchanged=file_hash(out / "paired.pt")
        == r["arms"]["paired"]["checkpoint_sha256"],
    )
    r["qualification_admission"] = (
        r["arms"]["paired"]["stable_acquisition"] and r["arms"]["paired"]["retention_admission"]
    )
    r["verdict"] = (
        "DEVELOPMENT_READY_FOR_SEPARATE_SEALED_PROTOCOL"
        if r["qualification_admission"]
        else (
            "PARTIAL_DEVELOPMENT_ONLY_STOP"
            if any(a["stable_acquisition"] for a in r["arms"].values())
            else "FAIL_ACQUISITION_UNSTABLE_STOP"
        )
    )
    assert digest(parent.model.state_dict()) == ROOT_TENSOR_HASH
    r["input"] = dict(
        unique_training=len(rows),
        minimal_pairs=len(mini),
        max_tokens=max(map(len, seqs + mseqs)),
        context=context,
        training_presentations_per_arm=STEPS * 18,
    )
    store()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume-check")
    a = p.parse_args()
    require_remote_cuda()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Set declared CUBLAS workspace before starting the remote process")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    require_numerical_policy(deterministic_numerical_policy())
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if a.resume_check:
        resume_check(a.resume_check, out / "resume_child.json")
    elif a.root:
        run(out, Path(a.root))
    else:
        p.error("--root required unless --resume-check")


if __name__ == "__main__":
    main()
