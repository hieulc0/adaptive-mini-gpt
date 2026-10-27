"""REMOTE-ONLY conditional frozen-backbone/existing-module feasibility."""

from __future__ import annotations

import argparse
import copy
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
from annotate_frame_a2_diagnostics import signature
from frame_a2_common import (
    ROOT_HASH,
    ROOT_TENSOR_HASH,
    fork_owner,
    record_stage_provenance,
    state_digest,
)
from run_a4_replay import SEED, STEPS, advance, prepare
from run_attribute_learning import encode, evaluate, file_hash
from run_foundational_audit import digest
from run_stable_attributes import compact, enrich, stable

from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103
from adaptive_mini_gpt.training.learning_state import (
    LearningState,
    deterministic_numerical_policy,
    require_numerical_policy,
)
from adaptive_mini_gpt.training.optimizer_contract import optimizer_membership
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda


def modular_t0_admitted(known, renamed):
    return all(math.isfinite(v) and 0 <= v <= 1 and v >= 0.95 for v in (known, renamed))


def state_by_name(owner):
    return {
        n: state_digest(owner.optimizer.state[p])
        for n, p in owner.model.named_parameters()
        if p in owner.optimizer.state
    }


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


def run(path, out):
    reference = json.loads(Path("data/manifests/a4_replay_v1.json").read_text())
    if (
        reference["status"] != "COMPLETE"
        or reference["qualification_admission"]
        or reference["provenance"]["git_sha"] != "4afeffce617d3b32a9b8b7c58c3f22e9a697b562"
    ):
        raise ValueError("Conditional replay prerequisite is absent or not failed")
    if file_hash(path) != ROOT_HASH:
        raise ValueError("Wrong persistent parent")
    parent = LearningState.load(path)
    if digest(parent.model.state_dict()) != ROOT_TENSOR_HASH or parent.scheduler is not None:
        raise ValueError("Wrong parent state")
    new, old, dev, previous, plans, audit = prepare()
    dataset = dict(
        new=[asdict(x) for x in new],
        old_training=[asdict(x) for x in old],
        development=[asdict(x) for x in dev],
        plans=plans,
        audit=audit,
    )
    data_hash = hashlib.sha256(json.dumps(dataset, sort_keys=True).encode()).hexdigest()
    if data_hash != reference["data_sha256"]:
        raise ValueError("Conditional trial data differs from fixed replay protocol")
    tokenizer = ByteTextTokenizerM103()
    context = parent.model.student.base_model.config.context_length
    batch, seqs = encode(new + old, tokenizer, context)
    new_batch, new_seq = encode(new, tokenizer, context)
    dev_batch, dev_seq = encode(dev, tokenizer, context)
    if any(not torch.equal(a[: len(new)], b) for a, b in zip(batch, new_batch, strict=True)):
        raise ValueError("Padding changed")
    oldsets = {
        "old_known": previous["familiar_familiar"],
        "old_renamed": previous["novel_familiar"],
    }
    old_batches = {s: encode(rows, tokenizer, context)[0] for s, rows in oldsets.items()}
    old_training_batch = tuple(v[len(new) :] for v in batch)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    (out / "dataset.json").write_text(json.dumps(dataset))
    (out / "tokens.json").write_text(
        json.dumps(dict(new_old=seqs, new_only=new_seq, development=dev_seq))
    )
    r: dict[str, Any] = dict(
        provenance=dict(
            git_sha=source,
            platform=require_remote_cuda(),
            python=platform.python_version(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),
        ),
        protocol=json.loads(Path("data/specs/selective_a4_v1.json").read_text()),
        parent_sha256=ROOT_HASH,
        data_sha256=data_hash,
        reference_source=reference["provenance"]["git_sha"],
        arms={},
        promoted=False,
        sealed_generated=False,
    )
    initial_old = {
        s: enrich(evaluate(parent.model, old_batches[s], rows)) for s, rows in oldsets.items()
    }
    if min(x["exact"] for x in initial_old.values()) != 1:
        raise ValueError("Parent lacks old baseline competence")
    r["initial_old"] = initial_old
    r["initial_new"] = enrich(evaluate(parent.model, new_batch, new))
    r["initial_development"] = enrich(evaluate(parent.model, dev_batch, dev))
    # Existing ECA is already registered; this is a disposable route-only switch.
    eca = copy.deepcopy(parent.model)
    before = digest(eca.state_dict())
    eca.default_strategy = "entity_conditioned_attention"
    eca_old = {s: enrich(evaluate(eca, old_batches[s], rows)) for s, rows in oldsets.items()}
    r["eca_initial"] = dict(
        old=eca_old,
        new=enrich(evaluate(eca, new_batch, new)),
        development=enrich(evaluate(eca, dev_batch, dev)),
        weight_hash_unchanged=digest(eca.state_dict()) == before,
        stored_parameters=sum(p.numel() for p in eca.parameters()),
    )
    if not r["eca_initial"]["weight_hash_unchanged"]:
        raise RuntimeError("ECA inspection changed tensor state")
    r["eca_initial"]["admitted"] = modular_t0_admitted(
        eca_old["old_known"]["exact"], eca_old["old_renamed"]["exact"]
    )
    del eca
    arms = ["frozen_backbone"] + (["frozen_backbone_eca"] if r["eca_initial"]["admitted"] else [])
    r["eca_training_skipped"] = not r["eca_initial"]["admitted"]

    def store():
        (out / "results.json").write_text(json.dumps(r, indent=2))

    store()
    for name in arms:
        owner = fork_owner(parent, SEED)
        before_weights = digest(owner.model.state_dict())
        before_states = state_by_name(owner)
        before_steps = owner.metadata["steps"]
        before_student = digest(owner.model.student.state_dict())
        owner.set_mode("frozen_backbone")
        if name == "frozen_backbone_eca":
            owner.model.default_strategy = "entity_conditioned_attention"
        after_states = state_by_name(owner)
        if (
            before_states != after_states
            or before_weights != digest(owner.model.state_dict())
            or before_steps != owner.metadata["steps"]
            or any(optimizer_membership(owner.model, owner.optimizer).values())
        ):
            raise RuntimeError("Freezing changed weights, optimizer states, ownership or clock")
        initial = {
            s: enrich(evaluate(owner.model, old_batches[s], rows)) for s, rows in oldsets.items()
        }
        expected = eca_old if name == "frozen_backbone_eca" else initial_old
        if any(initial[s]["logits"] != expected[s]["logits"] for s in initial):
            raise RuntimeError("Trainability switch altered initial predictions")
        record_stage_provenance(
            owner,
            training_sha=source,
            publication_sha=source,
            parent_hash=ROOT_HASH,
            data_hash=data_hash,
            stage="A.4_selective",
        )
        owner.metadata.update(
            numerical_policy=deterministic_numerical_policy(),
            a4_replay_count=6,
            a4_strategy=name,
            supervision_schema=dict(attribute_ids=(10, 15), state_targets=None),
            seed=SEED,
        )
        owner.external.update(
            a4_indices=plans["6"],
            a4_cursor=0,
            a4_replay_training=[asdict(x) for x in old],
            a4_replay_audit=audit,
        )
        a: dict[str, Any] = dict(
            initial_weight_hash=before_weights,
            initial_optimizer_state_by_name_sha256=state_digest(before_states),
            optimizer_states_unchanged_by_freeze=before_states == after_states,
            initial_old=initial,
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
            a["new_presentations"] += sum(i < len(new) for i in ix)
            a["replay_presentations"] += sum(i >= len(new) for i in ix)
            a["tokens"] += int((batch[0][ix] != 256).sum())
            if step == 0:
                a["first_step_active_trainable_parameters"] = sum(
                    p.numel()
                    for p in owner.model.parameters()
                    if p.requires_grad and p.grad is not None
                )
                gradients = [
                    p.grad for p in owner.model.eca_extractor.parameters() if p.grad is not None
                ]
                a["eca_first_step_gradient_norm"] = math.sqrt(
                    sum(float(g.square().sum()) for g in gradients)
                )
                if name == "frozen_backbone_eca" and a["eca_first_step_gradient_norm"] == 0:
                    raise RuntimeError("ECA path has no learning gradient")
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
        a["backbone_hash_unchanged"] = digest(owner.model.student.state_dict()) == before_student
        if not a["backbone_hash_unchanged"]:
            raise RuntimeError("Frozen backbone changed")
        final_states = state_by_name(owner)
        a["frozen_optimizer_states_unchanged"] = all(
            final_states.get(n) == value
            for n, value in before_states.items()
            if n.startswith("student.")
        )
        if not a["frozen_optimizer_states_unchanged"]:
            raise RuntimeError("Frozen backbone optimizer state changed")
        a["final_hash"] = digest(owner.model.state_dict())
        a["lifetime_updates"] = owner.metadata["steps"]
        a["adam_counters"] = {
            n: int(owner.optimizer.state[p]["step"])
            for n, p in owner.model.named_parameters()
            if p in owner.optimizer.state
        }
        owner.save(out / f"{name}.pt")
        a["checkpoint_sha256"] = file_hash(out / f"{name}.pt")
        torch.cuda.synchronize()
        a["gpu_wall_seconds"] = time.perf_counter() - start
        r["arms"][name] = a
        store()
        del owner
    child = out / "resume_child.json"
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--resume-check",
            str(out / "frozen_backbone.pt"),
            "--output-dir",
            str(out),
        ],
        check=True,
        timeout=120,
    )
    actual = json.loads(child.read_text())
    owner = LearningState.load(out / "frozen_backbone.pt")
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
        candidate_unchanged=file_hash(out / "frozen_backbone.pt")
        == r["arms"]["frozen_backbone"]["checkpoint_sha256"],
    )
    a = r["arms"]["frozen_backbone"]
    r["qualification_admission"] = all(
        a[k] for k in ("stable_acquisition", "retention_admission", "development_admission")
    )
    r["verdict"] = (
        "DEVELOPMENT_READY_FOR_SEPARATE_SEALED_PROTOCOL"
        if r["qualification_admission"]
        else "SELECTIVE_GATE_FAIL_STOP"
    )
    r["input"] = dict(
        new_training=len(new),
        old_training=len(old),
        development=len(dev),
        max_tokens=max(map(len, seqs + dev_seq)),
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
