"""Remote Stage A only. Fixed protocol; no holdout-driven tuning or promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from run_concept_binding import language_model
from run_foundational_audit import digest

from adaptive_mini_gpt.data.attribute_learning import RULES, TAG_IDS, generate, manifest
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103
from adaptive_mini_gpt.training.learning_state import LearningState
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda
from adaptive_mini_gpt.training.reproducibility import seed_all

PARENT_HASH = "4a4f79fcd07b5bbe58bad5da04ce1a934696438c8603e9e6296e398a4b5ed5d8"


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(rows, tokenizer, context):
    sequences = [tokenizer.encode_prompt(r.question, RULES, max_length=context) for r in rows]
    for r, ids in zip(rows, sequences, strict=True):
        if tokenizer.decode(ids[ids.index(tokenizer.QUESTION_ID) + 1 : -1]) != r.question:
            raise ValueError("Incomplete description")
        if (
            tokenizer.decode(ids[ids.index(tokenizer.RULE_ID) + 1 : ids.index(tokenizer.SEP_ID)])
            != RULES
        ):
            raise ValueError("Incomplete semantics")
    tokens, _ = tokenizer.pad_batch(sequences, device=torch.device("cuda"))
    targets = torch.zeros(len(rows), 26, device="cuda")
    targets[:, TAG_IDS] = torch.tensor([r.labels for r in rows], device="cuda", dtype=torch.float32)
    return (tokens, torch.zeros(len(rows), device="cuda", dtype=torch.long), targets), sequences


def scores(logits, truth, rows):
    pred = (logits >= 0).int()
    truth = truth.int()
    tp = int((pred.bool() & truth.bool()).sum())
    fp = int((pred.bool() & ~truth.bool()).sum())
    fn = int((~pred.bool() & truth.bool()).sum())
    exact = pred.eq(truth).all(-1)
    return {
        "exact": float(exact.float().mean()),
        "micro_f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "per_property_accuracy": pred.eq(truth).float().mean(0).tolist(),
        "status_accuracy": {
            status: sum(
                int(pred[i, j] == truth[i, j])
                for i, r in enumerate(rows)
                for j, s in enumerate(r.statuses)
                if s == status
            )
            / sum(s == status for r in rows for s in r.statuses)
            for status in ("positive", "negated", "absent")
        },
        "family_exact": {
            f: float(exact[[r.family == f for r in rows]].float().mean())
            for f in sorted({r.family for r in rows})
        },
        "predictions": pred.tolist(),
        "logits": logits.tolist(),
        "truth": truth.tolist(),
    }


def evaluate(model, batch, rows, *, features=None):
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            if features is None:
                _, tags, _, _ = model(batch[0][i : i + 32], batch[1][i : i + 32])
                chunks.append(tags[:, TAG_IDS])
            else:
                chunks.append(model(features[i : i + 32]))
    return scores(torch.cat(chunks), batch[2][:, TAG_IDS], rows)


def words(row, kind):
    w = re.findall(r"\w+", row.question.lower())
    return [
        " ".join(w[i : i + n])
        for n in (range(1, 4) if kind == "ngram_linear" else (1,))
        for i in range(len(w) - n + 1)
    ]


def run(out, parent):
    if file_hash(parent) != PARENT_HASH:
        raise ValueError("Wrong persistent parent; refusing fresh fallback")
    seed = 42
    spec = json.loads(Path("data/specs/attribute_learning_v1.json").read_text())
    splits = generate()
    data = manifest(splits)
    tokenizer = ByteTextTokenizerM103()
    continuing = LearningState.load(parent)
    if continuing.metadata["steps"] != 200 or continuing.model.mode != "selective_plasticity":
        raise ValueError("Unexpected parent lineage")
    context = continuing.model.student.base_model.config.context_length
    batches, sequences = {}, {}
    for s, rows in splits.items():
        batches[s], sequences[s] = encode(rows, tokenizer, context)
    data.update(
        context_length=context,
        max_tokens=max(len(t) for seqs in sequences.values() for t in seqs),
        delivered=sum(len(x) for x in sequences.values()),
    )
    (out / "manifest.json").write_text(json.dumps(data, indent=2))
    (out / "cases.json").write_text(
        json.dumps({s: [asdict(r) for r in rows] for s, rows in splits.items()}, indent=2)
    )
    (out / "token_sequences.json").write_text(json.dumps(sequences))
    result = {
        "spec": spec,
        "data": data,
        "neural": {},
        "baselines": {},
        "lineage": {
            "parent_sha256": PARENT_HASH,
            "parent_protocol": "concept-binding-v1",
            "parent_updates": 200,
            "historical_sampler": "Not owned in parent. New stage sampler explicitly initialized; no exact retrospective resume claim.",
        },
    }

    def store():
        (out / "results.json").write_text(json.dumps(result, indent=2))

    arms = (
        ("continuing_selective_two", "selective_plasticity", TAG_IDS),
        ("fresh_frozen_two", "frozen_backbone", TAG_IDS),
        ("fresh_selective_two", "selective_plasticity", TAG_IDS),
        ("fresh_full_two", "trainable_backbone", TAG_IDS),
        ("fresh_selective_all26", "selective_plasticity", None),
    )
    for name, mode, ids in arms:
        seed_all(seed)
        owner = continuing if name.startswith("continuing") else LearningState(language_model(mode))
        owner.set_mode(mode)
        previous_metadata = dict(owner.metadata)
        if any(
            group["lr"] != spec["learning_rate"] or group["weight_decay"] != 0
            for group in owner.optimizer.param_groups
        ):
            raise ValueError("Unexpected optimizer settings; no silent reset")
        owner.external["sampler"] = random.Random(seed)
        owner.metadata.update(
            previous_stage_metadata=previous_metadata,
            parent_checkpoint_sha256=PARENT_HASH if name.startswith("continuing") else None,
            stage="A",
            protocol=data["protocol"],
            data_hash=data["data_sha256"],
            git_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            supervision={
                "kind": "attribute_only",
                "attribute_ids": ids,
                "unused_columns": "unobserved" if ids else "explicit absent diagnostic",
            },
            label_schema={
                "tags": dict(zip(("flammable", "conductive"), TAG_IDS, strict=True)),
                "states": "NOT_SUPERVISED",
                "actions": "constant placeholder; attribute head independent of action",
            },
        )
        initial_hash = digest(owner.model.state_dict())
        # Behavioral competence is not established by this parameter invariant.
        state_keys = ("action_emb", "transition_net", "post_norm", "state_head")
        transition_before = digest(
            {n: p for n, p in owner.model.state_dict().items() if any(k in n for k in state_keys)}
        )
        arm = {
            "initial_hash": initial_hash,
            "initial_total_steps": owner.metadata["steps"],
            "stored_parameters": sum(p.numel() for p in owner.model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in owner.model.parameters() if p.requires_grad
            ),
            "updates": 0,
            "examples": 0,
            "tokens": 0,
            "curves": [],
        }
        torch.cuda.synchronize()
        start = time.perf_counter()
        for phase in ("tiny", "train"):
            batch = batches[phase]
            for step in range(spec["updates"][phase]):
                ix = owner.external["sampler"].sample(
                    range(len(splits[phase])), min(spec["batch_max"], len(splits[phase]))
                )
                sampled = [v[ix] for v in batch]
                loss = owner.step(*sampled, attribute_ids=ids)
                arm["updates"] += 1
                arm["examples"] += len(ix)
                arm["tokens"] += int((sampled[0] != tokenizer.PAD_ID).sum())
                owner.metadata.update(
                    phase=phase,
                    phase_step=step + 1,
                    stage_examples=arm["examples"],
                    stage_tokens=arm["tokens"],
                )
                if (step + 1) % 100 == 0:
                    val = evaluate(owner.model, batches["validation"], splits["validation"])
                    arm["curves"].append(
                        {
                            "updates": arm["updates"],
                            "examples": arm["examples"],
                            "loss": loss,
                            "validation_exact": val["exact"],
                            "gpu_wall_seconds": time.perf_counter() - start,
                        }
                    )
            arm[phase] = evaluate(owner.model, batch, splits[phase])
            if phase == "tiny" and arm[phase]["exact"] < spec["gates"]["tiny_exact"]:
                arm["stop_reason"] = "Fixed tiny acquisition gate failed; main and heldout NOT RUN"
                break
        if "train" in arm:
            for s in ("names", "syntax"):
                arm[s] = evaluate(owner.model, batches[s], splits[s])
        torch.cuda.synchronize()
        arm["gpu_wall_seconds"] = time.perf_counter() - start
        transition_after = digest(
            {n: p for n, p in owner.model.state_dict().items() if any(k in n for k in state_keys)}
        )
        if transition_before != transition_after:
            raise RuntimeError("Unsupervised transition weights changed")
        arm["transition_weights_unchanged"] = True
        arm["final_hash"] = digest(owner.model.state_dict())
        arm["total_steps"] = owner.metadata["steps"]
        path = out / f"{name}.pt"
        owner.save(path)
        arm["checkpoint_sha256"] = file_hash(path)
        result["neural"][name] = arm
        store()
        if owner is not continuing:
            del owner

    for kind in ("bow_linear", "bow_mlp", "ngram_linear"):
        seed_all(seed)
        rng = random.Random(seed)
        vocab = sorted({w for row in splits["train"] for w in words(row, kind)})
        features = {}
        for s, rows in splits.items():
            counts = [Counter(words(r, kind)) for r in rows]
            features[s] = torch.tensor(
                [[float(c[w]) for w in vocab] for c in counts], device="cuda"
            )
        model = (
            torch.nn.Sequential(
                torch.nn.Linear(len(vocab), 32), torch.nn.GELU(), torch.nn.Linear(32, 2)
            )
            if kind == "bow_mlp"
            else torch.nn.Linear(len(vocab), 2)
        ).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], weight_decay=0)
        arm = {"updates": 0, "examples": 0, "feature_dimensions": len(vocab), "curves": []}
        torch.cuda.synchronize()
        start = time.perf_counter()
        for phase in ("tiny", "train"):
            for step in range(spec["updates"][phase]):
                ix = rng.sample(
                    range(len(splits[phase])), min(spec["batch_max"], len(splits[phase]))
                )
                opt.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(
                    model(features[phase][ix]), batches[phase][2][ix][:, TAG_IDS]
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                arm["updates"] += 1
                arm["examples"] += len(ix)
                if (step + 1) % 100 == 0:
                    val = evaluate(
                        model,
                        batches["validation"],
                        splits["validation"],
                        features=features["validation"],
                    )
                    arm["curves"].append(
                        {
                            "updates": arm["updates"],
                            "examples": arm["examples"],
                            "loss": float(loss.detach()),
                            "validation_exact": val["exact"],
                        }
                    )
        for s, rows in splits.items():
            arm[s] = evaluate(model, batches[s], rows, features=features[s])
        torch.cuda.synchronize()
        arm["gpu_wall_seconds"] = time.perf_counter() - start
        result["baselines"][kind] = arm
        store()
    counts = Counter(r.labels for r in splits["train"])
    majority = counts.most_common(1)[0][0]
    result["baselines"]["majority"] = {
        s: scores(
            torch.tensor([[1.0 if v else -1.0 for v in majority]] * len(rows), device="cuda"),
            batches[s][2][:, TAG_IDS],
            rows,
        )
        for s, rows in splits.items()
    }
    for arm in result["neural"].values():
        basic = arm.get("train", {}).get("exact", 0) >= spec["gates"]["train_exact"]
        advantages = {}
        for s in ("names", "syntax"):
            m = arm.get(s, {})
            basic = (
                basic
                and m.get("exact", 0) >= spec["gates"]["heldout_exact"]
                and min(m.get("status_accuracy", {"missing": 0}).values())
                >= spec["gates"]["status_accuracy"]
            )
            advantages[s] = m.get("exact", 0) - max(
                b[s]["exact"] for b in result["baselines"].values()
            )
        arm["basic_acquisition_pass"] = basic
        arm["advantage_over_strongest_simple"] = advantages
        arm["reusable_qualification_pass"] = (
            basic and min(advantages.values()) >= spec["gates"]["simple_advantage"]
        )
    primary = result["neural"]["continuing_selective_two"]
    result["verdict"] = (
        "STAGE_A_FEASIBILITY_ONLY"
        if primary["reusable_qualification_pass"]
        else "STOP_STAGE_A_NOT_QUALIFIED"
    )
    result["later_stages"] = (
        "NOT_RUN; no automatic progression/promotion, retention, transfer or self-improvement claim"
    )
    store()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent", required=True)
    args = parser.parse_args()
    remote = require_remote_cuda()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "platform": remote,
                "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "seed": 42,
            },
            indent=2,
        )
    )
    run(out, Path(args.parent))


if __name__ == "__main__":
    main()
