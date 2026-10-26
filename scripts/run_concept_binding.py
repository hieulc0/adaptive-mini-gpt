"""Prespecified remote feasibility; no architecture/threshold/seed search."""

from __future__ import annotations

import argparse
import json
import platform
import random
import re
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from run_foundational_audit import digest

from adaptive_mini_gpt.data.concept_binding import generate, manifest
from adaptive_mini_gpt.natural_language_learning_m103.semantic_bottleneck_m104_e import (
    IntegratedStudentM104_E,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import TextStudentM103
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103
from adaptive_mini_gpt.training.learning_state import LearningState
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda
from adaptive_mini_gpt.training.reproducibility import seed_all

TAG_IDS = {"flammable": 10, "conductive": 15}


def encode(rows, tokenizer, context):
    ids = [tokenizer.encode_prompt(c.question, c.rules, max_length=context) for c in rows]
    for c, t in zip(rows, ids, strict=True):
        assert tokenizer.decode(t[t.index(tokenizer.QUESTION_ID) + 1 : -1]) == c.question
        assert (
            tokenizer.decode(t[t.index(tokenizer.RULE_ID) + 1 : t.index(tokenizer.SEP_ID)])
            == c.rules
        )
    tokens, _ = tokenizer.pad_batch(ids, device=torch.device("cuda"))
    targets = torch.zeros(len(rows), 26, device="cuda")
    for i, c in enumerate(rows):
        for tag in c.tags:
            targets[i, TAG_IDS[tag]] = 1
    return (
        tokens,
        torch.tensor([c.action_id for c in rows], device="cuda"),
        targets,
        torch.tensor([c.state for c in rows], device="cuda"),
    ), ids


def metrics(logits, attr, batch, rows):
    pred = logits.argmax(-1)
    tags = attr.sigmoid() >= 0.5
    truth = batch[2].bool()
    tp = (tags & truth).sum().item()
    fp = (tags & ~truth).sum().item()
    fn = (~tags & truth).sum().item()
    per = defaultdict(list)
    for i, row in enumerate(rows):
        per[row.family].append(bool(pred[i] == batch[3][i]))
    return {
        "accuracy": float(pred.eq(batch[3]).float().mean()),
        "attribute_f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "attribute_exact": float(tags.eq(truth).all(-1).float().mean()),
        "families": {name: sum(v) / len(v) for name, v in per.items()},
        "state_predictions": pred.tolist(),
        "tag_predictions": tags.int().tolist(),
    }


def evaluate(model, batch, rows, feature=None):
    model.eval()
    st, attrs = [], []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            if feature is None:
                state, tag, _, _ = model(batch[0][i : i + 32], batch[1][i : i + 32])
            else:
                output = model(feature[i : i + 32])
                tag, state = output[:, :26], output[:, 26:]
            st.append(state)
            attrs.append(tag)
    return metrics(torch.cat(st), torch.cat(attrs), batch, rows)


def language_model(mode):
    return IntegratedStudentM104_E(
        TextStudentM103(prune_orphaned_lm_head=True), mode=mode, default_strategy="question_only"
    ).cuda()


def feature_keys(row, kind):
    words = re.findall(r"\w+", (row.rules + " " + row.question).lower())
    return [
        " ".join(words[i : i + n])
        for n in (range(1, 4) if kind == "ngram_linear" else (1,))
        for i in range(len(words) - n + 1)
    ]


def run(out):
    seed = 42
    steps = {"tiny": 200, "train": 800}
    splits = generate(seed)
    tokenizer = ByteTextTokenizerM103()
    # Actual model budget, no auto-resizing and no inference-time oracle features.
    seed_all(seed)
    probe = language_model("frozen_backbone")
    context = probe.student.base_model.config.context_length
    del probe
    batches = {}
    sequences = {}
    for name, rows in splits.items():
        batches[name], sequences[name] = encode(rows, tokenizer, context)
    spec = manifest(splits, seed=seed)
    spec.update(
        context_length=context,
        max_tokens=max(len(t) for rows in sequences.values() for t in rows),
        gates={
            "tiny_fit": 0.95,
            "family_accuracy": 0.8,
            "attribute_exact": 0.8,
            "advantage_over_simple": 0.1,
        },
        steps=steps,
    )
    (out / "manifest.json").write_text(json.dumps(spec, indent=2))
    (out / "cases.json").write_text(
        json.dumps({s: [asdict(c) for c in rows] for s, rows in splits.items()}, indent=2)
    )
    (out / "token_sequences.json").write_text(json.dumps(sequences))
    result = {
        "protocol": spec,
        "neural": {},
        "baselines": {},
        "curves": {},
        "scope": "Single-seed two-trait feasibility; no retention/transfer/self-modification qualification",
    }
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

    def store():
        (out / "results.json").write_text(json.dumps(result, indent=2))

    for mode in ("frozen_backbone", "selective_plasticity", "trainable_backbone"):
        seed_all(seed)
        owner = LearningState(
            language_model(mode), metadata={"git_sha": sha, "data_hash": spec["data_sha256"]}
        )
        initial = digest(owner.model.state_dict())
        owner.external["sampler"] = random.Random(seed)
        owner.metadata["label_schema"] = {
            "state": {0: "light_off", 1: "light_on"},
            "tags": TAG_IDS,
            "actions": {0: "fire", 2: "spark"},
        }
        arm = {
            "initial_hash": initial,
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
            for step in range(steps[phase]):
                ix = owner.external["sampler"].sample(
                    range(len(splits[phase])), min(16, len(splits[phase]))
                )
                sampled = [x[ix] for x in batch]
                loss = owner.step(*sampled)
                arm["updates"] += 1
                arm["examples"] += len(ix)
                arm["tokens"] += int((sampled[0] != tokenizer.PAD_ID).sum())
                owner.metadata.update(
                    phase=phase, phase_step=step + 1, examples=arm["examples"], tokens=arm["tokens"]
                )
                if phase == "train" and (step + 1) % 100 == 0:
                    val = evaluate(owner.model, batches["validation"], splits["validation"])
                    arm["curves"].append(
                        {
                            "updates": arm["updates"],
                            "examples": arm["examples"],
                            "training_loss": loss,
                            "validation_accuracy": val["accuracy"],
                            "validation_attribute_exact": val["attribute_exact"],
                            "elapsed_seconds": time.perf_counter() - start,
                        }
                    )
            arm[phase] = evaluate(owner.model, batch, splits[phase])
            if (
                phase == "tiny"
                and min(arm[phase]["accuracy"], arm[phase]["attribute_exact"]) < 0.95
            ):
                arm["stop_reason"] = "Tiny fit failed fixed gate; stop this arm"
                break
        if "train" in arm:
            for split in ("description", "composition"):
                arm[split] = evaluate(owner.model, batches[split], splits[split])
        torch.cuda.synchronize()
        arm["gpu_wall_seconds"] = time.perf_counter() - start
        arm["final_hash"] = digest(owner.model.state_dict())
        owner.save(out / f"{mode}.pt")
        result["neural"][mode] = arm
        store()
        del owner
    for kind in ("bow_linear", "bow_mlp", "ngram_linear", "structured_oracle_mlp"):
        seed_all(seed)
        rng = random.Random(seed)
        if kind == "structured_oracle_mlp":
            features = {
                s: torch.tensor(
                    [c.structured_diagnostic for c in rows], dtype=torch.float32, device="cuda"
                )
                for s, rows in splits.items()
            }
        else:
            vocabulary = sorted({key for row in splits["train"] for key in feature_keys(row, kind)})

            def features_of(rows, kind=kind, vocabulary=vocabulary):
                values = []
                for row in rows:
                    counts = Counter(feature_keys(row, kind))
                    values.append(
                        [float(counts[w]) for w in vocabulary]
                        + [float(row.action_id == 0), float(row.action_id == 2)]
                    )
                return torch.tensor(values, device="cuda")

            features = {s: features_of(rows) for s, rows in splits.items()}
        dim = features["train"].shape[-1]
        model = (
            torch.nn.Sequential(torch.nn.Linear(dim, 32), torch.nn.GELU(), torch.nn.Linear(32, 42))
            if kind.endswith("mlp")
            else torch.nn.Linear(dim, 42)
        ).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0)
        start = time.perf_counter()
        arm = {"updates": 0, "examples": 0, "feature_dimensions": dim, "curves": []}
        for phase in ("tiny", "train"):
            x = features[phase]
            batch = batches[phase]
            for step in range(steps[phase]):
                ix = rng.sample(range(len(x)), min(16, len(x)))
                opt.zero_grad(set_to_none=True)
                pred = model(x[ix])
                loss = F.cross_entropy(
                    pred[:, 26:], batch[3][ix]
                ) + F.binary_cross_entropy_with_logits(pred[:, :26], batch[2][ix])
                loss.backward()
                opt.step()
                arm["updates"] += 1
                arm["examples"] += len(ix)
                if phase == "train" and (step + 1) % 100 == 0:
                    val = evaluate(
                        model, batches["validation"], splits["validation"], features["validation"]
                    )
                    arm["curves"].append(
                        {
                            "updates": arm["updates"],
                            "examples": arm["examples"],
                            "validation_accuracy": val["accuracy"],
                            "validation_attribute_exact": val["attribute_exact"],
                        }
                    )
        for s, rows in splits.items():
            arm[s] = evaluate(model, batches[s], rows, features[s])
        torch.cuda.synchronize()
        arm["gpu_wall_seconds"] = time.perf_counter() - start
        result["baselines"][kind] = arm
        store()
    for s, rows in splits.items():
        majority = Counter(c.state for c in splits["train"]).most_common(1)[0][0]
        result["baselines"].setdefault("majority", {})[s] = {
            "accuracy": sum(c.state == majority for c in rows) / len(rows)
        }
    eligible = []
    for name, arm in result["neural"].items():
        checks = []
        for split in ("description", "composition"):
            scores = arm.get(split, {})
            strongest = max(
                result["baselines"][b][split]["accuracy"]
                for b in ("bow_linear", "bow_mlp", "ngram_linear", "majority")
            )
            checks.append(
                bool(scores)
                and min(scores["families"].values()) >= 0.8
                and scores["attribute_exact"] >= 0.8
                and scores["accuracy"] >= strongest + 0.1
            )
        if all(checks) and arm.get("train", {}).get("accuracy", 0) >= 0.95:
            eligible.append(name)
    result["verdict"] = (
        "FEASIBILITY_PASS_NOT_QUALIFIED" if eligible else "FAIL_STOP_NO_TRANSFER_OR_GROWTH"
    )
    result["eligible_arms"] = eligible
    result["feedback_retention"] = (
        "NOT_RUN: requires independent concept qualification beyond a single feasibility world"
    )
    store()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
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
    run(out)


if __name__ == "__main__":
    main()
