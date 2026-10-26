"""Bounded approved repair diagnostics; all neural work requires remote CUDA."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from run_foundational_audit import digest, evaluate

from adaptive_mini_gpt.concept_instruction.capacity_feasibility import StageACapacityBenchmark
from adaptive_mini_gpt.model import MiniGPT, ModelConfig
from adaptive_mini_gpt.natural_language_learning_m103.concept_learner_m104_e import (
    ConceptLearnerM104_E,
)
from adaptive_mini_gpt.natural_language_learning_m103.input_protocol import (
    ObservedCase,
    dataset_hash,
    encode_observation,
    validate_cases,
)
from adaptive_mini_gpt.natural_language_learning_m103.semantic_bottleneck_m104_e import (
    IntegratedStudentM104_E,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import TextStudentM103
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103
from adaptive_mini_gpt.training.optimizer_contract import adamw_groups, optimizer_membership
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda
from adaptive_mini_gpt.training.reproducibility import isolated_rng, seed_all

BASELINE = "a6d8e11eda1ecfb7067f1552b7c8e7eb13d2eb11"
RULES = "Report stated attributes only. Fire ignites flammable items. Spark charges conductive items. Otherwise dormant."
TRAIN_TEMPLATES = [
    "{name} is {material} and {prop}. Apply {action}.",
    "Use {action} on {name}, a {prop} {material} item.",
    "Item {name}: {material}; {prop}. Action: {action}.",
    "A {material} object named {name} is {prop}. It receives {action}.",
]
TEST_TEMPLATES = [
    "Give {name} {action}. Its properties are {material} and {prop}.",
    "Consider {name}, made of {material} and described as {prop}; perform {action}.",
]


def make_cases(split):
    templates = (
        TRAIN_TEMPLATES
        if split == "train"
        else TRAIN_TEMPLATES[:1]
        if split in ("tiny", "name_only")
        else TEST_TEMPLATES
    )
    names = (
        ["ta", "tb"]
        if split == "tiny"
        else [f"tr{i}" for i in range(8)]
        if split in ("train", "description_only")
        else [f"uv{i}" for i in range(8)]
    )
    result = []
    for idx, template in enumerate(templates):
        for name in names:
            for material, prop in (("wood", "flammable"), ("metal", "conductive")):
                for action in ("fire", "spark"):
                    state = (
                        "Ignited"
                        if material == "wood" and action == "fire"
                        else "Charged"
                        if material == "metal" and action == "spark"
                        else "Dormant"
                    )
                    result.append(
                        ObservedCase(
                            f"{split}:{idx}:{name}:{material}:{action}",
                            template.format(name=name, material=material, prop=prop, action=action),
                            RULES,
                            "Apply_Fire" if action == "fire" else "Apply_Spark",
                            (material, prop),
                            state,
                            {material: material, prop: prop},
                        )
                    )
    return result


def encode(cases, model, learner, tokenizer):
    context = model.student.base_model.config.context_length
    sequences = [encode_observation(c, tokenizer, context) for c in cases]
    tokens, _ = tokenizer.pad_batch(sequences, device=torch.device("cuda"))
    tags = torch.zeros(len(cases), len(learner.tag_vocab), device="cuda")
    for i, case in enumerate(cases):
        for tag in case.tags:
            tags[i, learner.tag_vocab[tag]] = 1
    return (
        tokens,
        torch.tensor([learner.action_vocab[c.action] for c in cases], device="cuda"),
        tags,
        torch.tensor([learner.state_vocab[c.state] for c in cases], device="cuda"),
    ), sequences


def scores(attr, state, batch):
    truth = batch[2].bool()
    pred = attr >= 0.5
    tp, fp, fn = (pred & truth).sum(), (pred & ~truth).sum(), (~pred & truth).sum()
    return {
        "micro_f1": (2 * tp / (2 * tp + fp + fn).clamp_min(1)).item(),
        "attribute_exact": pred.eq(truth).all(-1).float().mean().item(),
        "state_accuracy": state.argmax(-1).eq(batch[3]).float().mean().item(),
    }


def learning(out, steps=200):
    seed = 42  # Prespecified; no seed search or post-test threshold selection.
    learner = ConceptLearnerM104_E(torch.device("cuda"))
    tokenizer = ByteTextTokenizerM103()
    data = {
        s: make_cases(s) for s in ("tiny", "train", "name_only", "description_only", "combined")
    }
    validate_cases(
        [c for rows in data.values() for c in rows],
        learner.tag_vocab,
        learner.action_vocab,
        learner.state_vocab,
    )
    train_text = {c.question for c in data["train"]}
    assert all(
        c.question not in train_text
        for s in ("name_only", "description_only", "combined")
        for c in data[s]
    )
    (out / "cases.json").write_text(
        json.dumps({s: [asdict(c) for c in rows] for s, rows in data.items()}, indent=2)
    )
    results = {
        "seed": seed,
        "data_hash": dataset_hash(data),
        "threshold": 0.5,
        "steps_per_phase": steps,
        "arms": {},
        "baselines": {},
    }
    for mode in ("frozen_backbone", "selective_plasticity", "trainable_backbone"):
        seed_all(seed)
        model = IntegratedStudentM104_E(
            TextStudentM103(prune_orphaned_lm_head=True),
            mode=mode,
            default_strategy="question_only",
        ).cuda()
        batches, tokens = {}, {}
        for s, cases in data.items():
            batches[s], tokens[s] = encode(cases, model, learner, tokenizer)
        (out / "token_sequences.json").write_text(
            json.dumps(
                {
                    "context_length": model.student.base_model.config.context_length,
                    "sequences": tokens,
                    "max_tokens": max(len(x) for rows in tokens.values() for x in rows),
                },
                indent=2,
            )
        )
        optimizer = torch.optim.AdamW(adamw_groups(model, 0), lr=0.002)
        arm = {
            "initial_hash": digest(model.state_dict()),
            "optimizer_membership": optimizer_membership(model, optimizer),
            "updates": 0,
            "examples": 0,
            "tokens": 0,
            "gradients": {},
        }
        assert not any(arm["optimizer_membership"].values())
        rng = random.Random(seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for phase in ("tiny", "train"):
            full = batches[phase]
            for step in range(steps):
                indices = rng.sample(range(len(data[phase])), min(16, len(data[phase])))
                batch = [x[indices] for x in full]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                state, attr, _, _ = model(batch[0], batch[1])
                loss = F.binary_cross_entropy_with_logits(attr, batch[2])
                if phase == "train":
                    loss = loss + F.cross_entropy(state, batch[3])
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite supervised loss")
                loss.backward()
                if step in (0, 1) or step == steps - 1:
                    arm["gradients"][f"{phase}:{step}"] = [
                        name
                        for name, p in model.named_parameters()
                        if p.grad is not None and p.grad.abs().sum().item() > 0
                    ]
                    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                arm["updates"] += 1
                arm["examples"] += len(indices)
                arm["tokens"] += (batch[0] != tokenizer.PAD_ID).sum().item()
            arm[phase] = evaluate(model, full)
            if phase == "tiny" and arm[phase]["micro_f1"] < 0.95:
                arm["stop_reason"] = "Tiny acquisition below fixed 95% F1; no further learning"
                break
        if "train" in arm:
            for split in ("name_only", "description_only", "combined"):
                arm[split] = evaluate(model, batches[split])
            t, a, y, state = batches["combined"]
            arm["description_mismatch"] = evaluate(model, (t.roll(2, 0), a, y, state))
        torch.cuda.synchronize()
        arm["gpu_wall_seconds"] = time.perf_counter() - start
        arm["final_hash"] = digest(model.state_dict())
        torch.save(model.state_dict(), out / f"{mode}.pt")
        results["arms"][mode] = arm
        del model, optimizer
        (out / "learning.json").write_text(json.dumps(results, indent=2))
    # Research-only simple baseline. Its vocabulary is learned on train text;
    # no hard-coded extraction/rule solver enters the student.
    vocabulary = sorted(
        {word for c in data["train"] for word in re.findall(r"\w+", c.question.lower())}
    )

    def features(rows):
        return torch.tensor(
            [
                [float(w in re.findall(r"\w+", c.question.lower())) for w in vocabulary]
                for c in rows
            ],
            device="cuda",
        )

    seed_all(seed)
    linear = torch.nn.Linear(
        len(vocabulary), len(learner.tag_vocab) + len(learner.state_vocab)
    ).cuda()
    opt = torch.optim.AdamW(linear.parameters(), lr=0.002, weight_decay=0)
    rng = random.Random(seed)
    baseline_start = time.perf_counter()
    for phase in ("tiny", "train"):
        x = features(data[phase])
        target, labels = batches[phase][2:]
        for _ in range(steps):
            ix = rng.sample(range(len(x)), min(16, len(x)))
            opt.zero_grad()
            prediction = linear(x[ix])
            loss = F.binary_cross_entropy_with_logits(
                prediction[:, : len(learner.tag_vocab)], target[ix]
            )
            if phase == "train":
                loss = loss + F.cross_entropy(prediction[:, len(learner.tag_vocab) :], labels[ix])
            loss.backward()
            opt.step()
    for split, rows in data.items():
        with torch.no_grad():
            prediction = linear(features(rows))
            results["baselines"].setdefault("bag_of_words_linear", {})[split] = scores(
                prediction[:, : len(learner.tag_vocab)].sigmoid(),
                prediction[:, len(learner.tag_vocab) :],
                batches[split],
            )
            prior_tags = batches["train"][2].mean(0).expand(len(rows), -1)
            prior_state = (
                F.one_hot(batches["train"][3].bincount(minlength=16).argmax(), num_classes=16)
                .float()
                .expand(len(rows), -1)
            )
            results["baselines"].setdefault("majority", {})[split] = scores(
                prior_tags, prior_state, batches[split]
            )
    torch.cuda.synchronize()
    results["baseline_budget"] = {
        "updates": 2 * steps,
        "examples": steps * (min(16, len(data["tiny"])) + min(16, len(data["train"]))),
        "gpu_wall_seconds": time.perf_counter() - baseline_start,
        "minibatch_rng": seed,
        "vocabulary": "train only",
        "checkpoint_hash": digest(linear.state_dict()),
    }
    results["scope"] = (
        "One world/seed; no retention, transfer or learned self-modification qualification"
    )
    (out / "learning.json").write_text(json.dumps(results, indent=2))


def m90_diagnosis(out):
    # Historical evaluator is executed unchanged; all historical artifacts remain.
    source = subprocess.check_output(
        [
            "git",
            "show",
            f"{BASELINE}:src/adaptive_mini_gpt/concept_instruction/capacity_feasibility.py",
        ],
        text=True,
    )
    path = out / "historical_capacity.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("historical_capacity", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load historical evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    with isolated_rng(42, "backbone"):
        model = MiniGPT(
            ModelConfig(
                vocab_size=1024, context_length=128, d_model=128, n_heads=4, n_layers=4, d_ff=512
            )
        ).cuda()
    for p in model.parameters():
        p.requires_grad = False
    from adaptive_mini_gpt.continual_retention.curriculum_m89 import M89CurriculumGenerator

    curriculum = M89CurriculumGenerator().generate_curriculum(num_tasks=6, seed=42)
    data_hash = hashlib.sha256(
        json.dumps([asdict(t) for t in curriculum], sort_keys=True, default=str).encode()
    ).hexdigest()
    records = {
        "data_hash": data_hash,
        "backbone_hash": digest(model.state_dict()),
        "curriculum_seed": 42,
        "epochs": 50,
        "ranks": [16, 32],
        "gate": ">=85% at rank32, unchanged",
        "cases": {},
    }
    cases = [
        ("legacy_ambient111", module.StageACapacityBenchmark, 111, False),
        ("legacy_ambient222", module.StageACapacityBenchmark, 222, False),
        ("seeded_legacy_decay", StageACapacityBenchmark, 111, True),
        ("repaired_ambient111", StageACapacityBenchmark, 111, False),
        ("repaired_ambient222", StageACapacityBenchmark, 222, False),
    ]
    for name, benchmark, ambient, legacy_decay in cases:
        seed_all(ambient)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with patch(
            "adaptive_mini_gpt.concept_instruction.capacity_feasibility.adamw_groups",
            side_effect=(lambda model, wd: list(model.parameters()))
            if legacy_decay
            else adamw_groups,
        ):
            result = benchmark.evaluate_capacity_sweep(
                model, torch.device("cuda"), ranks=[16, 32], num_tasks=6, seed=42, epochs=50
            )
        torch.cuda.synchronize()
        records["cases"][name] = {
            "report": asdict(result),
            "gpu_wall_seconds": time.perf_counter() - start,
        }
        (out / "m90_diagnosis.json").write_text(json.dumps(records, indent=2))
    assert (
        records["cases"]["repaired_ambient111"]["report"]
        == records["cases"]["repaired_ambient222"]["report"]
    )


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
                "kernel": "sparklinghydraulics0/amg-approved-audit-repair",
                "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
            indent=2,
        )
    )
    m90_diagnosis(out)
    learning(out)


if __name__ == "__main__":
    main()
