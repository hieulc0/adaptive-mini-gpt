"""Single-seed remote feasibility diagnostic, not a milestone qualification.

Complete short prompts; novel names and held-out phrasing. No oracle features
enter the model. Labels below belong solely to synthetic supervision/evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adaptive_mini_gpt.natural_language_learning_m103.concept_learner_m104_e import (
    ConceptLearnerM104_E,
)
from adaptive_mini_gpt.natural_language_learning_m103.semantic_bottleneck_m104_e import (
    IntegratedStudentM104_E,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import TextStudentM103
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda


def digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fixtures(split):
    # Distinct names and syntax across splits; crossed material/action labels.
    templates = {
        "tiny": "{name} is {material}. Apply {action}.",
        "train": "The object {name} is {material}. We apply {action}.",
        "dev": "{name}, made of {material}, receives {action}.",
        "sealed": "Use {action} on {name}. Its material is {material}.",
    }
    rows = []
    for i in range(1 if split == "tiny" else 8):
        for material in ("wood", "metal"):
            for action in ("fire", "spark"):
                rows.append(
                    {
                        "question": templates[split].format(
                            name=f"{split}_{i}", material=material, action=action
                        ),
                        "tags": ["wood", "flammable"]
                        if material == "wood"
                        else ["metal", "conductive"],
                        "action": "Apply_Fire" if action == "fire" else "Apply_Spark",
                        "state": "Ignited"
                        if material == "wood" and action == "fire"
                        else "Charged"
                        if material == "metal" and action == "spark"
                        else "Dormant",
                    }
                )
    return rows


def encode(rows, learner, tokenizer):
    ids = [tokenizer.encode_prompt(r["question"], max_length=256) for r in rows]
    tokens, _ = tokenizer.pad_batch(ids, device=torch.device("cuda"))
    tags = torch.zeros(len(rows), len(learner.tag_vocab), device="cuda")
    for i, row in enumerate(rows):
        for tag in row["tags"]:
            tags[i, learner.tag_vocab[tag]] = 1
    return (
        tokens,
        torch.tensor([learner.action_vocab[r["action"]] for r in rows], device="cuda"),
        tags,
        torch.tensor([learner.state_vocab[r["state"]] for r in rows], device="cuda"),
    )


def evaluate(model, batch):
    model.eval()
    tokens, actions, tags, states = batch
    with torch.no_grad():
        logits, attr, _, _ = model(tokens, actions)
        pred = attr.sigmoid() >= 0.5  # Fixed before dev/sealed evaluation.
        truth = tags.bool()
        tp = (pred & truth).sum().item()
        fp = (pred & ~truth).sum().item()
        fn = (~pred & truth).sum().item()
        return {
            "micro_f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "attribute_exact": pred.eq(truth).all(dim=1).float().mean().item(),
            "state_accuracy": logits.argmax(-1).eq(states).float().mean().item(),
            "state_predictions": logits.argmax(-1).tolist(),
            "attribute_predictions": pred.int().tolist(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="runs/foundational-audit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    remote = require_remote_cuda()
    if not 1 <= args.steps <= 400:
        raise ValueError("Feasibility budget must be 1..400 steps")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    learner = ConceptLearnerM104_E(torch.device("cuda"))
    tokenizer = ByteTextTokenizerM103()
    data = {split: fixtures(split) for split in ("tiny", "train", "dev", "sealed")}
    batch = {split: encode(rows, learner, tokenizer) for split, rows in data.items()}
    report = {
        "status": "FEASIBILITY_ONLY",
        "platform": remote,
        "kernel_or_session": __import__("os").environ.get("KAGGLE_KERNEL_RUN_TYPE", "colab"),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "seed": args.seed,
        "steps_per_phase": args.steps,
        "lr": 0.002,
        "data_sha256": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest(),
        "split_sizes": {k: len(v) for k, v in data.items()},
        "qualification": "No retention, transfer or self-modification claim",
        "arms": {},
    }
    (out / "fixtures.json").write_text(json.dumps(data, indent=2))
    for mode in ("selective_plasticity", "trainable_backbone", "frozen_backbone"):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model = IntegratedStudentM104_E(
            TextStudentM103(prune_orphaned_lm_head=True),
            mode=mode,
            default_strategy="question_only",
        ).cuda()
        # Zero decay isolates optimization from known legacy direct-AdamW grouping.
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.002, weight_decay=0
        )
        arm = {"initial_hash": digest(model.state_dict()), "optimizer_steps": 0, "curve": []}
        torch.cuda.synchronize()
        start = time.perf_counter()
        for phase in ("tiny", "train"):
            tokens, actions, tags, states = batch[phase]
            for step in range(1, args.steps + 1):
                model.train()
                opt.zero_grad(set_to_none=True)
                logits, attr, _, _ = model(tokens, actions)
                loss = F.binary_cross_entropy_with_logits(attr, tags)
                if phase == "train":
                    loss = loss + F.cross_entropy(logits, states)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                if phase == "tiny" and step == 1:
                    arm["nonzero_gradient_names"] = [
                        name
                        for name, p in model.named_parameters()
                        if p.grad is not None and p.grad.abs().sum().item() > 0
                    ]
                    arm["optimizer_covers_trainable"] = {
                        id(p) for g in opt.param_groups for p in g["params"]
                    } == {id(p) for p in model.parameters() if p.requires_grad}
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                arm["optimizer_steps"] += 1
                if step in (1, 10, 50, 100, args.steps):
                    arm["curve"].append(
                        {
                            "phase": phase,
                            "step": step,
                            "loss": loss.item(),
                            "train": evaluate(model, batch[phase]),
                        }
                    )
            arm[phase] = evaluate(model, batch[phase])
            if phase == "tiny" and arm[phase]["micro_f1"] < 0.95:
                arm["stop_reason"] = "Tiny-batch acquisition failed; generalization not attempted"
                break
        if "train" in arm:
            # Dev and sealed are observed only after fixed training; no selection.
            arm["dev"] = evaluate(model, batch["dev"])
            arm["sealed"] = evaluate(model, batch["sealed"])
            # Text relevance ablation: roll prompts while keeping action/targets fixed.
            t, a, y, s = batch["sealed"]
            arm["mismatched_text"] = evaluate(model, (t.roll(2, 0), a, y, s))
        torch.cuda.synchronize()
        arm["gpu_wall_seconds"] = time.perf_counter() - start
        arm["final_hash"] = digest(model.state_dict())
        torch.save(model.state_dict(), out / f"{mode}.pt")
        report["arms"][mode] = arm
        (out / "report.json").write_text(json.dumps(report, indent=2))
        del model, opt
    report["status"] = "COMPLETED_FEASIBILITY_ONLY"
    (out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
