"""REMOTE-REQUIRED: fixed original training subset, optimizer mechanism only."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import torch
from run_attribute_learning import encode, evaluate, file_hash
from run_concept_binding import language_model
from run_foundational_audit import digest
from run_stage_a_diagnostic import ORIGINAL, gradient_audit, parameter_order

from adaptive_mini_gpt.data.attribute_learning import TAG_IDS, AttributeCase
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103
from adaptive_mini_gpt.training.learning_state import LearningState
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda
from adaptive_mini_gpt.training.reproducibility import seed_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parent", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    platform_name = require_remote_cuda()
    if file_hash(args.parent) != ORIGINAL:
        raise ValueError("Original200-update parent required")
    parent = LearningState.load(args.parent)
    if parent.metadata["steps"] != 200:
        raise ValueError("Wrong parent history")
    rows = [
        AttributeCase(**r)
        for r in json.loads(Path("data/fixtures/stage_a_original_tiny.json").read_text())["cases"]
    ]
    tokenizer = ByteTextTokenizerM103()
    batch, seqs = encode(rows, tokenizer, parent.model.student.base_model.config.context_length)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "scope": "original9 training cases only, never viewed old heldout",
        "seed": 42,
        "parent_sha256": ORIGINAL,
        "new_updates": 200,
        "examples_per_arm": 1800,
        "lr": 0.002,
        "weight_decay": 0,
        "clip": 1,
        "tokens": seqs,
        "provenance": {
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "platform": platform_name,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "arms": {},
    }
    trials = (
        ("inherited_weights_inherited_adam", True, "inherit"),
        ("inherited_weights_fresh_adam", True, "fresh"),
        ("fresh_weights_inherited_adam", False, "inherit"),
        ("fresh_weights_fresh_adam", False, "fresh"),
        ("inherited_weights_reset_semantic_adam", True, "reset_semantic"),
        ("inherited_weights_reset_backbone_adam", True, "reset_backbone"),
    )
    for name, weights, strategy in trials:
        seed_all(42)
        owner = LearningState(
            copy.deepcopy(parent.model) if weights else language_model("selective_plasticity")
        )
        owner.set_mode("selective_plasticity")
        assert parameter_order(owner) == parameter_order(parent)
        if strategy != "fresh":
            owner.optimizer.load_state_dict(copy.deepcopy(parent.optimizer.state_dict()))
        cleared = []
        for n, param in owner.model.named_parameters():
            semantic = n.startswith("bottleneck.concept_proj.") or n.startswith(
                "bottleneck.attribute_head."
            )
            backbone = n.startswith("student.")
            if (strategy == "reset_semantic" and semantic) or (
                strategy == "reset_backbone" and backbone
            ):
                if param in owner.optimizer.state:
                    del owner.optimizer.state[param]
                    cleared.append(n)
        owner.metadata.update(
            parent_sha256=ORIGINAL,
            history_intervention=strategy,
            inherited_weights=weights,
            steps=200 if weights else 0,
            objective={"attribute_ids": TAG_IDS, "scale": 1.0},
            scope="unpromoted original-training-only mechanism",
        )
        owner.external["sampler"] = random.Random(42)
        seed_all(42)
        arm = {
            "initial_hash": digest(owner.model.state_dict()),
            "cleared_optimizer_state_names": cleared,
            "initial_metrics": evaluate(owner.model, batch, rows),
            "initial_gradient": gradient_audit(owner, batch),
            "curves": [],
        }
        torch.cuda.synchronize()
        start = time.perf_counter()
        for step in range(200):
            ix = owner.external["sampler"].sample(range(9), 9)
            loss = owner.step(*[v[ix] for v in batch], attribute_ids=TAG_IDS)
            if (step + 1) % 50 == 0:
                arm["curves"].append(
                    {
                        "updates": step + 1,
                        "loss": loss,
                        "exact": evaluate(owner.model, batch, rows)["exact"],
                    }
                )
        arm["final_metrics"] = evaluate(owner.model, batch, rows)
        arm["final_gradient"] = gradient_audit(owner, batch)
        torch.cuda.synchronize()
        arm["gpu_wall_seconds"] = time.perf_counter() - start
        arm["final_hash"] = digest(owner.model.state_dict())
        owner.save(out / f"{name}.pt")
        arm["checkpoint_sha256"] = file_hash(out / f"{name}.pt")
        result["arms"][name] = arm
        (out / "results.json").write_text(json.dumps(result, indent=2))
    result["verdict"] = (
        "MECHANISM_DIAGNOSTIC_ONLY; no heldout/retention/negative-transfer qualification"
    )
    (out / "results.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
