"""Paired development-only curriculum intervention; sealed evaluation is admission-gated."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from frame_a2_common import (
    ROOT_HASH,
    ROOT_TENSOR_HASH,
    fork_owner,
    record_stage_provenance,
    state_digest,
)
from run_attribute_learning import encode, evaluate, file_hash, words
from run_foundational_audit import digest
from run_stage_a_diagnostic import gradient_audit

from adaptive_mini_gpt.data.attribute_diagnostic import factorial
from adaptive_mini_gpt.data.attribute_learning import TAG_IDS
from adaptive_mini_gpt.data.frame_curriculum import (
    ARMS,
    draw_factors,
    evaluation_sets,
    training_case,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103
from adaptive_mini_gpt.training.learning_state import LearningState
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda

STEPS = 1200
BATCH = 16
SEEDS = (42, 43)


def brief(m):
    return {k: v for k, v in m.items() if k not in ("logits", "predictions", "truth")}


def vocabulary_features(rows, vocabulary):
    counts = [Counter(words(row, "ngram_linear")) for row in rows]
    return torch.tensor([[float(c[w]) for w in vocabulary] for c in counts], device="cuda")


def admitted(arm):
    final = arm["evaluation"]
    return (
        arm["training_probe"]["exact"] >= 0.95
        and final["novel_train"]["exact"] >= 0.95
        and all(
            final[s]["exact"] >= 0.9
            and min(final[s]["family_exact"].values()) >= 0.85
            and min(final[s]["status_accuracy"].values()) >= 0.95
            for s in ("known_dev", "novel_dev")
        )
        and min(arm["retention"][s]["exact"] for s in arm["retention"]) >= 0.95
        and all(not ids for ids in arm["unseen_tokens"].values())
    )


def run(out, root_path):
    if file_hash(root_path) != ROOT_HASH:
        raise ValueError("Wrong immutable1400-update parent")
    root = LearningState.load(root_path)
    if digest(root.model.state_dict()) != ROOT_TENSOR_HASH:
        raise ValueError("Wrong student weights")
    tokenizer = ByteTextTokenizerM103()
    context = root.model.student.base_model.config.context_length
    dev = evaluation_sets()
    batches = {}
    seqs = {}
    for split, rows in dev.items():
        batches[split], seqs[split] = encode(rows, tokenizer, context)
    old = factorial()
    retention = {s: old[s] for s in ("familiar_familiar", "novel_familiar")}
    rbatch = {s: encode(rows, tokenizer, context)[0] for s, rows in retention.items()}
    initial_retention = {s: evaluate(root.model, rbatch[s], rows) for s, rows in retention.items()}
    assert min(m["exact"] for m in initial_retention.values()) == 1.0
    (out / "development_cases.json").write_text(
        json.dumps({s: [asdict(r) for r in rows] for s, rows in dev.items()}, indent=2)
    )
    (out / "development_tokens.json").write_text(json.dumps(seqs))
    spec = json.loads(Path("data/specs/frame_a2_v1.json").read_text())
    result = {
        "protocol": spec,
        "provenance": {
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "platform": require_remote_cuda(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "parent_sha256": ROOT_HASH,
        "initial_retention": initial_retention,
        "arms": {},
        "baselines": {},
        "sealed_generated": False,
        "scope": "development qualification only until admission; no model promotion",
    }
    cache = {}

    def remember(rows, ids):
        for r, t in zip(rows, ids, strict=True):
            key = hashlib.sha256(r.question.encode()).hexdigest()
            value = {"question": r.question, "labels": r.labels, "tokens": t}
            if key in cache:
                assert cache[key] == value
            else:
                cache[key] = value

    def store():
        (out / "results.json").write_text(json.dumps(result, indent=2))

    for seed in SEEDS:
        stream_rng = random.Random(seed)
        stream = [draw_factors(stream_rng) for _ in range(STEPS * BATCH)]
        (out / f"factor_stream_{seed}.json").write_text(json.dumps(stream))
        for arm in ARMS:
            name = f"{arm}_{seed}"
            owner = fork_owner(root, seed)
            training_id = hashlib.sha256(
                json.dumps(
                    {
                        "spec_sha256": file_hash("data/specs/frame_a2_v1.json"),
                        "source": result["provenance"]["git_sha"],
                        "seed": seed,
                        "arm": arm,
                        "stream": stream,
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            record_stage_provenance(
                owner,
                training_sha=result["provenance"]["git_sha"],
                publication_sha=result["provenance"]["git_sha"],
                parent_hash=ROOT_HASH,
                data_hash=training_id,
                stage="A.2",
            )
            owner.metadata.update(
                stage="A.2",
                parent_sha256=ROOT_HASH,
                curriculum=arm,
                seed=seed,
                promoted=False,
                objective={"attribute_ids": TAG_IDS, "loss_scale": 1.0},
            )
            training_rows = [
                training_case(f, arm, f"{name}-probe-{i}") for i, f in enumerate(stream[:192])
            ]
            probe, _ = encode(training_rows, tokenizer, context)
            a = {
                "initial_hash": digest(owner.model.state_dict()),
                "initial_optimizer_hash": state_digest(owner.optimizer.state_dict()),
                "stored_parameters": sum(p.numel() for p in owner.model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in owner.model.parameters() if p.requires_grad
                ),
                "curves": [],
                "updates": 0,
                "examples": 0,
                "tokens": 0,
                "initial_evaluation": {
                    s: brief(evaluate(owner.model, batches[s], rows)) for s, rows in dev.items()
                },
            }
            prompt_hash = hashlib.sha256()
            semantic_hash = hashlib.sha256()
            seen = set()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for step in range(STEPS):
                factors = [draw_factors(owner.external["frame_a2_sampler"]) for _ in range(BATCH)]
                assert factors == stream[step * BATCH : (step + 1) * BATCH]
                rows = [training_case(f, arm, f"{name}-{step}-{j}") for j, f in enumerate(factors)]
                batch, ids = encode(rows, tokenizer, context)
                remember(rows, ids)
                for row, seq in zip(rows, ids, strict=True):
                    seen.update(seq)
                    prompt_hash.update(json.dumps(seq).encode())
                    semantic_hash.update(json.dumps((row.name, row.statuses, row.labels)).encode())
                loss = owner.step(*batch, attribute_ids=TAG_IDS)
                a["updates"] += 1
                a["examples"] += BATCH
                a["tokens"] += int((batch[0] != tokenizer.PAD_ID).sum())
                owner.metadata.update(
                    phase_step=step + 1, examples=a["examples"], tokens=a["tokens"]
                )
                if (step + 1) % 300 == 0:
                    a["curves"].append(
                        {
                            "updates": step + 1,
                            "examples": a["examples"],
                            "loss": loss,
                            "training_probe_exact": evaluate(owner.model, probe, training_rows)[
                                "exact"
                            ],
                            "development": {
                                s: brief(evaluate(owner.model, batches[s], rows))
                                for s, rows in dev.items()
                            },
                            "retention": {
                                s: brief(evaluate(owner.model, rbatch[s], rows))
                                for s, rows in retention.items()
                            },
                        }
                    )
            a["training_probe"] = evaluate(owner.model, probe, training_rows)
            a["evaluation"] = {
                s: evaluate(owner.model, batches[s], rows) for s, rows in dev.items()
            }
            a["retention"] = {
                s: evaluate(owner.model, rbatch[s], rows) for s, rows in retention.items()
            }
            a["unseen_tokens"] = {
                s: sorted({i for seq in seqs[s] for i in seq} - seen) for s in dev
            }
            a["gradient_training"] = gradient_audit(owner, [v[:16] for v in probe])
            a["prompt_stream_sha256"] = prompt_hash.hexdigest()
            a["semantic_stream_sha256"] = semantic_hash.hexdigest()
            owner.metadata.update(
                prompt_stream_sha256=prompt_hash.hexdigest(),
                semantic_stream_sha256=semantic_hash.hexdigest(),
            )
            a["development_admission"] = admitted(a)
            torch.cuda.synchronize()
            a["gpu_wall_seconds"] = time.perf_counter() - start
            a["final_hash"] = digest(owner.model.state_dict())
            a["lifetime_updates"] = owner.metadata["steps"]
            owner.save(out / f"{name}.pt")
            a["checkpoint_sha256"] = file_hash(out / f"{name}.pt")
            result["arms"][name] = a
            store()
            del owner
            # Strong raw-text baseline; train features come from this arm's actual stream only.
            all_rows = [training_case(f, arm) for f in stream]
            vocabulary = sorted({w for row in all_rows for w in words(row, "ngram_linear")})
            features = vocabulary_features(all_rows, vocabulary)
            truth = torch.tensor([r.labels for r in all_rows], device="cuda", dtype=torch.float32)
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(seed)
                model = torch.nn.Linear(len(vocabulary), 2).cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for step in range(STEPS):
                sl = slice(step * BATCH, (step + 1) * BATCH)
                opt.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(model(features[sl]), truth[sl])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
            base = {
                s: evaluate(model, batches[s], rows, features=vocabulary_features(rows, vocabulary))
                for s, rows in dev.items()
            }
            torch.cuda.synchronize()
            result["baselines"][name] = {
                "updates": STEPS,
                "examples": STEPS * BATCH,
                "feature_dimensions": len(vocabulary),
                "gpu_wall_seconds": time.perf_counter() - start,
                "evaluation": base,
            }
            torch.save(
                {"vocabulary": vocabulary, "weights": model.state_dict()}, out / f"ngram_{name}.pt"
            )
            store()
            del model, features, truth
        assert len({result["arms"][f"{arm}_{seed}"]["semantic_stream_sha256"] for arm in ARMS}) == 1
    primary = [result["arms"][f"multi_both_{seed}"] for seed in SEEDS]
    result["development_converged"] = all(a["development_admission"] for a in primary)
    if result["development_converged"]:
        sealed = evaluation_sets(sealed=True, admitted=True)
        result["sealed_generated"] = True
        (out / "sealed_cases.json").write_text(
            json.dumps({s: [asdict(r) for r in rows] for s, rows in sealed.items()}, indent=2)
        )
        result["sealed"] = {}
        for seed in SEEDS:
            owner = LearningState.load(out / f"multi_both_{seed}.pt")
            control = torch.load(out / f"ngram_multi_both_{seed}.pt", weights_only=False)
            vocab = control["vocabulary"]
            base = torch.nn.Linear(len(vocab), 2).cuda()
            base.load_state_dict(control["weights"])
            result["sealed"][str(seed)] = {}
            for s, rows in sealed.items():
                batch, ids = encode(rows, tokenizer, context)
                remember(rows, ids)
                result["sealed"][str(seed)][s] = {
                    "student": evaluate(owner.model, batch, rows),
                    "ngram": evaluate(base, batch, rows, features=vocabulary_features(rows, vocab)),
                }
        result["verdict"] = (
            "SEALED_EVALUATED_ONCE; language and baseline-saturation gates must be reported separately"
        )
    else:
        result["verdict"] = "DEVELOPMENT_FAIL_STOP; sealed qualification and Stage D NOT RUN"
    result["complete_input"] = {
        "unique_training_or_sealed_prompts": len(cache),
        "max_tokens": max(len(v["tokens"]) for v in cache.values()),
        "context": context,
        "development_max_tokens": max(len(t) for seq in seqs.values() for t in seq),
    }
    (out / "delivered_prompt_cache.json").write_text(json.dumps(cache))
    store()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    require_remote_cuda()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run(out, Path(a.root))


if __name__ == "__main__":
    main()
