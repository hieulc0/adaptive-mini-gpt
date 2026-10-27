"""Development-only mechanism experiments; no sealed qualification or promotion."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from run_attribute_learning import encode, evaluate, file_hash, words
from run_concept_binding import language_model
from run_foundational_audit import digest

from adaptive_mini_gpt.data.attribute_diagnostic import CELLS, factorial
from adaptive_mini_gpt.data.attribute_learning import TAG_IDS, AttributeCase
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103
from adaptive_mini_gpt.training.learning_state import LearningState
from adaptive_mini_gpt.training.optimizer_contract import optimizer_membership
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda
from adaptive_mini_gpt.training.reproducibility import seed_all

LATEST = "9b4d8a937e553c9d27befbd218612724e5fb43909f632f8ebbb988ed224c4f3e"
ORIGINAL = "4a4f79fcd07b5bbe58bad5da04ce1a934696438c8603e9e6296e398a4b5ed5d8"


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2))


def parameter_order(owner):
    names = {id(p): n for n, p in owner.model.named_parameters()}
    return [[names[id(p)] for p in g["params"]] for g in owner.optimizer.param_groups]


def make_arm(parent, inherited_weights, inherited_moments):
    seed_all(42)
    model = (
        copy.deepcopy(parent.model) if inherited_weights else language_model("selective_plasticity")
    )
    owner = LearningState(model)
    owner.set_mode("selective_plasticity")
    if parameter_order(owner) != parameter_order(parent):
        raise ValueError("Optimizer mapping is not name-identical")
    if inherited_moments:
        owner.optimizer.load_state_dict(copy.deepcopy(parent.optimizer.state_dict()))
    metadata: dict[str, Any] = copy.deepcopy(parent.metadata) if inherited_weights else {"steps": 0}
    owner.metadata = metadata
    owner.metadata["diagnostic_lineage"] = {
        "inherited_weights": inherited_weights,
        "inherited_moments": inherited_moments,
        "parent_sha256": LATEST,
        "history_claim": "mechanism fork, not budget-matched transfer",
    }
    owner.external = (
        copy.deepcopy(parent.external)
        if inherited_weights
        else {"replay": [], "memory": {}, "controller": {}}
    )
    owner.external["sampler"] = random.Random(42)
    seed_all(42)  # Match new-stage stochastic streams after construction/copy.
    return owner


def gradient_audit(owner, batch):
    model = owner.model
    model.eval()
    _, tags, _, _ = model(batch[0], batch[1])
    inactive = [i for i in range(26) if i not in TAG_IDS]
    active = F.binary_cross_entropy_with_logits(tags[:, TAG_IDS], batch[2][:, TAG_IDS])
    absent = F.binary_cross_entropy_with_logits(tags[:, inactive], batch[2][:, inactive])
    all26 = F.binary_cross_entropy_with_logits(tags, batch[2])
    assert torch.allclose(all26, (2 * active + 24 * absent) / 26)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    ga = torch.autograd.grad(active, [p for _, p in named], retain_graph=True, allow_unused=True)
    gi = torch.autograd.grad(absent, [p for _, p in named], retain_graph=False, allow_unused=True)

    def shared(n):
        return n.startswith("student.") or "concept_proj" in n

    all_active = torch.cat([g.flatten() for g in ga if g is not None])
    all_inactive = torch.cat(
        [
            (g if g is not None else torch.zeros_like(p)).flatten()
            for (_, p), g, h in zip(named, gi, ga, strict=True)
            if h is not None
        ]
    )
    a = torch.cat(
        [g.flatten() for (n, _), g in zip(named, ga, strict=True) if g is not None and shared(n)]
    )
    b = torch.cat(
        [
            (g if g is not None else torch.zeros_like(p)).flatten()
            for (n, p), g, h in zip(named, gi, ga, strict=True)
            if h is not None and shared(n)
        ]
    )
    return {
        "active_loss": float(active.detach()),
        "inactive_loss": float(absent.detach()),
        "all26_loss": float(all26.detach()),
        "per_label_loss": F.binary_cross_entropy_with_logits(tags, batch[2], reduction="none")
        .mean(0)
        .detach()
        .tolist(),
        "active_total_norm": float(all_active.norm()),
        "inactive_total_norm": float(all_inactive.norm()),
        "all26_total_norm": float((all_active / 13 + all_inactive * 12 / 13).norm()),
        "active_shared_norm": float(a.norm()),
        "inactive_shared_norm": float(b.norm()),
        "shared_cosine": float(F.cosine_similarity(a, b, dim=0)),
        "all26_active_gradient_coefficient": 2 / 26,
        "all26_inactive_gradient_coefficient": 24 / 26,
        "flow": {
            n: {
                "active_norm": float(g.norm()) if g is not None else None,
                "inactive_norm": float(h.norm()) if h is not None else None,
            }
            for (n, _), g, h in zip(named, ga, gi, strict=True)
        },
        "frozen_names": [n for n, p in model.named_parameters() if not p.requires_grad],
        "optimizer_membership": optimizer_membership(model, owner.optimizer),
    }


def pooling_trace(model, batch, rows, tokenizer):
    """Observed text span ablation only; labels/statuses never define its mask."""
    model.eval()
    pooled = []
    concepts = []
    content_logits = []
    details = []
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            tokens = batch[0][start : start + 32]
            hidden = model.extract_sequence_hidden_states(tokens)
            rep, _ = model.pool_representations(hidden, tokens)
            z = model.bottleneck.concept_proj(rep)
            pooled.extend(rep.tolist())
            concepts.extend(z.tolist())
            for j, row in enumerate(rows[start : start + 32]):
                ids = tokens[j].tolist()
                q = ids.index(tokenizer.QUESTION_ID)
                begin = q + 1 + len(row.question[: row.question.index(row.content)].encode())
                end = begin + len(row.content.encode())
                text_rep = hidden[j, begin:end].mean(0)
                out = model.bottleneck.attribute_head(model.bottleneck.concept_proj(text_rep))
                content_logits.append(out[list(TAG_IDS)].tolist())
                details.append(
                    {
                        "content_start": begin,
                        "content_end": end,
                        "pool_start": q,
                        "pool_end": ids.index(tokenizer.ANSWER_ID),
                        "content_fraction": (end - begin) / (ids.index(tokenizer.ANSWER_ID) - q),
                    }
                )
    return {
        "pooled": pooled,
        "concept": concepts,
        "observed_span_pool_logits_DIAGNOSTIC_ONLY": content_logits,
        "pool_masks": details,
    }


def attention_trace(model, batch, row, tokenizer):
    module = model.student.base_model.blocks[-1].attn
    if module.pos_encoding != "absolute":
        raise ValueError("Inspector only validated for absolute positions")
    captured = {}

    def before(m, args):
        captured["x"] = args[0].detach()

    def after(m, args, out):
        captured["out"] = out.detach()

    a = module.register_forward_pre_hook(before)
    b = module.register_forward_hook(after)
    model.eval()
    try:
        with torch.no_grad():
            model(batch[0], batch[1])
    finally:
        a.remove()
        b.remove()
    with torch.no_grad():
        x = captured["x"]
        B, T, D = x.shape
        q, k, v = [
            t.view(B, T, module.n_heads, module.head_dim).transpose(1, 2)
            for t in module.c_attn(x).chunk(3, -1)
        ]
        weights = (
            ((q @ k.transpose(-1, -2)) / (module.head_dim**0.5))
            .masked_fill(~torch.ones(T, T, device=x.device, dtype=torch.bool).tril(), float("-inf"))
            .softmax(-1)
        )
        actual = module.c_proj((weights @ v).transpose(1, 2).contiguous().view(B, T, D))
        delta = float((actual - captured["out"]).abs().max())
        assert torch.allclose(actual, captured["out"], atol=1e-4, rtol=1e-4)
        ids = batch[0][0].tolist()
        qpos = ids.index(tokenizer.QUESTION_ID)
        apos = ids.index(tokenizer.ANSWER_ID)
        start = qpos + 1 + len(row.question[: row.question.index(row.content)].encode())
        end = start + len(row.content.encode())
        relevance = weights[0, :, qpos:apos].mean((0, 1))
        regions = {
            "rule": list(range(ids.index(tokenizer.RULE_ID) + 1, ids.index(tokenizer.SEP_ID))),
            "property_text": list(range(start, end)),
            "frame_and_name": list(range(qpos + 1, start)) + list(range(end, apos)),
        }
        return {
            "reconstruction_max_abs": delta,
            "pool_size": apos - qpos,
            "attention_mass": {name: float(relevance[ix].sum()) for name, ix in regions.items()},
            "key_attention": relevance.tolist(),
            "tokens": ids,
            "question": row.question,
            "scope": "last-layer attention averaged across pooled query positions/heads; inspection, not causal explanation",
        }


def features(rows, vocabulary):
    counts = [Counter(words(row, "ngram_linear")) for row in rows]
    return torch.tensor([[float(c[w]) for w in vocabulary] for c in counts], device="cuda")


def run(out, latest_path, original_path):
    if file_hash(latest_path) != LATEST or file_hash(original_path) != ORIGINAL:
        raise ValueError("Incorrect immutable parent artifacts")
    parent = LearningState.load(latest_path)
    if parent.metadata["steps"] != 400:
        raise ValueError("Latest history must have400 updates")
    splits = factorial()
    tokenizer = ByteTextTokenizerM103()
    context = parent.model.student.base_model.config.context_length
    batches = {}
    seqs = {}
    for name, rows in splits.items():
        batches[name], seqs[name] = encode(rows, tokenizer, context)
    seen_ids = {i for seq in seqs["train"] for i in seq}
    unseen_ids = {cell: sorted({i for seq in seqs[cell] for i in seq} - seen_ids) for cell in CELLS}
    if unseen_ids["familiar_novel"]:
        raise ValueError("Template diagnostic must not confound unseen byte tokens")
    for cell in CELLS:
        assert len(splits[cell]) == len(splits["train"])
        for a, b in zip(splits["train"], splits[cell], strict=True):
            assert (a.content, a.labels, a.matched_id) == (b.content, b.labels, b.matched_id)
    old_tiny = [
        AttributeCase(**row)
        for row in json.loads(Path("data/fixtures/stage_a_original_tiny.json").read_text())["cases"]
    ]
    old_batch, old_seqs = encode(old_tiny, tokenizer, context)
    data = {s: [asdict(r) for r in rows] for s, rows in splits.items()}
    data_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    save_json(out / "cases.json", data)
    save_json(out / "token_sequences.json", seqs)
    result = {
        "protocol": json.loads(Path("data/specs/stage_a_diagnostic_v1.json").read_text()),
        "spec_sha256": file_hash("data/specs/stage_a_diagnostic_v1.json"),
        "scope": "development only; no sealed labels, promotion or transfer qualification",
        "data_hash": data_hash,
        "parent_latest": LATEST,
        "parent_original": ORIGINAL,
        "neural": {},
        "loss_controls": {},
        "baselines": {},
        "provenance": {
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "platform": require_remote_cuda(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "seed": 42,
        },
        "budget": {
            "tiny": 200,
            "main": 800,
            "batch_max": 16,
            "lr": 0.002,
            "weight_decay": 0,
            "clip_norm": 1,
        },
        "input": {
            "cases": sum(map(len, splits.values())),
            "max_tokens": max(len(v) for seq in seqs.values() for v in seq),
            "context": context,
            "all_four_cells_matched": True,
            "unseen_token_ids_by_cell": unseen_ids,
        },
    }

    def store():
        save_json(out / "results.json", result)

    for weights, moments in ((True, True), (True, False), (False, True), (False, False)):
        name = (
            ("inherited" if weights else "fresh")
            + "_weights_"
            + ("inherited" if moments else "fresh")
            + "_moments"
        )
        owner = make_arm(parent, weights, moments)
        owner.metadata.update(
            stage="A_development",
            data_hash=data_hash,
            git_sha=result["provenance"]["git_sha"],
            objective={"attribute_ids": TAG_IDS, "scale": 1.0},
        )
        arm = {
            "stored_parameters": sum(p.numel() for p in owner.model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in owner.model.parameters() if p.requires_grad
            ),
            "initial_hash": digest(owner.model.state_dict()),
            "initial_steps": owner.metadata["steps"],
            "initial_adam_step_counts": dict(
                Counter(int(v["step"]) for v in owner.optimizer.state.values() if "step" in v)
            ),
            "updates": 0,
            "examples": 0,
            "tokens": 0,
            "curves": [],
            "gradient_initial": gradient_audit(owner, batches["tiny"]),
        }
        torch.cuda.synchronize()
        start = time.perf_counter()
        for phase, updates in (("tiny", 200), ("train", 800)):
            batch = batches[phase]
            for step in range(updates):
                ix = owner.external["sampler"].sample(
                    range(len(splits[phase])), min(16, len(splits[phase]))
                )
                selected = [v[ix] for v in batch]
                loss = owner.step(*selected, attribute_ids=TAG_IDS)
                arm["updates"] += 1
                arm["examples"] += len(ix)
                arm["tokens"] += int((selected[0] != tokenizer.PAD_ID).sum())
                owner.metadata.update(
                    phase=phase,
                    phase_step=step + 1,
                    stage_examples=arm["examples"],
                    stage_tokens=arm["tokens"],
                )
                if (step + 1) % 100 == 0:
                    arm["curves"].append(
                        {
                            "updates": arm["updates"],
                            "examples": arm["examples"],
                            "loss": loss,
                            "training_exact": evaluate(owner.model, batch, splits[phase])["exact"],
                        }
                    )
            arm[phase] = evaluate(owner.model, batch, splits[phase])
            arm[phase + "_hash"] = digest(owner.model.state_dict())
            arm["gradient_" + phase] = gradient_audit(owner, batches["tiny"])
        arm["cells"] = {cell: evaluate(owner.model, batches[cell], splits[cell]) for cell in CELLS}
        arm["representations"] = {
            cell: pooling_trace(owner.model, batches[cell], splits[cell], tokenizer)
            for cell in CELLS
        }
        reference = torch.tensor(arm["representations"]["familiar_familiar"]["pooled"])
        zref = torch.tensor(arm["representations"]["familiar_familiar"]["concept"])
        arm["matched_shift"] = {
            cell: {
                "pooled_cosine_mean": float(
                    F.cosine_similarity(
                        reference, torch.tensor(arm["representations"][cell]["pooled"]), dim=-1
                    ).mean()
                ),
                "concept_cosine_mean": float(
                    F.cosine_similarity(
                        zref, torch.tensor(arm["representations"][cell]["concept"]), dim=-1
                    ).mean()
                ),
            }
            for cell in CELLS
        }
        arm["attention"] = {}
        for cell in CELLS:
            ix = next(i for i, r in enumerate(splits[cell]) if r.labels == (1, 1))
            arm["attention"][cell] = attention_trace(
                owner.model, [v[ix : ix + 1] for v in batches[cell]], splits[cell][ix], tokenizer
            )
        torch.cuda.synchronize()
        arm["gpu_wall_seconds"] = time.perf_counter() - start
        arm["final_hash"] = digest(owner.model.state_dict())
        arm["total_steps"] = owner.metadata["steps"]
        owner.save(out / f"{name}.pt")
        arm["checkpoint_sha256"] = file_hash(out / f"{name}.pt")
        result["neural"][name] = arm
        store()
        del owner
    # Reproduce original tiny acquisition from the original TRAINING parent, not heldout.
    seed_all(42)
    original = LearningState.load(original_path)
    seed_all(42)
    original.external["sampler"] = random.Random(42)
    for _ in range(200):
        ix = original.external["sampler"].sample(range(len(old_tiny)), len(old_tiny))
        original.step(*[v[ix] for v in old_batch], attribute_ids=TAG_IDS)
    result["historical_training_reproduction"] = {
        "updates": 200,
        "metrics": evaluate(original.model, old_batch, old_tiny),
        "delivered_tokens": old_seqs,
        "final_hash": digest(original.model.state_dict()),
        "equals_archived_latest_tensor_hash": digest(original.model.state_dict())
        == digest(parent.model.state_dict()),
        "scope": "original training subset only, no prior heldout accessed",
    }
    del original
    for name, ids, scale in (
        ("active2", TAG_IDS, 1.0),
        ("active2_div13", TAG_IDS, 1 / 13),
        ("all26", None, 1.0),
        ("all26_times13", None, 13.0),
    ):
        owner = make_arm(parent, False, False)
        arm = {
            "initial_hash": digest(owner.model.state_dict()),
            "ids": ids,
            "loss_scale": scale,
            "gradient_initial": gradient_audit(owner, batches["tiny"]),
        }
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(200):
            ix = owner.external["sampler"].sample(range(len(splits["tiny"])), len(splits["tiny"]))
            owner.step(*[v[ix] for v in batches["tiny"]], attribute_ids=ids, loss_scale=scale)
        arm["tiny"] = evaluate(owner.model, batches["tiny"], splits["tiny"])
        arm["final_hash"] = digest(owner.model.state_dict())
        arm["gradient_final"] = gradient_audit(owner, batches["tiny"])
        torch.cuda.synchronize()
        arm["gpu_wall_seconds"] = time.perf_counter() - start
        arm["updates"] = 200
        arm["examples"] = 1800
        arm["scope"] = "200update development loss mechanism, not qualification"
        result["loss_controls"][name] = arm
        store()
        del owner
    seed_all(42)
    rng = random.Random(42)
    vocabulary = sorted({w for row in splits["train"] for w in words(row, "ngram_linear")})
    xs = {s: features(rows, vocabulary) for s, rows in splits.items()}
    model = torch.nn.Linear(len(vocabulary), 2).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0)
    torch.cuda.synchronize()
    baseline_start = time.perf_counter()
    for phase, steps in (("tiny", 200), ("train", 800)):
        for _ in range(steps):
            ix = rng.sample(range(len(splits[phase])), min(16, len(splits[phase])))
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                model(xs[phase][ix]), batches[phase][2][ix][:, TAG_IDS]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
    assert (
        result["loss_controls"]["active2"]["final_hash"]
        == result["neural"]["fresh_weights_fresh_moments"]["tiny_hash"]
    )
    result["loss_repetition_exact"] = True
    result["baselines"]["ngram_linear"] = {
        cell: evaluate(model, batches[cell], splits[cell], features=xs[cell]) for cell in CELLS
    }
    torch.cuda.synchronize()
    result["baseline_budget"] = {
        "updates": 1000,
        "examples": 14600,
        "gpu_wall_seconds": time.perf_counter() - baseline_start,
        "feature_dimensions": len(vocabulary),
    }
    result["verdict"] = "DEVELOPMENT_DIAGNOSTIC_NOT_SEALED_QUALIFICATION"
    result["progression"] = "STOP_STAGE_A; no automatic promotion, Stage B or structural evolution"
    store()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--latest", required=True)
    parser.add_argument("--original", required=True)
    args = parser.parse_args()
    require_remote_cuda()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run(out, Path(args.latest), Path(args.original))


if __name__ == "__main__":
    main()
