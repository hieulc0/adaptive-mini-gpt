"""M24: Reliable Neural Arithmetic Transitions and Deep Extrapolation Benchmark Runner.

Executes controlled 5-arm evaluation across 3 independent lineages (Seeds 42, 43, 44):
  - Arm A: M23 Recurrent Baseline Reference
  - Arm B: Canonical Structured State Encoding ([IN] a=A b=B c=C -> [OUT] d=D c=C_OUT)
  - Arm C: Balanced Elementary Transition Curriculum (Uniform coverage across all 200 transitions)
  - Arm D: Separate Output-Digit and Carry-State Supervision (C_OUT first conditioning)
  - Arm E: Best Configuration + Strictly Disjoint Prose Replay (10% replay)

Key Scientific Guarantees:
  1. Exhaustive 200-transition audit across all decimal combinations.
  2. First-failure column and error-type diagnostics.
  3. Generalization tested across 2–8, 16, and 32 digits, plus cascading carry chains.
  4. Zero Python arithmetic during neural inference.
  5. Strictly disjoint replay and retention evaluation corpora.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.rules.arithmetic_dataset import (
    ArithmeticProblem,
    generate_controlled_addition_dataset,
)
from adaptive_mini_gpt.rules.reliable_transition import (
    ElementaryTransition,
    ElementaryTransitionDataset,
    audit_elementary_transition_space,
    collate_elementary_batch,
    execute_reliable_recurrent_addition,
    get_all_200_transitions,
)
from adaptive_mini_gpt.training.checkpoint import CheckpointManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m24_reliable_transitions")

# Audited Retention Benchmark Suite (Disjoint from training replay)
EXPANDED_RETENTION_CORPUS = {
    "prose_original": (
        "Paris is the capital of France. William Shakespeare wrote Hamlet, Macbeth, and Othello in London. "
        "The speed of light in vacuum is approximately 300,000 km/s. "
        "In quantum physics, photons, bosons, and fermions govern fundamental interactions. "
        "Tokyo is the capital of Japan, and Rome is in Italy."
    ),
    "technical_science": (
        "Thermodynamics defines the relationships between heat, work, temperature, and energy. "
        "The first law states that energy cannot be created or destroyed, only transformed. "
        "Entropy measures molecular disorder and increases in all irreversible processes. "
        "DNA transcription and translation synthesize polypeptides from nucleotide sequences."
    ),
    "geography_history": (
        "The Nile River flows north through northeastern Africa into the Mediterranean Sea. "
        "The Roman Empire expanded across Europe, North Africa, and the Middle East before dividing. "
        "The Industrial Revolution transformed manufacturing from manual labor to machine-driven factories."
    ),
    "mathematics_prose": (
        "Prime numbers are integers greater than one divisible only by one and themselves. "
        "Euclidean geometry explores planes, angles, triangles, and parallel postulates. "
        "Differential calculus computes instantaneous rates of change along smooth curves."
    ),
}

# Strictly Independent Replay Corpus for Arm E (Audited: Zero overlap with retention suite)
INDEPENDENT_REPLAY_CORPUS = [
    (
        "Photosynthesis is the process used by plants, algae and cyanobacteria to convert light energy "
        "into chemical energy that can later be released to fuel the organism's metabolic activities. "
        "This chemical energy is stored in carbohydrate molecules such as sugars and starches."
    ),
    (
        "The Amazon rainforest is a moist broadleaf tropical rainforest in the Amazon basin of South America. "
        "It represents over half of the planet's remaining rainforests and comprises the largest and most "
        "biodiverse tract of tropical rainforest in the world."
    ),
    (
        "Johannes Kepler published his three laws of planetary motion between 1609 and 1619. "
        "They describe how planets orbit the Sun in ellipses with the Sun at one focus, "
        "sweeping out equal areas in equal intervals of time."
    ),
    (
        "Plate tectonics is the scientific theory that Earth's outer shell is divided into several plates "
        "that glide over the mantle, the rocky inner layer above the core. The plates act like a rigid shell."
    ),
]


def evaluate_language_loss(model: MiniGPT, tokenizer: BPETokenizer, device: torch.device) -> dict[str, float]:
    """Evaluate cross-entropy loss on multi-corpus retention suite."""
    model.eval()
    losses: dict[str, float] = {}
    with torch.no_grad():
        for name, text in EXPANDED_RETENTION_CORPUS.items():
            toks = tokenizer.encode(text)
            if len(toks) > model.config.context_length:
                toks = toks[: model.config.context_length]
            inp = torch.tensor([toks], dtype=torch.long, device=device)
            logits, _ = model(inp)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inp[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(shift_logits.view(-1, model.config.vocab_size), shift_labels.view(-1))
            losses[name] = round(loss.item(), 4)
    return losses


def train_transition_agent(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    transitions: list[ElementaryTransition],
    val_transitions: list[ElementaryTransition],
    format_type: str,
    device: torch.device,
    epochs: int = 10,
    lr: float = 5e-4,
    batch_size: int = 16,
    replay_texts: list[str] | None = None,
    replay_ratio: float = 0.0,
    seed: int = 42,
) -> tuple[MiniGPT, list[dict[str, float]]]:
    """Train MiniGPT on elementary column transitions with target-only loss."""
    train_dataset = ElementaryTransitionDataset(
        transitions=transitions,
        tokenizer=tokenizer,
        format_type=format_type,
        context_length=model.config.context_length,
        replay_texts=replay_texts,
        replay_ratio=replay_ratio,
        seed=seed,
    )
    val_dataset = ElementaryTransitionDataset(
        transitions=val_transitions,
        tokenizer=tokenizer,
        format_type=format_type,
        context_length=model.config.context_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_elementary_batch(b, pad_token_id=0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_elementary_batch(b, pad_token_id=0),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best_val_loss = float("inf")
    best_state: dict[str, Any] | None = None
    history: list[dict[str, float]] = []

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        train_batches = 0

        for b_in, b_lab in train_loader:
            b_in = b_in.to(device)
            b_lab = b_lab.to(device)

            optimizer.zero_grad()
            logits, loss = model(b_in, targets=b_lab)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            train_batches += 1

        avg_train_loss = total_loss / max(1, train_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for b_in, b_lab in val_loader:
                b_in = b_in.to(device)
                b_lab = b_lab.to(device)
                _, vloss = model(b_in, targets=b_lab)
                val_loss += vloss.item()
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)
        history.append({"epoch": ep + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate_recurrent_addition_suite(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    problems: list[ArithmeticProblem],
    device: torch.device,
    format_type: str = "canonical_state",
    closed_loop: bool = True,
) -> dict[str, Any]:
    """Evaluate multi-digit addition suite with first-failure diagnostics."""
    total = len(problems)
    exact_count = 0
    all_transitions_correct_count = 0
    all_carries_correct_count = 0
    latencies: list[float] = []

    first_failure_cols: list[int] = []
    failure_types: dict[str, int] = {"write_digit": 0, "carry_out": 0, "both": 0}

    by_length: dict[int, dict[str, Any]] = {}

    for prob in problems:
        max_l = prob.max_length
        if max_l not in by_length:
            by_length[max_l] = {"total": 0, "exact": 0, "trans_ok": 0, "carries_ok": 0}
        by_length[max_l]["total"] += 1

        res = execute_reliable_recurrent_addition(
            model=model,
            tokenizer=tokenizer,
            prob=prob,
            device=device,
            format_type=format_type,
            closed_loop=closed_loop,
        )

        is_ex = res["is_exact"]
        trans_ok = res["all_transitions_correct"]
        carries_ok = res["all_carries_correct"]

        if is_ex:
            exact_count += 1
            by_length[max_l]["exact"] += 1

        if trans_ok:
            all_transitions_correct_count += 1
            by_length[max_l]["trans_ok"] += 1

        if carries_ok:
            all_carries_correct_count += 1
            by_length[max_l]["carries_ok"] += 1

        if not is_ex and res["first_failure_col"] is not None:
            first_failure_cols.append(res["first_failure_col"])
            ft = res.get("first_failure_type", "both")
            failure_types[ft] = failure_types.get(ft, 0) + 1

        latencies.append(res["latency_per_digit_ms"])

    by_len_res = {}
    for l, d in sorted(by_length.items()):
        cnt = d["total"]
        by_len_res[l] = {
            "total": cnt,
            "exact_accuracy": round(d["exact"] / max(1, cnt) * 100.0, 2),
            "transition_accuracy": round(d["trans_ok"] / max(1, cnt) * 100.0, 2),
            "carry_accuracy": round(d["carries_ok"] / max(1, cnt) * 100.0, 2),
        }

    avg_lat = sum(latencies) / max(1, len(latencies))

    return {
        "exact_accuracy": round(exact_count / max(1, total) * 100.0, 2),
        "transition_accuracy": round(all_transitions_correct_count / max(1, total) * 100.0, 2),
        "carry_accuracy": round(all_carries_correct_count / max(1, total) * 100.0, 2),
        "avg_latency_per_digit_ms": round(avg_lat, 2),
        "total_problems": total,
        "by_length": by_len_res,
        "failure_diagnostics": {
            "total_failures": total - exact_count,
            "failure_types": failure_types,
            "mean_first_failure_col": round(sum(first_failure_cols) / max(1, len(first_failure_cols)), 2)
            if first_failure_cols
            else None,
        },
    }


def generate_cascading_carry_suite() -> list[ArithmeticProblem]:
    """Generate high-difficulty consecutive carry chain problems (e.g. 9999 + 1, 99999999 + 1)."""
    suite: list[ArithmeticProblem] = []
    # Test lengths 4, 6, 8, 12, 16, 24, 32
    for l in (4, 6, 8, 12, 16, 24, 32):
        # 99...99 + 1 = 100...00
        a = int("9" * l)
        b = 1
        suite.append(ArithmeticProblem.create(a, b, split="cascading_9plus1"))

        # 99...98 + 3 = 100...01
        a2 = int("9" * (l - 1) + "8")
        b2 = 3
        suite.append(ArithmeticProblem.create(a2, b2, split="cascading_98plus3"))

        # Random dense carry chain
        half = "4" * (l // 2) + "5" * (l - l // 2)
        other = "5" * (l // 2) + "5" * (l - l // 2)
        suite.append(ArithmeticProblem.create(int(half), int(other), split="cascading_dense"))

    return suite


def generate_deep_extrapolation_suite(lengths: tuple[int, ...] = (16, 32), num_per_length: int = 50, seed: int = 8888) -> list[ArithmeticProblem]:
    """Generate verified deep extrapolation problems at 16 and 32 digits."""
    rng = random.Random(seed)
    suite: list[ArithmeticProblem] = []
    for l in lengths:
        min_v = 10 ** (l - 1)
        max_v = (10**l) - 1
        for _ in range(num_per_length):
            a = rng.randint(min_v, max_v)
            b = rng.randint(min_v, max_v)
            suite.append(ArithmeticProblem.create(a, b, split=f"deep_extrap_{l}d"))
    return suite


def run_m24_benchmark(
    base_checkpoints_dir: Path,
    output_dir: Path,
    device_str: str = "cuda",
    seeds: list[int] = [42, 43, 44],
    epochs: int = 10,
    lr: float = 5e-4,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Execute complete M24 Reliable Neural Arithmetic Transitions benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    logger.info("==================================================================")
    logger.info("M24: Reliable Neural Arithmetic Transitions & Deep Extrapolation")
    logger.info("Lineages: %s | Device: %s | Epochs: %d | LR: %e", seeds, device, epochs, lr)
    logger.info("==================================================================")

    # 1. Dataset Generation
    logger.info("Generating multi-digit evaluation suites (2-8, 16, 32 digits)...")
    # In-distribution (2-3 digits)
    test_in_dist = generate_controlled_addition_dataset(seed=3024, num_samples=200, lengths=(2, 3), split="test_id")
    # Extrapolation (4-8 digits)
    test_extrap_4to8: list[ArithmeticProblem] = []
    for l in (4, 5, 6, 7, 8):
        sub_probs = generate_controlled_addition_dataset(seed=4000 + l, num_samples=50, lengths=(l,), split=f"extrap_{l}d")
        test_extrap_4to8.extend(sub_probs)

    # Deep Extrapolation (16 and 32 digits!)
    test_deep_16_32 = generate_deep_extrapolation_suite(lengths=(16, 32), num_per_length=50, seed=8888)

    # Cascading Carry Challenge Suite
    test_cascading = generate_cascading_carry_suite()

    # 2. Elementary Transitions
    all_200 = get_all_200_transitions()
    # Partition for Arms A and B (160 train, 40 held-out)
    rng_split = random.Random(42)
    shuffled_bases = list(all_200)
    rng_split.shuffle(shuffled_bases)
    train_160 = shuffled_bases[:160]
    held_out_40 = shuffled_bases[160:]
    held_out_set = {(t.a, t.b, t.c_in) for t in held_out_40}

    # Samples for training (1200 items per epoch matching standard compute budget)
    def make_samples(pool: list[ElementaryTransition], n: int = 1200, seed: int = 1024) -> list[ElementaryTransition]:
        r = random.Random(seed)
        return [r.choice(pool) for _ in range(n)]

    samples_160 = make_samples(train_160, 1200)
    samples_200 = make_samples(all_200, 1200)  # Uniform balanced curriculum for Arm C, D, E

    logger.info("Dataset generation complete. In-dist: %d, Extrap 4-8: %d, Deep 16-32: %d, Cascading: %d",
                len(test_in_dist), len(test_extrap_4to8), len(test_deep_16_32), len(test_cascading))

    lineage_results: dict[str, Any] = {}

    for seed in seeds:
        seed_key = f"seed_{seed}"
        logger.info("==================================================================")
        logger.info("Processing Lineage %s", seed_key)
        logger.info("==================================================================")

        ckpt_path = base_checkpoints_dir / f"seed_{seed}" / "base_pretrain" / "best.pt"
        raw_ckpt = CheckpointManager.load_raw_checkpoint(ckpt_path)
        tokenizer = CheckpointManager.load_tokenizer(raw_ckpt)
        base_cfg_dict = raw_ckpt.get("model_config", {})
        if not isinstance(base_cfg_dict, dict):
            base_cfg_dict = base_cfg_dict.to_dict()

        cfg_dict = dict(base_cfg_dict)
        cfg_dict["pos_encoding"] = "rope"
        cfg = ModelConfig(**cfg_dict)

        lineage_eval: dict[str, Any] = {}

        # -------------------------------------------------------------
        # Arm A: M23 Recurrent Baseline Reference
        # -------------------------------------------------------------
        logger.info("--- Starting Arm A: M23 Baseline Reference [%s] ---", seed_key)
        torch.manual_seed(seed)
        model_a = MiniGPT(cfg)
        model_a.load_state_dict(raw_ckpt["model_state_dict"], strict=False)
        model_a.to(device)

        model_a, _ = train_transition_agent(
            model=model_a,
            tokenizer=tokenizer,
            transitions=samples_160,
            val_transitions=held_out_40,
            format_type="m23_baseline",
            device=device,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            seed=seed,
        )

        audit_a = audit_elementary_transition_space(model_a, tokenizer, device, format_type="m23_baseline", held_out_triples=held_out_set)
        eval_id_a = evaluate_recurrent_addition_suite(model_a, tokenizer, test_in_dist, device, format_type="m23_baseline")
        eval_ext_a = evaluate_recurrent_addition_suite(model_a, tokenizer, test_extrap_4to8, device, format_type="m23_baseline")
        eval_deep_a = evaluate_recurrent_addition_suite(model_a, tokenizer, test_deep_16_32, device, format_type="m23_baseline")
        eval_casc_a = evaluate_recurrent_addition_suite(model_a, tokenizer, test_cascading, device, format_type="m23_baseline")

        lineage_eval["arm_a_m23_reference"] = {
            "transition_audit": audit_a,
            "in_distribution": eval_id_a,
            "extrapolation_4to8": eval_ext_a,
            "deep_extrapolation_16_32": eval_deep_a,
            "cascading_carries": eval_casc_a,
        }

        # -------------------------------------------------------------
        # Arm B: Canonical Structured State Encoding
        # -------------------------------------------------------------
        logger.info("--- Starting Arm B: Canonical State Encoding [%s] ---", seed_key)
        torch.manual_seed(seed)
        model_b = MiniGPT(cfg)
        model_b.load_state_dict(raw_ckpt["model_state_dict"], strict=False)
        model_b.to(device)

        model_b, _ = train_transition_agent(
            model=model_b,
            tokenizer=tokenizer,
            transitions=samples_160,
            val_transitions=held_out_40,
            format_type="canonical_state",
            device=device,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            seed=seed,
        )

        audit_b = audit_elementary_transition_space(model_b, tokenizer, device, format_type="canonical_state", held_out_triples=held_out_set)
        eval_id_b = evaluate_recurrent_addition_suite(model_b, tokenizer, test_in_dist, device, format_type="canonical_state")
        eval_ext_b = evaluate_recurrent_addition_suite(model_b, tokenizer, test_extrap_4to8, device, format_type="canonical_state")
        eval_deep_b = evaluate_recurrent_addition_suite(model_b, tokenizer, test_deep_16_32, device, format_type="canonical_state")
        eval_casc_b = evaluate_recurrent_addition_suite(model_b, tokenizer, test_cascading, device, format_type="canonical_state")

        lineage_eval["arm_b_canonical_state"] = {
            "transition_audit": audit_b,
            "in_distribution": eval_id_b,
            "extrapolation_4to8": eval_ext_b,
            "deep_extrapolation_16_32": eval_deep_b,
            "cascading_carries": eval_casc_b,
        }

        # -------------------------------------------------------------
        # Arm C: Balanced Elementary Transition Curriculum
        # -------------------------------------------------------------
        logger.info("--- Starting Arm C: Balanced Curriculum (All 200 Transitions) [%s] ---", seed_key)
        torch.manual_seed(seed)
        model_c = MiniGPT(cfg)
        model_c.load_state_dict(raw_ckpt["model_state_dict"], strict=False)
        model_c.to(device)

        model_c, _ = train_transition_agent(
            model=model_c,
            tokenizer=tokenizer,
            transitions=samples_200,
            val_transitions=all_200,
            format_type="canonical_state",
            device=device,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            seed=seed,
        )

        audit_c = audit_elementary_transition_space(model_c, tokenizer, device, format_type="canonical_state")
        eval_id_c = evaluate_recurrent_addition_suite(model_c, tokenizer, test_in_dist, device, format_type="canonical_state")
        eval_ext_c = evaluate_recurrent_addition_suite(model_c, tokenizer, test_extrap_4to8, device, format_type="canonical_state")
        eval_deep_c = evaluate_recurrent_addition_suite(model_c, tokenizer, test_deep_16_32, device, format_type="canonical_state")
        eval_casc_c = evaluate_recurrent_addition_suite(model_c, tokenizer, test_cascading, device, format_type="canonical_state")

        # Teacher-forced carry comparison on 4-8 digits
        eval_tf_c = evaluate_recurrent_addition_suite(model_c, tokenizer, test_extrap_4to8, device, format_type="canonical_state", closed_loop=False)

        lineage_eval["arm_c_balanced_curriculum"] = {
            "transition_audit": audit_c,
            "in_distribution": eval_id_c,
            "extrapolation_4to8": eval_ext_c,
            "teacher_forced_4to8": eval_tf_c,
            "deep_extrapolation_16_32": eval_deep_c,
            "cascading_carries": eval_casc_c,
        }

        # -------------------------------------------------------------
        # Arm D: Separate Output-Digit and Carry-State Supervision
        # -------------------------------------------------------------
        logger.info("--- Starting Arm D: Separate Supervision (Carry First) [%s] ---", seed_key)
        torch.manual_seed(seed)
        model_d = MiniGPT(cfg)
        model_d.load_state_dict(raw_ckpt["model_state_dict"], strict=False)
        model_d.to(device)

        model_d, _ = train_transition_agent(
            model=model_d,
            tokenizer=tokenizer,
            transitions=samples_200,
            val_transitions=all_200,
            format_type="separate_supervision",
            device=device,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            seed=seed,
        )

        audit_d = audit_elementary_transition_space(model_d, tokenizer, device, format_type="separate_supervision")
        eval_id_d = evaluate_recurrent_addition_suite(model_d, tokenizer, test_in_dist, device, format_type="separate_supervision")
        eval_ext_d = evaluate_recurrent_addition_suite(model_d, tokenizer, test_extrap_4to8, device, format_type="separate_supervision")
        eval_deep_d = evaluate_recurrent_addition_suite(model_d, tokenizer, test_deep_16_32, device, format_type="separate_supervision")
        eval_casc_d = evaluate_recurrent_addition_suite(model_d, tokenizer, test_cascading, device, format_type="separate_supervision")

        lineage_eval["arm_d_separate_supervision"] = {
            "transition_audit": audit_d,
            "in_distribution": eval_id_d,
            "extrapolation_4to8": eval_ext_d,
            "deep_extrapolation_16_32": eval_deep_d,
            "cascading_carries": eval_casc_d,
        }

        # -------------------------------------------------------------
        # Arm E: Best Configuration + Strictly Disjoint Prose Replay
        # -------------------------------------------------------------
        logger.info("--- Starting Arm E: Balanced Curriculum + Audited Replay [%s] ---", seed_key)
        torch.manual_seed(seed)
        model_e = MiniGPT(cfg)
        model_e.load_state_dict(raw_ckpt["model_state_dict"], strict=False)
        model_e.to(device)

        retention_pre_e = evaluate_language_loss(model_e, tokenizer, device)

        model_e, _ = train_transition_agent(
            model=model_e,
            tokenizer=tokenizer,
            transitions=samples_200,
            val_transitions=all_200,
            format_type="canonical_state",
            device=device,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            replay_texts=INDEPENDENT_REPLAY_CORPUS,
            replay_ratio=0.1,
            seed=seed,
        )

        audit_e = audit_elementary_transition_space(model_e, tokenizer, device, format_type="canonical_state")
        eval_id_e = evaluate_recurrent_addition_suite(model_e, tokenizer, test_in_dist, device, format_type="canonical_state")
        eval_ext_e = evaluate_recurrent_addition_suite(model_e, tokenizer, test_extrap_4to8, device, format_type="canonical_state")
        eval_deep_e = evaluate_recurrent_addition_suite(model_e, tokenizer, test_deep_16_32, device, format_type="canonical_state")
        eval_casc_e = evaluate_recurrent_addition_suite(model_e, tokenizer, test_cascading, device, format_type="canonical_state")

        retention_post_e = evaluate_language_loss(model_e, tokenizer, device)
        drift_e = {k: round(retention_post_e[k] - retention_pre_e[k], 4) for k in retention_pre_e}
        mean_drift_e = round(sum(drift_e.values()) / len(drift_e), 4)

        lineage_eval["arm_e_audited_replay"] = {
            "transition_audit": audit_e,
            "in_distribution": eval_id_e,
            "extrapolation_4to8": eval_ext_e,
            "deep_extrapolation_16_32": eval_deep_e,
            "cascading_carries": eval_casc_e,
            "retention": {
                "pre": retention_pre_e,
                "post": retention_post_e,
                "drift": drift_e,
                "mean_drift": mean_drift_e,
            },
        }

        lineage_results[seed_key] = lineage_eval

    # Aggregating metrics across 3 lineages
    logger.info("Computing multi-lineage aggregate statistics...")

    def get_mean_std(values: list[float]) -> dict[str, float]:
        m = sum(values) / len(values)
        var = sum((x - m) ** 2 for x in values) / max(1, len(values) - 1)
        return {"mean": round(m, 2), "std": round(math.sqrt(var), 2)}

    aggregate: dict[str, Any] = {}

    for arm_k in ("arm_a_m23_reference", "arm_b_canonical_state", "arm_c_balanced_curriculum", "arm_d_separate_supervision", "arm_e_audited_replay"):
        arm_exact_trans = [lineage_results[f"seed_{s}"][arm_k]["transition_audit"]["overall_exact_acc"] for s in seeds]
        arm_carry_trans = [lineage_results[f"seed_{s}"][arm_k]["transition_audit"]["overall_carry_acc"] for s in seeds]
        arm_id_exact = [lineage_results[f"seed_{s}"][arm_k]["in_distribution"]["exact_accuracy"] for s in seeds]
        arm_ext_exact = [lineage_results[f"seed_{s}"][arm_k]["extrapolation_4to8"]["exact_accuracy"] for s in seeds]
        arm_deep_16 = [lineage_results[f"seed_{s}"][arm_k]["deep_extrapolation_16_32"]["by_length"][16]["exact_accuracy"] for s in seeds]
        arm_deep_32 = [lineage_results[f"seed_{s}"][arm_k]["deep_extrapolation_16_32"]["by_length"][32]["exact_accuracy"] for s in seeds]
        arm_casc_exact = [lineage_results[f"seed_{s}"][arm_k]["cascading_carries"]["exact_accuracy"] for s in seeds]

        arm_agg = {
            "overall_transition_exact": get_mean_std(arm_exact_trans),
            "overall_carry_exact": get_mean_std(arm_carry_trans),
            "in_distribution_exact": get_mean_std(arm_id_exact),
            "extrapolation_4to8_exact": get_mean_std(arm_ext_exact),
            "deep_extrapolation_16_exact": get_mean_std(arm_deep_16),
            "deep_extrapolation_32_exact": get_mean_std(arm_deep_32),
            "cascading_carries_exact": get_mean_std(arm_casc_exact),
        }

        if arm_k in ("arm_a_m23_reference", "arm_b_canonical_state"):
            held_exact = [lineage_results[f"seed_{s}"][arm_k]["transition_audit"]["by_split"]["held_out"]["exact_acc"] for s in seeds]
            held_carry = [lineage_results[f"seed_{s}"][arm_k]["transition_audit"]["by_split"]["held_out"]["carry_acc"] for s in seeds]
            arm_agg["held_out_transition_exact"] = get_mean_std(held_exact)
            arm_agg["held_out_carry_exact"] = get_mean_std(held_carry)

        if arm_k == "arm_c_balanced_curriculum":
            tf_exact = [lineage_results[f"seed_{s}"][arm_k]["teacher_forced_4to8"]["exact_accuracy"] for s in seeds]
            arm_agg["teacher_forced_4to8_exact"] = get_mean_std(tf_exact)

        if arm_k == "arm_e_audited_replay":
            e_drifts = [lineage_results[f"seed_{s}"][arm_k]["retention"]["mean_drift"] for s in seeds]
            arm_agg["mean_retention_drift"] = get_mean_std(e_drifts)

        aggregate[arm_k] = arm_agg

    report = {
        "milestone": "M24",
        "description": "Reliable Neural Arithmetic Transitions and Deep Extrapolation Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "config": {
            "seeds": seeds,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "device": str(device),
            "model_parameters": sum(p.numel() for p in model_a.parameters()),
        },
        "aggregate": aggregate,
        "lineages": lineage_results,
    }

    report_path = output_dir / "m24_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("==================================================================")
    logger.info("M24 Benchmark Completed Successfully! Saved to %s", report_path)
    logger.info("==================================================================")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="M24 Reliable Neural Arithmetic Transitions Benchmark")
    parser.add_argument("--base_checkpoints_dir", "--checkpoints-dir", type=Path, default=Path("checkpoints"), dest="base_checkpoints_dir")
    parser.add_argument("--output_dir", "--output-dir", type=Path, default=Path("runs/m24_reliable_transitions"), dest="output_dir")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=16, dest="batch_size")

    args = parser.parse_args()
    run_m24_benchmark(
        base_checkpoints_dir=args.base_checkpoints_dir,
        output_dir=args.output_dir,
        device_str=args.device,
        seeds=args.seeds,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
