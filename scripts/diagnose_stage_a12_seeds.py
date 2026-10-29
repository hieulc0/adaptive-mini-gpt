"""Stage A.12 Phase 1: Seed-Dependent Binding Forensic Diagnosis.

Loads saved immutable Stage A.11-R2 checkpoints for R2_ARM_2_LAST_TOKEN
across Seed 42, Seed 43, and Seed 44 on GPU (0 training updates) and conducts:
1. Error breakdown on MP-1..3 and EB-1..4 with query-sensitivity analysis.
2. Familiar training entity names vs unseen development entity names binding probe.
3. Attention routing forensics: Target-vs-Distractor attention margins across layers and heads.
4. Detached linear/ridge probe on h_last_question_token.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from run_attribute_learning import encode

from adaptive_mini_gpt.data.clause_minimal_pairs_a11 import (
    MinimalPairCase,
    build_semantic_minimal_pairs_suite,
    compute_paired_metrics_from_predictions,
    generate_mp2_multi_entity_binding,
)
from adaptive_mini_gpt.data.entity_binding_a11r2 import (
    DEV_ENTITIES_BY_SEED,
    TRAIN_ENTITIES,
    build_entity_binding_suite,
)
from adaptive_mini_gpt.model.attention import apply_rotary_emb
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)
from adaptive_mini_gpt.training.learning_state import LearningState
from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda


class FullCosineTwoTimescaleScheduler:
    """Cosine scheduler for both backbone and readout groups."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        backbone_lr_start: float = 0.0005,
        backbone_lr_min: float = 0.00005,
        readout_lr_start: float = 0.0015,
        readout_lr_min: float = 0.00015,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.bb_start = backbone_lr_start
        self.bb_min = backbone_lr_min
        self.rd_start = readout_lr_start
        self.rd_min = readout_lr_min
        self.step_num = 0

    def step(self) -> None:
        self.step_num += 1
        progress = min(self.step_num, self.total_steps) / self.total_steps
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        cur_bb_lr = self.bb_min + (self.bb_start - self.bb_min) * cos_factor
        cur_rd_lr = self.rd_min + (self.rd_start - self.rd_min) * cos_factor

        if len(self.optimizer.param_groups) > 0:
            self.optimizer.param_groups[0]["lr"] = cur_bb_lr
        if len(self.optimizer.param_groups) > 1:
            self.optimizer.param_groups[1]["lr"] = cur_rd_lr

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "step_num": self.step_num,
            "bb_start": self.bb_start,
            "bb_min": self.bb_min,
            "rd_start": self.rd_start,
            "rd_min": self.rd_min,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.total_steps = state["total_steps"]
        self.step_num = state["step_num"]
        self.bb_start = state["bb_start"]
        self.bb_min = state["bb_min"]
        self.rd_start = state["rd_start"]
        self.rd_min = state["rd_min"]


setattr(sys.modules["__main__"], "FullCosineTwoTimescaleScheduler", FullCosineTwoTimescaleScheduler)  # noqa: B010



def compute_query_sensitivity(
    pairs: list[tuple[Any, Any]],
    preds_by_case_id: dict[str, list[int]],
) -> dict[str, Any]:
    """Computes query sensitivity on pairs sharing identical prefixes with swapped query targets."""
    total_pairs = len(pairs)
    if total_pairs == 0:
        return {"n_pairs": 0, "query_sensitivity_rate": 0.0, "query_blind_rate": 0.0}

    sensitive_count = 0
    blind_count = 0

    for case_a, case_b in pairs:
        pred_a = preds_by_case_id[case_a.case_id]
        pred_b = preds_by_case_id[case_b.case_id]
        if pred_a != pred_b:
            sensitive_count += 1
        else:
            blind_count += 1

    return {
        "n_pairs": total_pairs,
        "query_sensitive_pairs": sensitive_count,
        "query_blind_pairs": blind_count,
        "query_sensitivity_rate": float(sensitive_count / total_pairs),
        "query_blind_rate": float(blind_count / total_pairs),
    }


def extract_layer_attention_weights(
    model: Any,
    tokens: torch.Tensor,
) -> list[torch.Tensor]:
    """Extracts exact causal attention weight matrices across all blocks: list of (B, H, T, T)."""
    base_model = model.student.evolvable_backbone.base_model
    b, t = tokens.size()

    tok_emb = base_model.wte(tokens)
    if base_model.wpe is not None:
        pos = torch.arange(0, t, dtype=torch.long, device=tokens.device)
        x = base_model.drop(tok_emb + base_model.wpe(pos))
    else:
        x = base_model.drop(tok_emb)

    layer_attns: list[torch.Tensor] = []

    for block in base_model.blocks:
        normed = block.ln_1(x)
        qkv = block.attn.c_attn(normed)
        q, k, _ = qkv.chunk(3, dim=-1)

        q = q.view(b, t, block.attn.n_heads, block.attn.head_dim).transpose(1, 2)
        k = k.view(b, t, block.attn.n_heads, block.attn.head_dim).transpose(1, 2)

        # Apply RoPE
        if block.attn.pos_encoding == "rope":
            if not hasattr(block.attn, "inv_freq"):
                inv_f = 1.0 / (
                    10000.0 ** (torch.arange(0, block.attn.head_dim, 2).float() / block.attn.head_dim)
                )
                block.attn.register_buffer("inv_freq", inv_f.to(device=x.device), persistent=False)
            t_pos = torch.arange(t, device=x.device, dtype=torch.float32)
            freqs = torch.outer(t_pos, block.attn.inv_freq.to(device=x.device))
            cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1).view(
                1, 1, t, block.attn.head_dim
            ).to(dtype=q.dtype)
            sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1).view(
                1, 1, t, block.attn.head_dim
            ).to(dtype=q.dtype)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(block.attn.head_dim)
        causal_mask = torch.tril(torch.ones(t, t, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, -1e4)
        attn_weights = torch.softmax(scores, dim=-1)
        layer_attns.append(attn_weights.detach())

        # Update representation through block
        x, _ = block(x)

    return layer_attns


def compute_target_distractor_attention_margin(
    model: Any,
    tokenizer: ByteTextTokenizerM103,
    cases: list[MinimalPairCase],
) -> dict[str, Any]:
    """Computes attention margin (target - distractor) from final query token."""
    device = next(model.parameters()).device
    model.eval()

    n_layers = len(model.student.evolvable_backbone.base_model.blocks)
    n_heads = model.student.evolvable_backbone.base_model.blocks[0].attn.n_heads

    margins_by_layer_head = np.zeros((n_layers, n_heads))
    case_count = 0

    with torch.no_grad():
        for case in cases:
            # We focus on MP-2 multi-entity cases
            if getattr(case, "family", "") != "MP-2":
                continue

            (batch, _, _), seqs = encode([case], tokenizer, 256)
            tokens = batch.to(device)
            layer_attns = extract_layer_attention_weights(model, tokens)

            row = tokens[0].tolist()
            q_id = ByteTextTokenizerM103.QUESTION_ID
            a_id = ByteTextTokenizerM103.ANSWER_ID

            q_locs = [i for i, tok in enumerate(row) if tok == q_id]
            a_locs = [i for i, tok in enumerate(row) if tok == a_id]
            if not q_locs or not a_locs:
                continue

            q_start = q_locs[0] + 1
            t_last = a_locs[0] - 1

            # Identify target entity vs distractor entity token positions in prefix
            meta = getattr(case, "metadata", {})
            target_entity = meta.get("query_target", "")
            other_entity = meta.get("other_target", "")

            if not target_entity or not other_entity:
                continue

            target_bytes = target_entity.encode("utf-8")
            other_bytes = other_entity.encode("utf-8")
            raw_q_bytes = bytes([t for t in row[q_start:t_last] if 0 <= t < 256])

            pos_target = raw_q_bytes.find(target_bytes)
            pos_other = raw_q_bytes.find(other_bytes)

            if pos_target == -1 or pos_other == -1:
                continue

            # Prefix positions
            idx_target = [q_start + pos_target + j for j in range(len(target_bytes))]
            idx_other = [q_start + pos_other + j for j in range(len(other_bytes))]

            for l_idx, attn_l in enumerate(layer_attns):
                # attn_l: (1, H, T, T)
                alpha_query = attn_l[0, :, t_last, :]  # (H, T)
                alpha_target = alpha_query[:, idx_target].mean(dim=-1).cpu().numpy()
                alpha_other = alpha_query[:, idx_other].mean(dim=-1).cpu().numpy()
                margins_by_layer_head[l_idx] += (alpha_target - alpha_other)

            case_count += 1

    if case_count > 0:
        margins_by_layer_head /= case_count

    return {
        "num_analyzed_cases": case_count,
        "margins_matrix": margins_by_layer_head.tolist(),
        "mean_margin_per_layer": margins_by_layer_head.mean(axis=1).tolist(),
        "overall_mean_margin": float(margins_by_layer_head.mean()),
    }


def evaluate_familiar_vs_unseen_entities(
    model: Any,
    tokenizer: ByteTextTokenizerM103,
    seed: int,
) -> dict[str, Any]:
    """Compares MP-2 & EB-1 binding accuracy on familiar training names vs unseen dev names."""
    device = next(model.parameters()).device
    model.eval()

    # 1. MP-2 with training entity names
    train_mp2_pairs = generate_mp2_multi_entity_binding(TRAIN_ENTITIES[:2], "probe_train")
    train_mp2_cases = [c for p in train_mp2_pairs for c in p]

    (b_train, _, _), _ = encode(train_mp2_cases, tokenizer, 256)
    with torch.no_grad():
        _, tag_logits, _, _ = model(b_train.to(device), torch.zeros(len(train_mp2_cases), device=device, dtype=torch.long))
        preds_train = (tag_logits[:, [6, 7]] >= 0).long().tolist()
    train_metrics = compute_paired_metrics_from_predictions(train_mp2_cases, preds_train)

    # 2. MP-2 with dev unseen entity names for this seed
    dev_entities = DEV_ENTITIES_BY_SEED.get(seed, DEV_ENTITIES_BY_SEED[42])
    dev_mp2_pairs = generate_mp2_multi_entity_binding(dev_entities[:2], "probe_dev")
    dev_mp2_cases = [c for p in dev_mp2_pairs for c in p]

    (b_dev, _, _), _ = encode(dev_mp2_cases, tokenizer, 256)
    with torch.no_grad():
        _, tag_logits_dev, _, _ = model(b_dev.to(device), torch.zeros(len(dev_mp2_cases), device=device, dtype=torch.long))
        preds_dev = (tag_logits_dev[:, [6, 7]] >= 0).long().tolist()
    dev_metrics = compute_paired_metrics_from_predictions(dev_mp2_cases, preds_dev)

    return {
        "familiar_training_entities": {
            "entities": TRAIN_ENTITIES[:2],
            "case_exact_acc": train_metrics["case_exact_acc"],
            "pair_both_correct_acc": train_metrics["pair_both_correct_acc"],
        },
        "unseen_dev_entities": {
            "entities": dev_entities[:2],
            "case_exact_acc": dev_metrics["case_exact_acc"],
            "pair_both_correct_acc": dev_metrics["pair_both_correct_acc"],
        },
        "generalization_gap": float(
            train_metrics["pair_both_correct_acc"] - dev_metrics["pair_both_correct_acc"]
        ),
    }


def fit_representation_ridge_probe(
    model: Any,
    tokenizer: ByteTextTokenizerM103,
    cases: list[Any],
) -> dict[str, float]:
    """Fits detached linear ridge probe on h_last_question_token to predict target property."""
    device = next(model.parameters()).device
    model.eval()

    (batch, _, _), _ = encode(cases, tokenizer, 256)
    tokens = batch.to(device)

    with torch.no_grad():
        h_seq = model.extract_sequence_hidden_states(tokens)
        taps = model.extract_tapped_representations(h_seq, tokens)
        h_last = taps["h_last_question_token"]

    y_flammable = torch.tensor([c.labels[0] for c in cases], device=device)
    y_conductive = torch.tensor([c.labels[1] for c in cases], device=device)

    n_samples = len(cases)
    split_idx = n_samples // 2

    def _solve_ridge(X_tr: torch.Tensor, y_tr: torch.Tensor, X_te: torch.Tensor, y_te: torch.Tensor, alpha: float = 1.0) -> float:
        y_tr_f = (y_tr.float() * 2.0) - 1.0
        y_te_f = (y_te.float() * 2.0) - 1.0

        ones_tr = torch.ones((X_tr.size(0), 1), device=device)
        X_tr_b = torch.cat([X_tr, ones_tr], dim=-1)
        ones_te = torch.ones((X_te.size(0), 1), device=device)
        X_te_b = torch.cat([X_te, ones_te], dim=-1)

        d = X_tr_b.size(1)
        reg = alpha * torch.eye(d, device=device)
        reg[-1, -1] = 0.0

        w = torch.linalg.solve(X_tr_b.t() @ X_tr_b + reg, X_tr_b.t() @ y_tr_f)
        preds = (X_te_b @ w >= 0.0)
        return float((preds == (y_te_f > 0)).float().mean().item())

    acc_flam = _solve_ridge(h_last[:split_idx], y_flammable[:split_idx], h_last[split_idx:], y_flammable[split_idx:])
    acc_cond = _solve_ridge(h_last[:split_idx], y_conductive[:split_idx], h_last[split_idx:], y_conductive[split_idx:])

    return {
        "probe_flammable_test_acc": acc_flam,
        "probe_conductive_test_acc": acc_cond,
        "probe_mean_test_acc": float(0.5 * (acc_flam + acc_cond)),
    }


def run_phase1_forensics(checkpoint_dir: Path, out_json: Path) -> dict[str, Any]:
    """Executes the full Phase 1 mechanistic diagnosis across Seeds 42, 43, 44."""
    require_remote_cuda()
    device = torch.device("cuda")
    tokenizer = ByteTextTokenizerM103()

    seeds = [42, 43, 44]
    results: dict[str, Any] = {}

    for seed in seeds:
        chk_candidates = list(
            checkpoint_dir.rglob(f"r2_arm_2_last_token_seed_{seed}_7200.pt")
        )
        if not chk_candidates:
            # Fall back to root /kaggle/input search
            chk_candidates = list(
                Path("/kaggle/input").rglob(f"r2_arm_2_last_token_seed_{seed}_7200.pt")
            )
        if not chk_candidates:
            raise FileNotFoundError(
                f"Could not locate r2_arm_2_last_token_seed_{seed}_7200.pt in {checkpoint_dir} or /kaggle/input"
            )

        chk_path = chk_candidates[0]
        print(f"Loading checkpoint for Seed {seed}: {chk_path}")
        owner = LearningState.load(chk_path)
        model = owner.model.to(device)
        model.eval()

        # 1. Dev Minimal Pairs Suite
        mp_suite = build_semantic_minimal_pairs_suite()
        dev_mp_cases = mp_suite["dev_cases"]
        (b_mp, _, _), _ = encode(dev_mp_cases, tokenizer, 256)
        with torch.no_grad():
            _, tag_logits, _, _ = model(b_mp.to(device), torch.zeros(len(dev_mp_cases), device=device, dtype=torch.long))
            mp_preds = (tag_logits[:, [6, 7]] >= 0).long().tolist()

        preds_mp_dict = {c.case_id: p for c, p in zip(dev_mp_cases, mp_preds, strict=True)}
        mp_metrics = compute_paired_metrics_from_predictions(dev_mp_cases, mp_preds)

        # Compute query sensitivity on MP-2
        mp2_pairs = mp_suite["dev_pairs_by_family"]["MP-2"]
        mp2_sensitivity = compute_query_sensitivity(mp2_pairs, preds_mp_dict)

        # 2. Dev Entity Binding Suite
        eb_suite = build_entity_binding_suite(seed=seed, split="dev")
        dev_eb_cases = eb_suite["cases"]
        (b_eb, _, _), _ = encode(dev_eb_cases, tokenizer, 256)
        with torch.no_grad():
            _, tag_logits_eb, _, _ = model(b_eb.to(device), torch.zeros(len(dev_eb_cases), device=device, dtype=torch.long))
            eb_preds = (tag_logits_eb[:, [6, 7]] >= 0).long().tolist()

        preds_eb_dict = {c.case_id: p for c, p in zip(dev_eb_cases, eb_preds, strict=True)}
        eb_metrics = compute_paired_metrics_from_predictions(dev_eb_cases, eb_preds)

        eb1_pairs = eb_suite["pairs_by_family"]["EB-1_QUERY_SWAP"]
        eb1_sensitivity = compute_query_sensitivity(eb1_pairs, preds_eb_dict)

        # 3. Familiar vs. Unseen Entity Probe
        fam_vs_unseen = evaluate_familiar_vs_unseen_entities(model, tokenizer, seed)

        # 4. Attention Routing Analysis
        attn_margins = compute_target_distractor_attention_margin(model, tokenizer, dev_mp_cases)

        # 5. Hidden Representation Ridge Probe
        probe_res = fit_representation_ridge_probe(model, tokenizer, dev_mp_cases)

        results[f"seed_{seed}"] = {
            "checkpoint": str(chk_path),
            "mp_metrics": {
                "overall": {
                    "case_exact_acc": mp_metrics["case_exact_acc"],
                    "pair_both_correct_acc": mp_metrics["pair_both_correct_acc"],
                    "pair_counterfactual_flip_acc": mp_metrics["pair_counterfactual_flip_acc"],
                },
                "MP-1": mp_metrics["by_family"]["MP-1"],
                "MP-2": mp_metrics["by_family"]["MP-2"],
                "MP-3": mp_metrics["by_family"]["MP-3"],
            },
            "eb_metrics": {
                "overall": {
                    "case_exact_acc": eb_metrics["case_exact_acc"],
                    "pair_both_correct_acc": eb_metrics["pair_both_correct_acc"],
                },
                "by_family": eb_metrics["by_family"],
            },
            "query_sensitivity": {
                "MP-2": mp2_sensitivity,
                "EB-1": eb1_sensitivity,
            },
            "familiar_vs_unseen_entities": fam_vs_unseen,
            "attention_margins": attn_margins,
            "representation_ridge_probe": probe_res,
        }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Phase 1 diagnosis saved to {out_json}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Stage A.12 Phase 1 Seed Diagnosis")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--out-json", type=Path, default=Path("stage_a12_phase1_diagnosis.json"))
    args = parser.parse_args()

    run_phase1_forensics(args.checkpoint_dir, args.out_json)


if __name__ == "__main__":
    main()
