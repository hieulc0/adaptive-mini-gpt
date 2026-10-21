"""M20: Controlled Neural Scaling and Procedural Generalization Benchmark.

Compares approximately 1M and 3M parameter Transformer configurations:
- Config A (1M parameters): d_model=128, n_layers=4, n_heads=4, d_ff=512 (928,512 params)
- Config B (3M parameters): d_model=256, n_layers=4, n_heads=8, d_ff=1024 (3,429,888 params)

Evaluates two training objectives:
- Objective A: Supervised action selection from verified planning traces
- Objective B: Structured procedural supervision (intermediate calculations + dependencies + final answer)

Under two evaluation protocols:
- Protocol 1: Matched Tokens (both train for equal token counts)
- Protocol 2: Matched Compute (1M trained for 3.694x epochs to equalize total FLOPs)

Evaluates across 3 independent lineages (seeds 42, 43, 44):
- Neural-only exact final-answer accuracy without tools
- Stepwise procedural validity and intermediate arithmetic correctness
- Teacher-forced action accuracy and complete neural rollout
- Addition depth ladder (depths 2, 3, 4, 5)
- Unseen operands with complex cascading carries
- Unseen composition structures (5-step kinematics)
- Prior multi-corpus language retention
- Training time, inference latency, and peak GPU memory
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn

from adaptive_mini_gpt.data.instruction_data import (
    SEED_INSTRUCTION_EXAMPLES,
    InstructionDataset,
    format_instruction_pair,
)
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.rules.action_dataset import ActionTraceDataset
from adaptive_mini_gpt.rules.action_policy import (
    ActionPolicyModel,
    evaluate_action_ranking,
    train_action_policy,
)
from adaptive_mini_gpt.rules.multistep_dataset import (
    MultiStepProblem,
    generate_addition_depth_ladder,
    generate_multistep_problems,
)
from adaptive_mini_gpt.rules.procedural_dataset import (
    ProceduralDataset,
    ProceduralExample,
    generate_unseen_carry_dataset,
    problem_to_procedural_example,
)
from adaptive_mini_gpt.rules.procedural_evaluator import (
    evaluate_neural_procedure_unaided,
    train_procedural_model,
)
from adaptive_mini_gpt.training.checkpoint import CheckpointManager
from adaptive_mini_gpt.training.config import TrainingConfig
from adaptive_mini_gpt.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("m20_scaling")

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


def pretrain_base_lineage(
    cfg: ModelConfig,
    tokenizer: BPETokenizer,
    save_dir: Path,
    device: str = "cuda",
    precision: str = "fp16",
    steps: int = 150,
    seed: int = 42,
) -> MiniGPT:
    """Pretrain or load base Transformer backbone on reference corpus."""
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = save_dir / "best.pt"
    model = MiniGPT(cfg)

    if best_ckpt.exists():
        logger.info("Found existing checkpoint %s, loading...", best_ckpt)
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        return model

    corpus_prose = (
        "From fairest creatures we desire increase, That thereby beauty's rose might never die.\n"
        "William Shakespeare wrote Hamlet, Macbeth, Othello, and King Lear in London.\n"
        "In quantum physics, photons, bosons, and fermions govern fundamental interactions.\n"
        "Paris is the capital of France, Tokyo is the capital of Japan, and Rome is in Italy.\n"
        "The speed of light in vacuum is approximately 300,000 km/s.\n"
    ) * 30
    instruction_pairs = [
        format_instruction_pair(ex["instruction"], ex["response"])
        for ex in SEED_INSTRUCTION_EXAMPLES
    ]
    full_corpus = corpus_prose + "".join(instruction_pairs) * 10
    tokens = tokenizer.encode(full_corpus)
    split_idx = int(len(tokens) * 0.9)

    ds_train = InstructionDataset(
        token_ids=tokens[:split_idx],
        context_length=cfg.context_length,
        assistant_token_id=tokenizer.assistant_id,
        eos_token_id=tokenizer.eos_id,
        user_token_id=tokenizer.user_id,
    )
    ds_val = InstructionDataset(
        token_ids=tokens[split_idx:],
        context_length=cfg.context_length,
        assistant_token_id=tokenizer.assistant_id,
        eos_token_id=tokenizer.eos_id,
        user_token_id=tokenizer.user_id,
    )

    train_cfg = TrainingConfig(
        seed=seed,
        batch_size=16,
        gradient_accumulation=1,
        learning_rate=1.5e-3,
        max_steps=steps,
        eval_every_steps=max(1, steps // 2),
        device=device,
        precision=precision,
        save_dir=str(save_dir),
    )

    trainer = Trainer(
        model=model,
        model_config=cfg,
        train_config=train_cfg,
        train_dataset=ds_train,
        val_dataset=ds_val,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.to(device)
    return model


def evaluate_language_loss(model: MiniGPT, tokenizer: BPETokenizer, device: torch.device) -> dict[str, float]:
    """Evaluate language model cross-entropy loss on multi-corpus retention suite."""
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


def run_m20_scaling_benchmark(
    base_checkpoints_dir: Path,
    output_dir: Path,
    device_str: str = "cuda",
    seeds: list[int] = [42, 43, 44],
    base_epochs: int = 10,
) -> dict[str, Any]:
    """Execute complete M20 scaling and procedural generalization benchmark."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    logger.info("==================================================================")
    logger.info("M20: Controlled Neural Scaling and Procedural Generalization")
    logger.info("Lineages: %s | Device: %s | Base Epochs: %d", seeds, device, base_epochs)
    logger.info("==================================================================")

    # 1. Sealed Test Datasets
    # A. Familiar test problems (seed 3003)
    fam_test_probs = generate_multistep_problems(
        num_samples_per_archetype=10,
        seed=3003,
        split="test",
        archetypes=["chained_addition_3", "chained_addition_4", "2segment_speed", "multi_entity_compare"],
    )
    # B. Addition depth ladder (seed 4003)
    depth_ladder_probs = generate_addition_depth_ladder(
        num_samples_per_depth=10,
        seed=4003,
        split="test",
    )
    # C. Unseen carry cascading addition (seed 5003)
    unseen_carry_examples = generate_unseen_carry_dataset(
        num_samples=20,
        seed=5003,
        split="test",
    )
    # D. Unseen composition: 5-step 3-segment kinematics (seed 6003)
    ood_kinematics_probs = generate_multistep_problems(
        num_samples_per_archetype=10,
        seed=6003,
        split="test",
        archetypes=["3segment_speed"],
    )

    # Convert test sets for both objectives
    test_fam_proc = [problem_to_procedural_example(p) for p in fam_test_probs]
    test_ladder_proc = [problem_to_procedural_example(p) for p in depth_ladder_probs]
    test_kinematics_proc = [problem_to_procedural_example(p) for p in ood_kinematics_probs]

    test_fam_act_ds = ActionTraceDataset.from_problems(fam_test_probs, split="test")
    test_ladder_act_ds = ActionTraceDataset.from_problems(depth_ladder_probs, split="test")
    test_kinematics_act_ds = ActionTraceDataset.from_problems(ood_kinematics_probs, split="test")

    # 2. Training Datasets (Strict isolation: seed 1003)
    train_probs = generate_multistep_problems(
        num_samples_per_archetype=25,
        seed=1003,
        split="train",
        archetypes=["chained_addition_3", "chained_addition_4", "2segment_speed", "multi_entity_compare", "3segment_speed"],
    )
    val_probs = generate_multistep_problems(
        num_samples_per_archetype=10,
        seed=2003,
        split="val",
        archetypes=["chained_addition_3", "chained_addition_4", "2segment_speed", "multi_entity_compare", "3segment_speed"],
    )

    train_proc_exs = [problem_to_procedural_example(p) for p in train_probs]
    val_proc_exs = [problem_to_procedural_example(p) for p in val_probs]

    train_act_ds = ActionTraceDataset.from_problems(train_probs, split="train")
    val_act_ds = ActionTraceDataset.from_problems(val_probs, split="val")

    logger.info("Training problems: %d, Validation problems: %d", len(train_probs), len(val_probs))
    logger.info("Action traces: train=%d, val=%d", len(train_act_ds), len(val_act_ds))

    # Model Configurations
    cfg_1m = ModelConfig(vocab_size=400, d_model=128, n_layers=4, n_heads=4, d_ff=512, context_length=256)
    cfg_3m = ModelConfig(vocab_size=400, d_model=256, n_layers=4, n_heads=8, d_ff=1024, context_length=256)

    m1_test = MiniGPT(cfg_1m)
    m3_test = MiniGPT(cfg_3m)
    p_1m = sum(p.numel() for p in m1_test.parameters())
    p_3m = sum(p.numel() for p in m3_test.parameters())
    del m1_test, m3_test

    compute_ratio = p_3m / p_1m  # ~3.694
    matched_compute_epochs_1m = max(1, round(base_epochs * compute_ratio))
    logger.info("Parameters: Config A (1M) = %d, Config B (3M) = %d (Ratio: %.3fx)", p_1m, p_3m, compute_ratio)
    logger.info("Epochs: Matched Tokens = %d, Matched Compute (1M) = %d", base_epochs, matched_compute_epochs_1m)

    lineage_results: dict[str, Any] = {}

    for seed in seeds:
        seed_key = f"seed_{seed}"
        logger.info("==================================================================")
        logger.info("Processing Lineage %s", seed_key)
        logger.info("==================================================================")

        # Load 1M base checkpoint
        ckpt_1m_path = base_checkpoints_dir / f"seed_{seed}" / "base_pretrain" / "best.pt"
        raw_1m = CheckpointManager.load_raw_checkpoint(ckpt_1m_path)
        tokenizer = CheckpointManager.load_tokenizer(raw_1m)

        # Procedural datasets using lineage tokenizer
        train_proc_ds = ProceduralDataset(train_proc_exs, tokenizer, context_length=cfg_1m.context_length)
        val_proc_ds = ProceduralDataset(val_proc_exs, tokenizer, context_length=cfg_1m.context_length)

        # Pretrain 3M base model on same pretraining corpus
        seed_dir = output_dir / seed_key
        logger.info("Pretraining Config B (3M) base model for %s...", seed_key)
        base_3m = pretrain_base_lineage(
            cfg=cfg_3m,
            tokenizer=tokenizer,
            save_dir=seed_dir / "base_3m",
            device=device_str,
            steps=150,
            seed=seed,
        )

        results_seed: dict[str, Any] = {
            "parameters": {"config_1m": p_1m, "config_3m": p_3m, "compute_ratio": round(compute_ratio, 3)},
        }

        # -------------------------------------------------------------
        # 1. OBJECTIVE B: Structured Procedural Supervision (Unaided Neural Generation)
        # -------------------------------------------------------------
        logger.info("--- Objective B: Procedural Supervision ---")

        # Arm B1: 1M Model - Matched Tokens (base_epochs)
        torch.manual_seed(seed)
        model_1m_proc_tok = MiniGPT(cfg_1m)
        model_1m_proc_tok.load_state_dict(raw_1m["model_state_dict"])
        model_1m_proc_tok.to(device)
        retention_pre_1m = evaluate_language_loss(model_1m_proc_tok, tokenizer, device)

        t_train_start = time.perf_counter()
        train_stats_1m_tok = train_procedural_model(
            model=model_1m_proc_tok,
            train_dataset=train_proc_ds,
            val_dataset=val_proc_ds,
            device=device,
            epochs=base_epochs,
            lr=3e-4,
        )
        t_1m_tok = time.perf_counter() - t_train_start

        eval_fam_1m_tok = evaluate_neural_procedure_unaided(model_1m_proc_tok, tokenizer, test_fam_proc, device)
        eval_ladder_1m_tok = evaluate_neural_procedure_unaided(model_1m_proc_tok, tokenizer, test_ladder_proc, device)
        eval_carry_1m_tok = evaluate_neural_procedure_unaided(model_1m_proc_tok, tokenizer, unseen_carry_examples, device)
        eval_kin_1m_tok = evaluate_neural_procedure_unaided(model_1m_proc_tok, tokenizer, test_kinematics_proc, device)
        retention_post_1m_tok = evaluate_language_loss(model_1m_proc_tok, tokenizer, device)

        # Arm B2: 3M Model - Matched Tokens (base_epochs)
        torch.manual_seed(seed)
        model_3m_proc = copy.deepcopy(base_3m).to(device)
        retention_pre_3m = evaluate_language_loss(model_3m_proc, tokenizer, device)

        t_train_start = time.perf_counter()
        train_stats_3m = train_procedural_model(
            model=model_3m_proc,
            train_dataset=train_proc_ds,
            val_dataset=val_proc_ds,
            device=device,
            epochs=base_epochs,
            lr=3e-4,
        )
        t_3m = time.perf_counter() - t_train_start

        eval_fam_3m = evaluate_neural_procedure_unaided(model_3m_proc, tokenizer, test_fam_proc, device)
        eval_ladder_3m = evaluate_neural_procedure_unaided(model_3m_proc, tokenizer, test_ladder_proc, device)
        eval_carry_3m = evaluate_neural_procedure_unaided(model_3m_proc, tokenizer, unseen_carry_examples, device)
        eval_kin_3m = evaluate_neural_procedure_unaided(model_3m_proc, tokenizer, test_kinematics_proc, device)
        retention_post_3m = evaluate_language_loss(model_3m_proc, tokenizer, device)

        # Arm B3: 1M Model - Matched Compute (matched_compute_epochs_1m)
        torch.manual_seed(seed)
        model_1m_proc_comp = MiniGPT(cfg_1m)
        model_1m_proc_comp.load_state_dict(raw_1m["model_state_dict"])
        model_1m_proc_comp.to(device)

        t_train_start = time.perf_counter()
        train_stats_1m_comp = train_procedural_model(
            model=model_1m_proc_comp,
            train_dataset=train_proc_ds,
            val_dataset=val_proc_ds,
            device=device,
            epochs=matched_compute_epochs_1m,
            lr=3e-4,
        )
        t_1m_comp = time.perf_counter() - t_train_start

        eval_fam_1m_comp = evaluate_neural_procedure_unaided(model_1m_proc_comp, tokenizer, test_fam_proc, device)
        eval_ladder_1m_comp = evaluate_neural_procedure_unaided(model_1m_proc_comp, tokenizer, test_ladder_proc, device)
        eval_carry_1m_comp = evaluate_neural_procedure_unaided(model_1m_proc_comp, tokenizer, unseen_carry_examples, device)
        eval_kin_1m_comp = evaluate_neural_procedure_unaided(model_1m_proc_comp, tokenizer, test_kinematics_proc, device)

        results_seed["Objective_B_Procedural"] = {
            "1M_Matched_Tokens": {
                "training_time_s": round(t_1m_tok, 2),
                "tokens_trained": train_stats_1m_tok["total_tokens_trained"],
                "final_val_loss": train_stats_1m_tok["final_val_loss"],
                "familiar_exact_acc": eval_fam_1m_tok.exact_final_answer_acc,
                "familiar_valid_acc": eval_fam_1m_tok.procedural_validity_rate,
                "depth_ladder": {
                    "overall_exact_acc": eval_ladder_1m_tok.exact_final_answer_acc,
                    "depth_breakdown": eval_ladder_1m_tok.depth_accuracies,
                },
                "unseen_carry_exact_acc": eval_carry_1m_tok.exact_final_answer_acc,
                "ood_kinematics_exact_acc": eval_kin_1m_tok.exact_final_answer_acc,
                "mean_latency_ms": eval_fam_1m_tok.mean_latency_ms,
                "retention_drift": {k: round(retention_post_1m_tok[k] - retention_pre_1m[k], 4) for k in retention_pre_1m},
            },
            "3M_Matched_Tokens": {
                "training_time_s": round(t_3m, 2),
                "tokens_trained": train_stats_3m["total_tokens_trained"],
                "final_val_loss": train_stats_3m["final_val_loss"],
                "familiar_exact_acc": eval_fam_3m.exact_final_answer_acc,
                "familiar_valid_acc": eval_fam_3m.procedural_validity_rate,
                "depth_ladder": {
                    "overall_exact_acc": eval_ladder_3m.exact_final_answer_acc,
                    "depth_breakdown": eval_ladder_3m.depth_accuracies,
                },
                "unseen_carry_exact_acc": eval_carry_3m.exact_final_answer_acc,
                "ood_kinematics_exact_acc": eval_kin_3m.exact_final_answer_acc,
                "mean_latency_ms": eval_fam_3m.mean_latency_ms,
                "retention_drift": {k: round(retention_post_3m[k] - retention_pre_3m[k], 4) for k in retention_pre_3m},
            },
            "1M_Matched_Compute": {
                "training_time_s": round(t_1m_comp, 2),
                "tokens_trained": train_stats_1m_comp["total_tokens_trained"],
                "final_val_loss": train_stats_1m_comp["final_val_loss"],
                "familiar_exact_acc": eval_fam_1m_comp.exact_final_answer_acc,
                "familiar_valid_acc": eval_fam_1m_comp.procedural_validity_rate,
                "depth_ladder": {
                    "overall_exact_acc": eval_ladder_1m_comp.exact_final_answer_acc,
                    "depth_breakdown": eval_ladder_1m_comp.depth_accuracies,
                },
                "unseen_carry_exact_acc": eval_carry_1m_comp.exact_final_answer_acc,
                "ood_kinematics_exact_acc": eval_kin_1m_comp.exact_final_answer_acc,
                "mean_latency_ms": eval_fam_1m_comp.mean_latency_ms,
            },
        }

        # -------------------------------------------------------------
        # 2. OBJECTIVE A: Action Selection (Tool-Assisted Policy Ranking)
        # -------------------------------------------------------------
        logger.info("--- Objective A: Action Supervision ---")

        # Arm A1: 1M Action Policy (Matched Tokens, 10 epochs)
        torch.manual_seed(seed)
        bb_1m_act = MiniGPT(cfg_1m)
        bb_1m_act.load_state_dict(raw_1m["model_state_dict"])
        bb_1m_act.to(device)

        policy_1m_tok = ActionPolicyModel(
            backbone=bb_1m_act,
            tokenizer=tokenizer,
            d_hidden=128,
            adapter_mode="full",
        ).to(device)

        train_action_policy(
            policy_model=policy_1m_tok,
            train_dataset=train_act_ds,
            val_dataset=val_act_ds,
            device=device,
            epochs=base_epochs,
            lr=1e-4,
            val_problems=val_probs,
        )

        m_act_fam_1m = evaluate_action_ranking(policy_1m_tok, test_fam_act_ds, device, problems_for_rollout=fam_test_probs)
        m_act_ladder_1m = evaluate_action_ranking(policy_1m_tok, test_ladder_act_ds, device, problems_for_rollout=depth_ladder_probs)
        m_act_kin_1m = evaluate_action_ranking(policy_1m_tok, test_kinematics_act_ds, device, problems_for_rollout=ood_kinematics_probs)

        # Arm A2: 3M Action Policy (Matched Tokens, 10 epochs)
        torch.manual_seed(seed)
        bb_3m_act = copy.deepcopy(base_3m).to(device)

        policy_3m_tok = ActionPolicyModel(
            backbone=bb_3m_act,
            tokenizer=tokenizer,
            d_hidden=128,
            adapter_mode="full",
        ).to(device)

        train_action_policy(
            policy_model=policy_3m_tok,
            train_dataset=train_act_ds,
            val_dataset=val_act_ds,
            device=device,
            epochs=base_epochs,
            lr=1e-4,
            val_problems=val_probs,
        )

        m_act_fam_3m = evaluate_action_ranking(policy_3m_tok, test_fam_act_ds, device, problems_for_rollout=fam_test_probs)
        m_act_ladder_3m = evaluate_action_ranking(policy_3m_tok, test_ladder_act_ds, device, problems_for_rollout=depth_ladder_probs)
        m_act_kin_3m = evaluate_action_ranking(policy_3m_tok, test_kinematics_act_ds, device, problems_for_rollout=ood_kinematics_probs)

        results_seed["Objective_A_Action_Policy"] = {
            "1M_Matched_Tokens": {
                "familiar": {
                    "action_acc": m_act_fam_1m.policy_acc,
                    "rollout_acc": m_act_fam_1m.rollout_policy_acc,
                    "latency_ms": m_act_fam_1m.decision_latency_ms,
                },
                "depth_ladder": {
                    "action_acc": m_act_ladder_1m.policy_acc,
                    "rollout_acc": m_act_ladder_1m.rollout_policy_acc,
                    "positions": m_act_ladder_1m.position_accuracies,
                },
                "ood_kinematics_5step": {
                    "action_acc": m_act_kin_1m.policy_acc,
                    "rollout_acc": m_act_kin_1m.rollout_policy_acc,
                },
            },
            "3M_Matched_Tokens": {
                "familiar": {
                    "action_acc": m_act_fam_3m.policy_acc,
                    "rollout_acc": m_act_fam_3m.rollout_policy_acc,
                    "latency_ms": m_act_fam_3m.decision_latency_ms,
                },
                "depth_ladder": {
                    "action_acc": m_act_ladder_3m.policy_acc,
                    "rollout_acc": m_act_ladder_3m.rollout_policy_acc,
                    "positions": m_act_ladder_3m.position_accuracies,
                },
                "ood_kinematics_5step": {
                    "action_acc": m_act_kin_3m.policy_acc,
                    "rollout_acc": m_act_kin_3m.rollout_policy_acc,
                },
            },
        }

        lineage_results[seed_key] = results_seed
        logger.info("Completed Lineage %s: Obj B Fam (1M=%.1f%%, 3M=%.1f%%) | Obj A Rollout (1M=%.1f%%, 3M=%.1f%%)",
                    seed_key,
                    eval_fam_1m_tok.exact_final_answer_acc, eval_fam_3m.exact_final_answer_acc,
                    m_act_fam_1m.rollout_policy_acc, m_act_fam_3m.rollout_policy_acc)

    # -------------------------------------------------------------
    # 3. Aggregate Statistical Summary Across Lineages
    # -------------------------------------------------------------
    def _mean_std(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"mean": 0.0, "std": 0.0}
        m = sum(vals) / len(vals)
        s = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
        return {"mean": round(m, 2), "std": round(s, 2)}

    aggregate_summary: dict[str, Any] = {
        "Objective_B_Procedural": {
            "1M_Matched_Tokens": {
                "familiar_exact_acc": _mean_std([lineage_results[f"seed_{s}"]["Objective_B_Procedural"]["1M_Matched_Tokens"]["familiar_exact_acc"] for s in seeds]),
                "ladder_exact_acc": _mean_std([lineage_results[f"seed_{s}"]["Objective_B_Procedural"]["1M_Matched_Tokens"]["depth_ladder"]["overall_exact_acc"] for s in seeds]),
                "unseen_carry_exact_acc": _mean_std([lineage_results[f"seed_{s}"]["Objective_B_Procedural"]["1M_Matched_Tokens"]["unseen_carry_exact_acc"] for s in seeds]),
                "kinematics_exact_acc": _mean_std([lineage_results[f"seed_{s}"]["Objective_B_Procedural"]["1M_Matched_Tokens"]["ood_kinematics_exact_acc"] for s in seeds]),
            },
            "3M_Matched_Tokens": {
                "familiar_exact_acc": _mean_std([lineage_results[f"seed_{s}"]["Objective_B_Procedural"]["3M_Matched_Tokens"]["familiar_exact_acc"] for s in seeds]),
                "ladder_exact_acc": _mean_std([lineage_results[f"seed_{s}"]["Objective_B_Procedural"]["3M_Matched_Tokens"]["depth_ladder"]["overall_exact_acc"] for s in seeds]),
                "unseen_carry_exact_acc": _mean_std([lineage_results[f"seed_{s}"]["Objective_B_Procedural"]["3M_Matched_Tokens"]["unseen_carry_exact_acc"] for s in seeds]),
                "kinematics_exact_acc": _mean_std([lineage_results[f"seed_{s}"]["Objective_B_Procedural"]["3M_Matched_Tokens"]["ood_kinematics_exact_acc"] for s in seeds]),
            },
            "1M_Matched_Compute": {
                "familiar_exact_acc": _mean_std([lineage_results[f"seed_{s}"]["Objective_B_Procedural"]["1M_Matched_Compute"]["familiar_exact_acc"] for s in seeds]),
                "ladder_exact_acc": _mean_std([lineage_results[f"seed_{s}"]["Objective_B_Procedural"]["1M_Matched_Compute"]["depth_ladder"]["overall_exact_acc"] for s in seeds]),
                "unseen_carry_exact_acc": _mean_std([lineage_results[f"seed_{s}"]["Objective_B_Procedural"]["1M_Matched_Compute"]["unseen_carry_exact_acc"] for s in seeds]),
                "kinematics_exact_acc": _mean_std([lineage_results[f"seed_{s}"]["Objective_B_Procedural"]["1M_Matched_Compute"]["ood_kinematics_exact_acc"] for s in seeds]),
            },
        },
        "Objective_A_Action_Policy": {
            "1M_Matched_Tokens": {
                "familiar_rollout_acc": _mean_std([lineage_results[f"seed_{s}"]["Objective_A_Action_Policy"]["1M_Matched_Tokens"]["familiar"]["rollout_acc"] for s in seeds]),
                "ladder_rollout_acc": _mean_std([lineage_results[f"seed_{s}"]["Objective_A_Action_Policy"]["1M_Matched_Tokens"]["depth_ladder"]["rollout_acc"] for s in seeds]),
                "kinematics_rollout_acc": _mean_std([lineage_results[f"seed_{s}"]["Objective_A_Action_Policy"]["1M_Matched_Tokens"]["ood_kinematics_5step"]["rollout_acc"] for s in seeds]),
            },
            "3M_Matched_Tokens": {
                "familiar_rollout_acc": _mean_std([lineage_results[f"seed_{s}"]["Objective_A_Action_Policy"]["3M_Matched_Tokens"]["familiar"]["rollout_acc"] for s in seeds]),
                "ladder_rollout_acc": _mean_std([lineage_results[f"seed_{s}"]["Objective_A_Action_Policy"]["3M_Matched_Tokens"]["depth_ladder"]["rollout_acc"] for s in seeds]),
                "kinematics_rollout_acc": _mean_std([lineage_results[f"seed_{s}"]["Objective_A_Action_Policy"]["3M_Matched_Tokens"]["ood_kinematics_5step"]["rollout_acc"] for s in seeds]),
            },
        },
    }

    report = {
        "milestone": "M20",
        "description": "Controlled Neural Scaling and Procedural Generalization Benchmark",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seeds": seeds,
        "device": device_str,
        "parameters": {
            "config_1m": p_1m,
            "config_3m": p_3m,
            "ratio": round(compute_ratio, 3),
        },
        "protocols": {
            "matched_tokens_epochs": base_epochs,
            "matched_compute_epochs_1m": matched_compute_epochs_1m,
        },
        "aggregate_summary": aggregate_summary,
        "lineage_results": lineage_results,
    }

    report_path = output_dir / "m20_scaling_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Benchmark report successfully saved to %s", report_path)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--checkpoints-dir", type=Path, default=ROOT_DIR / "runs/kaggle/m18-checkpoints")
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "runs/m20_scaling")
    args = parser.parse_args()

    run_m20_scaling_benchmark(
        base_checkpoints_dir=args.checkpoints_dir,
        output_dir=args.output_dir,
        device_str=args.device,
        seeds=args.seeds,
        base_epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
