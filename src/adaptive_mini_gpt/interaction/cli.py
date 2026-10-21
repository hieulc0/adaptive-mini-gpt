"""Minimal Interactive CLI for Mini-GPT (M29 Phase 1).

Allows asking queries, inspecting neural generations, viewing confidence signals,
auditing verifiers, checking knowledge-gap logs, and viewing checkpoint history.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from adaptive_mini_gpt.data.bpe_scaling import train_scaled_bpe_tokenizer
from adaptive_mini_gpt.data.language_corpus import PRETRAIN_PROSE_CORPUS
from adaptive_mini_gpt.data.subword_tokenizer import BPETokenizer
from adaptive_mini_gpt.evaluation.multi_domain import DomainType
from adaptive_mini_gpt.interaction.action_selector import InteractionActionSelector
from adaptive_mini_gpt.interaction.knowledge_gap import KnowledgeGapTracker
from adaptive_mini_gpt.interaction.loop import ClosedLoopInteractiveLearner
from adaptive_mini_gpt.interaction.verified_memory import VerifiedMemoryBank
from adaptive_mini_gpt.model.transformer import MiniGPT, ModelConfig
from adaptive_mini_gpt.training.checkpoint import CheckpointManager


def run_interactive_cli(
    model_path: Path | None = None,
    tokenizer_path: Path | None = None,
    device_str: str = "cpu",
) -> None:
    """Run an interactive terminal session with ClosedLoopInteractiveLearner."""
    device = torch.device(device_str)
    print("=" * 60)
    print("Adaptive Mini-GPT: M29 Closed-Loop Interactive Learner")
    print("=" * 60)

    # 1. Load model and tokenizer
    if model_path and model_path.exists():
        print(f"Loading checkpoint from: {model_path}")
        raw = CheckpointManager.load_raw_checkpoint(model_path)
        cfg = ModelConfig(**raw["model_config"])
        model = MiniGPT(cfg).to(device)
        sd = raw.get("model_state_dict", raw.get("model_state"))
        if sd:
            model.load_state_dict(sd)
        tokenizer = CheckpointManager.load_tokenizer(raw)
        ckpt_name = model_path.stem
    else:
        print("Initializing lightweight in-memory MiniGPT for demonstration...")
        all_prose = " ".join(PRETRAIN_PROSE_CORPUS[:15])
        tokenizer = train_scaled_bpe_tokenizer(all_prose, target_vocab_size=1024)
        cfg = ModelConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=128,
            d_model=128,
            num_layers=4,
            num_heads=4,
            d_ff=512,
        )
        model = MiniGPT(cfg).to(device)
        ckpt_name = "demo_in_memory"

    learner = ClosedLoopInteractiveLearner(
        model=model,
        tokenizer=tokenizer,
        device=device,
        checkpoint_name=ckpt_name,
    )

    print(f"Active Checkpoint: {ckpt_name}")
    print(f"Vocab Size: {tokenizer.vocab_size}")
    print(f"Device: {device}")
    print("Commands: 'quit', 'history', 'gaps', 'memory'")
    print("-" * 60)

    while True:
        try:
            line = input("Mini-GPT> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting interactive session.")
            break

        if not line:
            continue
        if line.lower() in ("quit", "exit"):
            print("Session ended.")
            break
        elif line.lower() == "history":
            print(f"Total Interaction Cycles: {learner.cycle_count}")
            diag = learner.gap_tracker.compute_diagnostics()
            print(f"Diagnostics: {diag}")
            continue
        elif line.lower() == "gaps":
            gaps = list(learner.gap_tracker.records.values())
            print(f"Detected Gaps ({len(gaps)}):")
            for g in gaps:
                print(f"  [{g.gap_id}] Domain={g.domain.value} Cat={g.error_category.value} Occur={g.occurrence_count}: {g.question}")
            continue
        elif line.lower() == "memory":
            entries = learner.memory_bank.get_all()
            print(f"Verified Memory Bank ({len(entries)}):")
            for e in entries:
                print(f"  [{e.entry_id}] Domain={e.domain.value} Source={e.evidence_source}: {e.topic_or_key} -> {e.verified_fact}")
            continue

        # Process user query (defaulting to Language domain)
        res = learner.interact(
            domain=DomainType.LANGUAGE,
            question=line,
            context=None,
            ground_truth=None,
        )

        print(f"Predicted Answer : {res.predicted_answer}")
        print(f"Confidence Score : {res.confidence_score:.4f}")
        print(f"Selected Action  : {res.decision.action.value.upper()}")
        print(f"Action Rationale : {res.decision.rationale}")
        print(f"Final Response   : {res.final_output}")
        print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini-GPT Interactive CLI")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    model_p = Path(args.model_path) if args.model_path else None
    tok_p = Path(args.tokenizer_path) if args.tokenizer_path else None
    run_interactive_cli(model_path=model_p, tokenizer_path=tok_p, device_str=args.device)


if __name__ == "__main__":
    main()
