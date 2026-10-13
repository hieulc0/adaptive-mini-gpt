"""Command-line interface for Adaptive Mini-GPT (amg)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from adaptive_mini_gpt.data.dataset import TokenDataset, load_corpus, split_data
from adaptive_mini_gpt.data.tokenizer import CharacterTokenizer
from adaptive_mini_gpt.evaluation.evaluator import evaluate_dataset, generate_sample
from adaptive_mini_gpt.model.config import ModelConfig
from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.training.checkpoint import CheckpointManager
from adaptive_mini_gpt.training.config import TrainingConfig
from adaptive_mini_gpt.training.trainer import Trainer


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def train_command(args: argparse.Namespace) -> None:
    config_data = {}
    if args.config and Path(args.config).is_file():
        config_data = load_yaml_config(args.config)

    # 1. Prepare raw text data and tokenizer
    text, data_fp = load_corpus(args.data)
    tokenizer = CharacterTokenizer()

    # If resuming, we might want to check the checkpoint for tokenizer
    resume_checkpoint_data = None
    if args.resume:
        resume_checkpoint_data = CheckpointManager.load_raw_checkpoint(args.resume)
        if "tokenizer" in resume_checkpoint_data and resume_checkpoint_data["tokenizer"].get(
            "vocab"
        ):
            tokenizer = CharacterTokenizer.from_dict(resume_checkpoint_data["tokenizer"])
        else:
            tokenizer.fit(text)
    else:
        tokenizer.fit(text)

    # 2. Build ModelConfig
    model_raw = config_data.get("model", {})
    model_raw["vocab_size"] = tokenizer.vocab_size
    model_config = ModelConfig.from_dict(model_raw)

    # 3. Build TrainingConfig
    train_raw = config_data.get("training", {})
    if args.save_dir:
        train_raw["save_dir"] = args.save_dir
    if args.backup_dir:
        train_raw["backup_dir"] = args.backup_dir
    if args.precision:
        train_raw["precision"] = args.precision
    if args.max_steps is not None:
        train_raw["max_steps"] = args.max_steps
    if args.device:
        train_raw["device"] = args.device
    if args.seed is not None:
        train_raw["seed"] = args.seed
    training_config = TrainingConfig.from_dict(train_raw)

    # 4. Tokenize and split corpus
    all_tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    train_tokens, val_tokens, _ = split_data(all_tokens, train_ratio=0.8, val_ratio=0.2)

    # Ensure dataset is large enough for context length
    if len(train_tokens) <= model_config.context_length:
        raise ValueError(
            f"Train tokens ({len(train_tokens)}) must be greater than context_length "
            f"({model_config.context_length}). Provide a larger corpus or reduce context_length."
        )
    if len(val_tokens) <= model_config.context_length:
        # Fallback to train tokens for validation if dataset is very small
        val_tokens = train_tokens

    train_dataset = TokenDataset(train_tokens, model_config.context_length)
    val_dataset = TokenDataset(val_tokens, model_config.context_length)

    # 5. Initialize Model & Trainer
    model = MiniGPT(model_config)
    trainer = Trainer(
        model=model,
        model_config=model_config,
        training_config=training_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        dataset_fingerprint=data_fp,
    )

    # 6. Resume if requested
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.resume_from_checkpoint(args.resume, strict_dataset_match=not args.force_resume)

    # 7. Start Training
    result = trainer.train()
    print(f"Training completed successfully. Checkpoint saved to: {result['save_dir']}")


def evaluate_command(args: argparse.Namespace) -> None:
    checkpoint_data = CheckpointManager.load_raw_checkpoint(args.checkpoint)
    model_config = ModelConfig.from_dict(checkpoint_data["model_config"])
    tokenizer = CharacterTokenizer.from_dict(checkpoint_data["tokenizer"])

    model = MiniGPT(model_config)
    model.load_state_dict(checkpoint_data["model_state_dict"])

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)

    text, _ = load_corpus(args.data)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    _, val_tokens, _ = split_data(tokens, train_ratio=0.8, val_ratio=0.2)
    if len(val_tokens) <= model_config.context_length:
        val_tokens = tokens

    val_dataset = TokenDataset(val_tokens, model_config.context_length)
    metrics = evaluate_dataset(
        model=model,
        dataset=val_dataset,
        batch_size=args.batch_size,
        device=device,
        max_batches=args.eval_steps,
    )

    print("\n--- Evaluation Results ---")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.2f}")
    print(f"Bits per character: {metrics['bits_per_character']:.4f}")
    print(f"Batches evaluated: {metrics['batches_evaluated']}")


def generate_command(args: argparse.Namespace) -> None:
    checkpoint_data = CheckpointManager.load_raw_checkpoint(args.checkpoint)
    model_config = ModelConfig.from_dict(checkpoint_data["model_config"])
    tokenizer = CharacterTokenizer.from_dict(checkpoint_data["tokenizer"])

    model = MiniGPT(model_config)
    model.load_state_dict(checkpoint_data["model_state_dict"])

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)

    res = generate_sample(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )

    print("\n--- Generated Output ---")
    print(res["generated_text"])


def inspect_command(args: argparse.Namespace) -> None:
    if args.checkpoint:
        ckpt = CheckpointManager.load_raw_checkpoint(args.checkpoint)
        model_config = ModelConfig.from_dict(ckpt["model_config"])
        model = MiniGPT(model_config)
        info = model.inspect_architecture()

        print("\n=== Checkpoint Inspection ===")
        print(f"Run ID: {ckpt.get('run_id')}")
        print(f"Step: {ckpt.get('step')}")
        print(f"Tokens seen: {ckpt.get('tokens_seen')}")
        print(f"Best Val Loss: {ckpt.get('best_val_loss')}")
        print(f"Dataset Fingerprint: {ckpt.get('dataset_fingerprint')}")
        print(f"Tokenizer Vocab Size: {len(ckpt.get('tokenizer', {}).get('vocab', []))}")
        print("\n=== Model Architecture ===")
        for k, v in info.items():
            if k != "layers":
                print(f"  {k}: {v}")
        print("\n=== Layers ===")
        for layer in info["layers"]:
            print(f"  Layer {layer['layer']}: {layer['params']:,} parameters")
    elif args.config:
        cfg_data = load_yaml_config(args.config)
        model_cfg = ModelConfig.from_dict(cfg_data.get("model", {}))
        model = MiniGPT(model_cfg)
        info = model.inspect_architecture()
        print("\n=== Configuration Inspection ===")
        print(json.dumps(cfg_data, indent=2))
        print("\n=== Model Architecture ===")
        for k, v in info.items():
            if k != "layers":
                print(f"  {k}: {v}")
        print("\n=== Layers ===")
        for layer in info["layers"]:
            print(f"  Layer {layer['layer']}: {layer['params']:,} parameters")
    else:
        print("Please specify either --checkpoint or --config to inspect.")
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amg",
        description="Adaptive Mini-GPT: decoder-only transformer research platform.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # amg train
    train_parser = subparsers.add_parser("train", help="Train Mini-GPT model.")
    train_parser.add_argument(
        "--config", type=str, default="configs/tiny.yaml", help="Path to config yaml"
    )
    train_parser.add_argument("--data", type=str, default=None, help="Path to text data file")
    train_parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume"
    )
    train_parser.add_argument(
        "--force-resume",
        action="store_true",
        help="Force resume even if dataset fingerprint differs",
    )
    train_parser.add_argument(
        "--save-dir", type=str, default=None, help="Directory to save run outputs"
    )
    train_parser.add_argument(
        "--backup-dir",
        type=str,
        default=None,
        help="External backup directory (e.g. Google Drive)",
    )
    train_parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["fp32", "fp16", "bf16"],
        help="Precision format",
    )
    train_parser.add_argument(
        "--max-steps", type=int, default=None, help="Override maximum training steps"
    )
    train_parser.add_argument(
        "--device", type=str, default=None, help="Compute device (cpu, cuda, auto)"
    )
    train_parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # amg evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate checkpoint on validation data.")
    eval_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    eval_parser.add_argument(
        "--data", type=str, default=None, help="Path to evaluation text data"
    )
    eval_parser.add_argument(
        "--eval-steps", type=int, default=20, help="Number of batches to evaluate"
    )
    eval_parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for evaluation"
    )
    eval_parser.add_argument("--device", type=str, default="auto", help="Compute device")

    # amg generate
    gen_parser = subparsers.add_parser("generate", help="Generate text from a saved checkpoint.")
    gen_parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    gen_parser.add_argument("--prompt", type=str, default="The ", help="Conditioning text prompt")
    gen_parser.add_argument(
        "--max-tokens", type=int, default=100, help="Number of tokens to generate"
    )
    gen_parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    gen_parser.add_argument("--top-k", type=int, default=40, help="Top-k filtering")
    gen_parser.add_argument("--device", type=str, default="auto", help="Compute device")

    # amg inspect
    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect model architecture or checkpoint."
    )
    inspect_parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint to inspect"
    )
    inspect_parser.add_argument(
        "--config", type=str, default=None, help="Path to config yaml to inspect"
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "generate":
        generate_command(args)
    elif args.command == "inspect":
        inspect_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
