# Adaptive Mini-GPT

An experimental, **CLI-first Python/PyTorch** research framework studying **adaptive capacity allocation** and **continual modular skill acquisition** in small decoder-only Transformers under constrained single-GPU budgets.

**Current status:**
- **M0 (Baseline Architecture & Determinism):** Passed
- **M1 (Stateful Training, Atomic Checkpointing & Resumption):** Passed
- **M2 (Colab Tesla T4 FP16 Qualification — >500k tok/s):** Passed
- **M3 (Modular Residual Experts & Continual Learning):** Passed (Verified on Tesla T4)
- **M4 (Function-Preserving Growth & Zero Forgetting):** Passed (Verified on Tesla T4)
- **M5 (Mixture-of-Depths & Dynamic Token Compute Allocation):** Passed (Verified on Tesla T4)

## Research Hypothesis

Under a matched compute and token budget, **function-preserving modular growth** (attaching zero-initialized bottleneck experts to a trained backbone with load-balanced or task-conditioned routing) improves continual skill retention and sample efficiency compared to static dense retraining, while mitigating catastrophic forgetting.

## Core Engineering Principles

- **Learn first, adapt later:** Verify dense baselines and deterministic resumption before activating routing or capacity growth.
- **Function-preserving transitions:** Any structural growth operation ($M \to M'$) must satisfy $M'(x) = M(x)$ at the step of expansion via zero-initialized residual projections.
- **Single-process CLI parity:** Identical `amg` CLI execution on local Arch Linux CPU and interactive Google Colab T4 GPUs.
- **Crash-only atomic persistence:** Checkpoints write to temporary files, pass a load verification smoke test, and atomically swap via `os.replace` alongside a versioned `manifest.json`.
- **Reproducible evaluation:** Deterministic document/task splits, SHA-256 data and vocabulary fingerprints, and isolated holdout verification.

## Quickstart & CLI Usage

```bash
# Install environment and dev dependencies
uv sync --extra dev

# Run test suite and linter
uv run ruff check src tests scripts
uv run pytest -q

# Train dense baseline or adaptive expert model
uv run amg train --config configs/tiny.yaml

# Resume interrupted run (with automatic fallback if latest checkpoint is truncated)
uv run amg train --config configs/tiny.yaml --resume runs/exp-001/latest.pt

# Grow a checkpoint by spawning a new residual expert for continual learning
uv run amg grow --checkpoint runs/exp-001/best.pt --output runs/exp-002/init.pt --add-experts 2 --freeze-backbone

# Evaluate perplexity, bits-per-character, and expert utilization
uv run amg evaluate --checkpoint runs/exp-001/best.pt

# Autoregressive generation
uv run amg generate --checkpoint runs/exp-001/best.pt --prompt "From fairest "
```

## Headless Google Colab GPU Workflow

To run tests, training, and multi-arm continual learning benchmarks directly on an authenticated Google Colab Tesla T4 GPU from your terminal:

```bash
# Check auth & allocate T4 session
uvx --from google-colab-cli colab new -s amg-t4 --gpu T4

# Package, upload, install & run test suite on Colab
tar --exclude='.git' --exclude='runs' --exclude='.venv' --exclude='__pycache__' -czf /tmp/repo.tar.gz .
uvx --from google-colab-cli colab upload -s amg-t4 /tmp/repo.tar.gz /content/repo.tar.gz
uvx --from google-colab-cli colab exec -s amg-t4 -f <(echo "
mkdir -p /content/adaptive-mini-gpt
tar -xzf /content/repo.tar.gz -C /content/adaptive-mini-gpt
pip install -e /content/adaptive-mini-gpt pytest
cd /content/adaptive-mini-gpt
pytest -q
")

# Release GPU session when done
uvx --from google-colab-cli colab stop -s amg-t4
```

See [docs/03-training-and-colab.md](docs/03-training-and-colab.md) for the full headless terminal guide and step-by-step training commands.

## Milestone Validation Summary

| Milestone | Key Deliverable | Verified Result on Tesla T4 |
| :--- | :--- | :--- |
| **M0/M1** | Deterministic Pre-norm Transformer & Atomic Checkpointing | Bitwise identical loss across resumes; safe recovery from corrupted checkpoints |
| **M2** | FP16 Mixed Precision & High-Throughput Training | Peak **513,012 tok/s**, sustained ~**420,000 tok/s** |
| **M3** | Modular Experts & Balanced Routing | High router entropy (**0.9742**), balanced expert load distribution |
| **M4** | Function-Preserving Structural Growth & Zero Forgetting | Exact $\Delta = 0.0000$ boundary loss drift; **0.0009** backbone forgetting |
| **M5** | Differentiable Mixture-of-Depths (MoD) | **35% active parameter reduction** per token, **38% arithmetic exact match** |

## Directory Structure

```
├── configs/                  # Reproducible training and architecture configs
│   └── tiny.yaml
├── data/raw/                 # Raw datasets (e.g., Shakespeare Sonnets)
├── docs/                     # Research discipline and design documentation
│   ├── 01-vision.md
│   ├── 02-architecture.md
│   ├── 03-training-and-colab.md
│   ├── 04-experiments.md
│   └── 05-roadmap.md
├── scripts/                  # Standalone benchmark experiments & runners
│   └── run_m3_m4_experiment.py
├── src/adaptive_mini_gpt/    # Core package
│   ├── adaptive/             # MoE layers, routers, growth operators, MoD
│   ├── data/                 # Tokenizer, procedural datasets, dataloader
│   ├── evaluation/           # Perplexity, BPC, arithmetic accuracy evaluators
│   ├── model/                # Transformer decoder, attention, configuration
│   └── training/             # Trainer, checkpoint manager, optimizer
└── tests/                    # 40+ deterministic unit and integration tests
```
