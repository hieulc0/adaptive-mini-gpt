# Adaptive Mini-GPT

An experimental, **CLI-first Python/PyTorch** research framework studying **adaptive capacity allocation** and **continual modular skill acquisition** in small decoder-only Transformers under constrained single-GPU budgets.

**Current status:**
- **M0 (Baseline Architecture & Determinism):** Passed
- **M1 (Stateful Training, Atomic Checkpointing & Resumption):** Passed
- **M2 (Colab Tesla T4 FP16 Qualification — >500k tok/s):** Passed
- **M3 & M4 (Modular Residual Experts & Function-Preserving Growth):** Active Implementation

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
uv run ruff check src tests
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

## Documentation Index

1. [01-vision.md](docs/01-vision.md) — Problem formulation, taxonomy of capacity, and research hypotheses.
2. [02-architecture.md](docs/02-architecture.md) — Pre-norm SDPA Transformer, bottleneck residual experts, load-balanced routing, and zero-init growth math.
3. [03-training-and-colab.md](docs/03-training-and-colab.md) — Data provenance, parameter-group optimization, atomic checkpoint contract, and Colab workflow.
4. [04-experiments.md](docs/04-experiments.md) — T4 qualification telemetry, continual learning benchmarks, and ablation protocols.
5. [05-roadmap.md](docs/05-roadmap.md) — Milestone gates (M0–M5) and acceptance criteria.
