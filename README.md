# Adaptive Mini-GPT

An experimental, **CLI-first Python/PyTorch** project to study whether a small decoder-only Transformer can acquire skills incrementally and gain useful capacity through validated modular growth, under a constrained GPU budget.

**Current status:** design and documentation scaffold only. This archive intentionally does not claim working training code. An implementation agent should deliver M0 and M1 first.

## Research hypothesis

Under a matched compute budget, targeted growth and selective reuse of trained modules may improve retained capability relative to a fixed dense baseline. This is a hypothesis to test, not an assumed fact.

## Core principles

- **Learn first, adapt later:** prove a fixed GPT baseline trains and resumes before adding experts or growth.
- **One Python process:** no microservices, GCP scheduler, API server, or orchestration service initially.
- **Local/Colab parity:** identical CLI on local CPU and supported Colab GPU; notebooks are optional launchers, not source code.
- **Durable experiments:** version configs, data provenance, metrics, complete checkpoints, and evaluations.
- **No unauthorized distillation:** do not collect outputs from proprietary assistants for competing-model training without explicit permission.
- **Reproducible comparisons:** evaluate held-out tasks and equivalent budgets; log uncertainty and failures.

## Proposed CLI (NOT IMPLEMENTED)

```bash
uv sync --extra dev
uv run amg train --config configs/tiny.yaml
uv run amg train --config configs/tiny.yaml --resume runs/exp-001/latest.pt
uv run amg evaluate --checkpoint runs/exp-001/best.pt
uv run amg generate --checkpoint runs/exp-001/best.pt --prompt 'The drone'
```

Read [docs/01-vision.md](docs/01-vision.md), [docs/02-architecture.md](docs/02-architecture.md), [docs/03-training-and-colab.md](docs/03-training-and-colab.md), [docs/04-experiments.md](docs/04-experiments.md), [docs/05-roadmap.md](docs/05-roadmap.md), and [AGENTS.md](AGENTS.md).
