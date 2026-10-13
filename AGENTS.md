# Agent instructions — Adaptive Mini-GPT

Read all `docs/*.md` and `configs/tiny.yaml` before changing source. Treat documented milestones and acceptance gates as binding. Implement **M0 then M1 only** unless explicitly instructed otherwise.

## Scope
- Python 3.11+; PyTorch; `uv`; Pytest; Ruff. Use a `src/` package and `amg` CLI. Prefer standard library `argparse` over extra CLI dependencies.
- No notebooks for model/training logic. Colab is an interactive runner of the same CLI.
- M0: tiny decoder-only Transformer with causal masking, tokenization, cross-entropy loss, model inspection; tests for shapes, causal independence, gradients, parameter counts, deterministic tiny-batch overfit.
- M1: deterministic dataset split and streaming/batched training, validation, text generation, metrics JSONL, safe checkpoint publication, interruption/restart, full training state restoration, and tests.
- Run CPU tests in CI-like local environment. CUDA path conditional; do not claim T4 validation unless actually run.

## Non-goals
- No experts, router, adaptive growth, RAG, web crawler, automatic Colab reconnect, multi-account rotation, database, Kubernetes, distributed workers, Rust rewrite, or proprietary teacher data in M0/M1.

## Delivery discipline
1. Implement the smallest coherent increment; run Ruff and Pytest.
2. Track experiment config and dataset fingerprint in checkpoints. Validate compatibility before restoring.
3. Never overwrite the sole good checkpoint. Publish complete checkpoints atomically on the local filesystem; for Drive, upload a uniquely named artifact, verify, then update a manifest. Keep last known-good.
4. Persist model, optimizer, scheduler if used, global step, RNG states, sampler/corpus position, vocabulary mapping, architecture config, and precision scaler if used. No `torch.load` from untrusted sources.
5. Log train/validation loss, steps, tokens, wall-clock, peak GPU memory (when CUDA exists), throughput, and checkpoint path. Avoid invented benchmark numbers.
6. For resumption test: compare continuous and interrupted training on CPU for same steps, deterministic data ordering, and numerical closeness. Document limitations if exact equality cannot be achieved.
7. Report what actually passed, what is blocked, and instructions to reproduce. Commit cohesive milestones if repository permissions allow.

## Design guardrails
- Prefer explicit, readable PyTorch code rather than clever abstractions.
- Architecture and tokenizer must be checkpoint-compatible; reject mismatches loudly.
- Model layer code must not import Google Colab, GCP, or Drive APIs.
- Keep all outputs/data/weights outside Git except tiny deterministic test fixtures.
- Freeze held-out evaluation before experiments. Do not tune against the final holdout.
