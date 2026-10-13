# Roadmap and milestone gates

## M0 — Correct baseline implementation (first)
Deliver: repository with `uv`, `ruff`, `pyright`, `pytest`, package + CLI; decoder-only Transformer; tokenizer; deterministic dummy batch; CPU tests for shapes, causal masking (future tokens cannot change earlier logits in eval mode), finite grads, and tiny-batch memorization. No GPU requirement. Gate: all tests green and demonstrable loss decline.

## M1 — Actual learning + reliable resume (second)
Deliver: lawful tiny text data path, stable deterministic split, train/validate/generate commands, metrics.jsonl, checkpoint manifest and full state, interrupt/resume tests, documented run instructions. Gate: fresh-process restore continues from correct step, latest corrupted checkpoint does not destroy last valid, validation recorded, reproducible baseline.

## M2 — T4 execution
Deliver: optional short Colab launcher, dependency pinning, GPU correctness smoke, AMP stability, memory/throughput measurements, Drive checkpoint publishing, tested fresh-session resume. Gate: evidence from an actual Colab run. No claims until exercised.

## M3 — Modular experts
Deliver: two task datasets with verifiers; frozen/shared backbone experiment; task-conditioned expert and simple router; before/after forgetting metrics; fixed dense comparison. Gate: nonregressing or quantified trade-offs.

## M4 — Adaptive growth
Deliver: growth proposal, compatible initialization, paired unchanged-control run, checkpoint version transition and rollback. Gate: significant/repeatable cost-adjusted gain or rejected candidate with measurements.

## M5 — Extended research (optional)
Learned routing, expert sparsity, adaptive width/depth, retrieval (distinct from weight training), multi-run controller. Only after baseline evidence.

### Stop conditions
Stop architecture expansion if it cannot beat simpler baselines, if validation worsens persistently, or if compute cost materially exceeds the available budget. Never infer 'self improvement' from training loss alone.
