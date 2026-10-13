# Architecture and boundaries

## v0.1: fixed decoder-only Transformer

```
text -> fixed vocabulary tokenizer -> token ids -> embeddings + positions
     -> [causal attention -> residual -> FFN -> residual] × L
     -> final norm -> tied/untied LM head -> logits -> cross-entropy
```

Implement pre-norm blocks, causal self-attention, scaled dot-product attention (PyTorch where practical), dropout and configurable FFN. Ensure future positions are masked; inference respects context limit.

### Baseline configuration
- L=2, d_model=128, n_heads=4, d_ff=512, context=256, dropout=0.1.
- Start with a fixed character vocabulary for simple end-to-end debugging, persisted with each run. Padding/special-token policy must be explicit; do not silently change vocabulary across runs.
- AdamW, gradient clipping, seed, learning-rate schedule optional. Use FP32 CPU first; use CUDA AMP only after tested.

## Later: modular learning (M3+ only)

```
shared backbone h -> gate/router -> selected Expert_i(h) -> weighted residual merge
                                              -> LM output head
```

Experts can be small bottleneck MLPs sharing input/output dimension `d_model`; independently versioned weights do **not** imply they are interoperable. Route per-token or per-sequence as an explicit experiment. Start with task-directed routing before learned routing. Track selection entropy, utilization distribution, expert gradients and per-task validation.

One candidate mathematical form:

`h' = h + sum(i in topk(router(h))) p_i(h) E_i(h)`

A frozen backbone can stabilize expert training but may be too weak if frozen prematurely. Shared/router changes can still cause forgetting.

## Future growth operators (M4+ only)
- Add a residual expert (simplest).
- Expand an expert bottleneck (requires initialization that preserves useful previous behavior).
- Add a zero-initialized residual Transformer block.
- Increase context with compatible positional strategy.
- Widen shared backbone (complex migration of connected tensor shapes and optimizer state).

Growth is a **checkpoint-boundary experiment** with a candidate model and a no-growth control, not an unverified online tensor mutation.

## Package layout
`model/` pure tensor modules; `data/` corpora/tokenization; `training/` trainer/checkpoint; `evaluation/` held-out tests; `adaptive/` reserved future experiment controller; `cli.py` frontend. No service, database, or cloud dependency inside `model/`.
