# Training, checkpointing, and Colab operation

## Normal development
Develop and test locally on CPU. Use the same `amg train` CLI in an interactive Colab session with a GPU if available. Colab notebook is at most a short, user-started bootstrap cell that clones/checks out a pinned commit, installs `uv`/dependencies, mounts storage, and calls the CLI.

## Data flow
Choose only datasets with verified rights/provenance, deduplicate, split by document/source into train/validation/final holdout, and record dataset fingerprint and tokenizer version. For early testing, a tiny allowed text corpus and deterministic synthetic math data suffice. Do not use GPT/Claude/Gemini outputs for training a competing general-purpose model without authorized terms or express permission.

## Training step
Sample causal sequence `x`, shift target `y`; forward -> cross-entropy -> backward -> optional gradient clip -> optimizer step -> schedule update. Batch/token counting should be explicit. Keep validation separate and do not backpropagate over it.

## Checkpoint contract
Checkpoint must include:
- model architecture, model weights, tokenizer/vocab ID and map
- optimizer and scheduler state (if used), AMP scaler (if used)
- current update step, tokens processed, dataset/sampler position, seed
- Python, NumPy, torch CPU/CUDA RNG states as relevant
- dataset/config fingerprint, code revision, best validation score and run ID

Write checkpoint to local temporary path, close and verify, then move atomically on the same filesystem. To copy to Google Drive, use immutable versioned checkpoint names, verify the copy, and only then update a small `latest.json` pointer. Preserve prior good checkpoints. Copying into Drive is not assumed atomic; filesystem/network interruption must not corrupt the only copy.

Restore after reconstructing the exact architecture and optimizer. Reject incompatible configs, vocabulary, or dataset identity. Never load unknown `torch.load` pickle checkpoints; `weights_only=False` is unsuitable for untrusted files.

## Sessions
- Colab free runtimes can terminate unpredictably. Do not assume a 5-hour runtime or fixed cooldown.
- Use periodic checkpointing initially every 5–10 minutes by elapsed wall time; measure serialization overhead and adjust.
- Start training locally on the Colab filesystem; publish checkpoints asynchronously only if verified for correctness, without risking loss on termination.
- After a new interactive session starts, restore latest **validated** checkpoint and continue, not restart.
- Avoid idle-bypass, account rotation, auto-reconnection, remote worker management or any mechanism intended to evade free Colab restrictions.

## Acceptance tests
- CPU checkpoint save/load and same-data resume across process restart.
- Continuous vs interrupted trajectory matches within documented deterministic tolerance.
- Simulated truncated checkpoint is ignored and last known-good loads.
- Fresh-runtime resume smoke performed in Colab before claiming GPU readiness.
