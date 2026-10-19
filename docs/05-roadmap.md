# Roadmap and Milestone Gates

| Milestone | Scope | Key Deliverables | Status |
| :--- | :--- | :--- | :--- |
| **M0** | Baseline Implementation | Pre-norm Transformer, CharacterTokenizer, causal mask invariance tests, tiny-batch memorization | **PASSED** |
| **M1** | Learning & Resumption | Lawful Shakespeare corpus, atomic `CheckpointManager`, RNG/sampler restoration, corruption recovery | **PASSED** |
| **M2** | Colab T4 Qualification | FP16 `autocast` + `GradScaler`, >500k tok/s on Tesla T4, Google Drive backup sync, 20/20 test pass | **PASSED** |
| **M3** | Modular Experts & Continual Tasks | Multi-skill datasets (Text + Procedural Math), `AdaptiveMoELayer`, load-balancing loss, forgetting telemetry | **PASSED** (Verified on Tesla T4) |
| **M4** | Function-Preserving Growth | Zero-init expert/layer expansion CLI (`amg grow`), checkpoint migration, matched-FLOP baseline comparison | **PASSED** (Verified on Tesla T4) |
| **M5** | Dynamic Compute Allocation | Token-level dynamic routing / Mixture-of-Depths (skipping FFN/MoE on easy tokens with true FLOP reduction) | **PASSED** (Verified on Tesla T4) |

---

## Detailed Milestone Verification Criteria

### M0: Baseline Decoder-Only Transformer
- Deterministic initialization and execution given seed.
- Multi-head self-attention with causal upper-triangular masking.
- Invariance test confirming logits at position $t$ are mathematically independent of tokens at $t' > t$.
- Tiny-batch overfit test driving cross-entropy loss $< 0.1$.

### M1: Resumption & Checkpoint Manager
- Structured checkpoint directory containing `latest.pt`, `best.pt`, and `manifest.json`.
- Atomic writes via temporary staging and `os.replace` to protect against sudden termination.
- Exact bitwise loss equivalence between uninterrupted training and interrupted-resumed training.
- Safe rollback to previous valid checkpoint if latest file is truncated or corrupted.

### M2: GPU Acceleration & Google Colab Parity
- Verified execution on NVIDIA Tesla T4 via FP16 mixed precision.
- Training throughput $\ge 400,000$ tokens/second.
- Automatic device selection (`cuda` if available, otherwise `cpu`).
- Remote qualification run log recorded in `docs/04-experiments.md`.

### M3: Modular Residual Experts & Continual Skill Evaluation
- Procedural multi-task data generation (Shakespeare text + procedural arithmetic).
- Sparse MoE routing with top-$k$ selection and load-balancing auxiliary loss.
- Quantification of catastrophic forgetting on prior tasks during sequential adaptation.
- Empirical validation of routing entropy and expert utilization.

### M4: Function-Preserving Structural Growth
- Dynamic addition of residual experts with zero-initialized output projections ($W_{\text{down}} = 0$).
- Verification that logit drift at the boundary step of growth is strictly $\Delta = 0.0000$.
- Selective unseen-symbol adaptation (`trainable_token_ids`) enabling new vocabulary learning without modifying prior token representations.
- Empirical confirmation of zero catastrophic forgetting ($\Delta < 0.0010$) on Task A during Task B fine-tuning.

### M5: Differentiable Causal Mixture-of-Depths (MoD)
- Physical token gathering (`flat_x[selected_indices]`) before FFN/MoE sub-layers for true FLOP and active-parameter reduction.
- Continuous gate multiplication $\sigma(z_{b,t})$ providing direct gradients to `mod_router`.
- Auxiliary BCE loss supervising the gate to predict top-$k$ token assignment during training.
- Token-local thresholding ($z_{b,t} \ge 0.0$) in `eval()` mode ensuring strict autoregressive causality.
- Empirical qualification on Tesla T4: 35% active parameter reduction per token with 38% arithmetic exact-match accuracy.
