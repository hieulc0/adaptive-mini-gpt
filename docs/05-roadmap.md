# Roadmap and Milestone Gates

| Milestone | Scope | Key Deliverables | Status |
| :--- | :--- | :--- | :--- |
| **M0** | Baseline Implementation | Pre-norm Transformer, CharacterTokenizer, causal mask invariance tests, tiny-batch memorization | **PASSED** |
| **M1** | Learning & Resumption | Lawful Shakespeare corpus, atomic `CheckpointManager`, RNG/sampler restoration, corruption recovery | **PASSED** |
| **M2** | Colab T4 Qualification | FP16 `autocast` + `GradScaler`, >500k tok/s on Tesla T4, Google Drive backup sync, 20/20 test pass | **PASSED** |
| **M3** | Modular Experts & Continual Tasks | Multi-skill datasets (Text + Procedural Math), `AdaptiveMoELayer`, load-balancing loss, forgetting telemetry | **PASSED** (Verified on Tesla T4) |
| **M4** | Function-Preserving Growth | Zero-init expert/layer expansion CLI (`amg grow`), checkpoint migration, matched-FLOP baseline comparison | **PASSED** (Verified on Tesla T4) |
| **M5** | Dynamic Compute Allocation | Token-level dynamic routing / Mixture-of-Depths (skipping FFN/MoE on easy tokens with true FLOP reduction) | **PASSED** (Verified on Tesla T4) |
| **M6** | Autonomous Causal Domain Gating | Autoregressive domain detection, zero-forgetting unconditioned Task A retention, `moe_head` adapter | **PASSED** (Verified on Tesla T4) |
| **M7** | Generalization & Multi-Seed Verification | 4-arm 3-seed benchmark, numerical prose decoupling (`domain_trigger_token_ids=['=']`), carry/borrow evaluation | **PASSED** (Verified on Tesla T4) |
| **M8** | Autonomous Curriculum & Growth Controller | State machine (`EVALUATE -> SELECT -> PLAN -> TRAIN -> VERIFY -> PROMOTE/ROLLBACK`), bounded compute budgets | **PASSED** |
| **M9** | Natural Language & Practical Reasoning | BPE subword tokenizer, instruction dataset, scaled Transformer backbone, verified reasoning tasks | **PASSED** |

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

### M6: Autonomous Causal Domain Gating & `moe_head` Residual Adapter
- Autoregressive causal domain prefix tracking eliminating manual task flags.
- Low-Rank Output Adapter (`moe_head`) solving the frozen `lm_head` expressivity bottleneck while maintaining strict `0.000000` drift on pure Task A inputs.
- Preservation of zero-init guarantees in `MiniGPT.__init__`.

### M7: Rigorous Generalization, Retention & Architecture Verification
- Comprehensive multi-domain evaluation: clean prose, numerical prose (dates, quantities), and mixed prose/equations.
- Disentangled domain gating: separation of representation adaptation (`trainable_token_ids`) from expert activation triggers (`domain_trigger_token_ids = ['=']`), completely eliminating numerical prose forgetting ($\Delta = 0.000000 \pm 0.0$).
- 4-arm architecture comparison across 3 seeds (`[42, 43, 44]`): Dense FT, Modular MoE, MoE + MoD, and Compute-Matched Dense Adapter Control.
- Fine-grained category evaluation (`ADD_NO_CARRY`, `ADD_CARRY`, `SUB_NO_BORROW`, `SUB_BORROW`, `PARITY`, `OOD_EXTRAPOLATION`).
- Empirical finding: MoD + MoE (Arm 3) delivers highest modular accuracy with 33% fewer active parameters, outperforming matched adapter baselines. Character-level carry addition/borrow subtraction requires algorithmic curricula or subword/scratchpad modeling before parameter scaling.

### M8: Autonomous Curriculum & Growth Controller
- Minimal learning controller implementing state machine:
  $$\text{EVALUATE} \to \text{SELECT\_TASK} \to \text{PLAN} \to \text{TRAIN} \to \text{VERIFY} \to \text{PROMOTE / ROLLBACK} \to \text{STOP}$$
- Machine-verifiable procedural task suite with independent external ground-truth verifiers: addition, subtraction, multiplication, parity, string reversal.
- Bounded training experiments: explicit step, batch, and token compute budgets.
- Isolated candidate training (`candidate_dir`): base checkpoint is guaranteed immutable during candidate evaluation.
- Dual-gate candidate verification: candidate must achieve minimum target skill improvement ($\Delta \text{acc} \ge \text{min\_gain}$) while strictly preserving prior task performance within tolerance ($\Delta \text{loss}_{\text{retention}} \le \text{tolerance}$).
- Verified rollback mechanism: candidates failing verification are cleanly rejected, keeping base checkpoint active without degradation.
- Crash and interruption recovery via persistent `controller_state.json`.
- `amg auto-learn` CLI interface.

### M9: Natural Language Learning & Practical Reasoning
- Deterministic Byte-Pair Encoding (`BPETokenizer`) subword tokenizer with explicit versioning (`bpe-v1`).
- Special tokens reserved for chat & instruction formatting: `<pad>`, `<unk>`, `<bos>`, `<eos>`, `<|user|>`, `<|assistant|>`, `<|system|>`.
- Strict lineage check (`validate_checkpoint_tokenizer_lineage`): prevents silent reinterpretation of character embeddings as subword tokens.
- Curated high-quality natural language corpus with procedural step-by-step scratchpads for multi-digit addition, subtraction, and general knowledge.
- `InstructionDataset` with loss masking on prompt tokens (supervising only assistant responses with cross-entropy `ignore_index=-1`).
- Language evaluation suite measuring instruction-following format compliance and held-out reasoning answer accuracy.
- T4-compatible scaling configuration: `configs/m9_language.yaml` (512 subword vocab, 4 layers, 128 d_model, 512 d_ff).
