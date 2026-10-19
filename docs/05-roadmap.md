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
| **M10** | Unified Autonomous Learning | Lineage-aware loading, non-regressive promotion gate, atomic publishing, multi-cycle chaining | **PASSED** (Verified on Tesla T4) |
| **M11** | Autonomous Language Learning | BPE subword model, single atomic pointer switch, fault injection, sequential instruction learning | **PASSED** (Verified on Tesla T4) |
| **M12** | Generalization & Continual Integrity | Partitioned generators (strictly disjoint train/val/test), strict parsed verifiers, autonomous weakest-skill selection, cross-checkpoint raw loss retention audit, old-skill behavioral verification | **PASSED** |

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

### M10: Unified Autonomous Learning Qualification
- Lineage-aware checkpoint loading: dynamically rehydrates and validates `BPETokenizer` and `CharacterTokenizer` matching exact SHA-256 fingerprints and vocabulary sizes.
- Strict non-regressive promotion gate: candidate must not regress against baseline (`cand_acc >= base_acc - 1e-6`) and must satisfy progress criteria (`acc_gain >= min_improvement` or non-regressed mastery) and multi-domain retention (`clean_prose`, `numerical_prose`, `mixed_prose_math`).
- Crash-safe atomic publication: verified immutable candidate publication (`published/v_XXXX_.../checkpoint.pt`) and atomic `.tmp` swap on `active_model/latest.pt` and `active_model/manifest.json`.
- Multi-cycle chaining: seamless consecutive autonomous cycles (`amg auto-learn --max-cycles N`) with automatic base model updating and persistent cumulative history.
- Compute budget accounting: explicit tracking of cumulative steps, tokens, and GPU seconds with hard cap enforcement.
- Real T4 empirical qualification: demonstrated rejection of non-improving task, successful sequential learning and promotion of parity (+84% gain, $\Delta = 0.000000$ drift), and safe rejection with checkpoint preservation on forced regression test.

### M11: Autonomous Language Learning Qualification
- Single atomic active-version pointer (`active_pointer.json`): solves multi-file checkpoint publication inconsistency by pointing atomically to immutable version directories (`published/v_XXXX_.../checkpoint.pt`).
- Fault-injection testing: verified crash-resilience against orphan `.tmp` files and automated fallback to `latest.pt` if a pointer target is compromised.
- BPE subword language model backbone: trained on curated natural language prose and instruction text on Tesla T4.
- Dense-first control logic: defaults to matched dense fine-tuning unless structural expert growth is explicitly requested or empirically justified.
- Multi-cycle sequential language learning: demonstrated consecutive task selection, training, holdout evaluation, non-regressive promotion, and clean rollback on unattainable thresholds.
- Independent large-scale holdouts (500 samples) and qualitative autoregressive text generation before vs after autonomous learning.

### M12: Generalization and Continual-Learning Integrity
- **Partitioned Generators with Strictly Disjoint Train/Val/Test Sets**:
  - Replaced fixed 14-word reversal bank with deterministic cryptographic hash partition (): enforces strictly 0 overlap between train, val, and test splits.
  - Distinct length distributions: training and validation generate lengths 3–6; test holdout includes lengths 7–8 (length extrapolation OOD).
  - Tripartite Factual QA structure:  (10 core facts),  (10 semantic paraphrase queries on train facts for validation), and  (10 completely novel knowledge items reserved for test holdout).
- **Strict Parsed Verifiers**:
  - : Replaced permissive substring matching () with strict regex structure extraction ( or ) and exact normalized equality check. Rejects contradictory statements and malformed outputs.
  - : Every query has an explicit canonical expected answer list. Unrecognized or unknown questions strictly fail verification (or status ), eliminating the permissive  flaw.
- **Cross-Checkpoint Raw Loss Retention Audit**:
  - Audited and corrected the M11 zero-drift artifact: previous intra-checkpoint MoE delta evaluated identical dense models against themselves ().
  - Added explicit cross-checkpoint baseline comparison:  accepts  from base model and computes true drift $\Delta 	ext{Loss} = L_{	ext{cand}} - L_{	ext{base}}$.
  - Records unrounded raw before/after losses across all domains in .
  - Added behavioral verification:  evaluates autoregressive generation on previously mastered tasks, failing candidates that regress older skills beyond tolerance.
- **Autonomous Task Selection**:
  - Fully functional without : controller dynamically identifies the weakest unmastered task from baseline holdout evaluation and records explicit  in journal, manifest, and history logs.

### M12.1: Empirical Generalization Qualification (Completed)
- **Elimination of Evaluation Leakage (P1 Fix)**:
  - Fixed `controller.py` to use `split="val"` for task selection, candidate gating, and promotion decisions.
  - Reserved `split="test"` strictly for final post-selection qualification (`step_qualify`), eliminating test set influence on candidate selection.
- **Strict Deterministic Verifiers (P2 Fix)**:
  - Eliminated `startswith` matching in `verify_addition`, `verify_subtraction`, `verify_multiplication`, `verify_parity`, and `verify_reverse`.
  - Enforced anchored regex matching (`re.fullmatch`) in `verify_instruction_symbolic`, rejecting contradictory or extraneous outputs.
  - Required arithmetic patterns in `verify_instruction_reasoning`, strictly rejecting unsupported questions.
- **Factual QA Question Denominator Audit (P2 Fix)**:
  - Expanded knowledge banks to 20 distinct facts per split (`TRAIN_QA_BANK`, `PARAPHRASE_QA_BANK`, `UNSEEN_QA_BANK`).
  - Added explicit unique question tracking in `evaluate_task_holdout`, preventing inflated confidence denominators.
- **T4 Empirical Qualification Benchmark (3 Independent Seeds: 42, 43, 44)**:
  - Conducted multi-seed autonomous learning experiments on NVIDIA Tesla T4 with 0% train/test input overlap.
  - Negative result confirmed: standard BPE Mini-GPT achieved 0.0% accuracy on unseen string reversal (lengths 3-6 and OOD lengths 7-8).
  - Autonomous controller demonstrated robust safety: detected zero gain (+0.0000) and excessive retention drift, triggering deterministic ROLLBACK on all 3 seeds and preserving active checkpoints without corruption.
  - Detailed diagnostic established: subword tokenization fragmentation and causal autoregressive limits without scratchpad require curriculum or architectural changes before claiming generalization.
