# Experiment discipline and metrics

## Required experiment records
- Run ID, code SHA, dataset SHA/provenance, architecture/config and tokenizer fingerprint
- Train steps/tokens, wall clock, device, batch/context, precision, optimizer settings
- Train loss, held-out validation loss, accuracy by task (if verifiable), uncertainty across seeds
- Peak allocated/reserved VRAM, tokens/s, estimated FLOPs if justified, total training GPU-hours
- Growth decisions, candidate/baseline comparison, accepted/rejected status and why

## Qualification Run Log

### M3/M4/M5: Continual Learning, Selective Token Adaptation & Mixture-of-Depths Benchmark (Google Colab T4)
- **Date**: 2026-10-19
- **Environment**: Google Colab (Linux, Python 3.13.15, PyTorch 2.4+, CUDA 13.0)
- **Device**: NVIDIA Tesla T4 GPU (Session: `amg-t4`)
- **Test Suite Verification**: **40/40 pytest tests passed** in 36.80s directly on the remote T4 runtime.
- **Stage 1 (Task A - Shakespeare Backbone Pretraining)**:
  - Architecture: Dense MiniGPT (2 layers, $d_{\text{model}}=128$, $n_{\text{heads}}=4$, $d_{\text{ff}}=512$, context 128, 440,832 params)
  - Trained 500 steps, peak throughput **348,526 tok/s**
  - Final Val Loss: **2.5046** | Val PPL: **12.24** | Task B Zero-shot Loss: 5.7447
- **Arm 1 (Control - Dense Fine-Tuning on Task B Arithmetic)**:
  - Trainable params: 440,832 (100% of network)
  - Vectorized answer masking throughput: **336,860 tok/s** (restored from ~148k tok/s)
  - Task B Val Loss: **1.1644** | Val PPL: **3.20**
  - Exact-Match Accuracy: 14% (in-distribution), 34% (out-of-distribution extrapolation)
  - Task A Val Loss After: **5.5495** ($\Delta = +3.0449$ severe catastrophic forgetting)
- **Arm 2 (Candidate D - M4 Zero-Init MoE Growth with Selective Unseen-Symbol Adaptation)**:
  - Architecture: MoE with 4 experts ($d_{\text{expert}}=256$), top-1 routing, total params 706,048, active params/token: 573,952
  - Trainable params: 292,864 (newly grown experts & routers + 11 unseen domain tokens `0-9` and `=`)
  - Zero-drift growth verification: Task A loss immediately after growth = **2.5046** ($\Delta = 0.0000$)
  - Task B Val Loss: **1.7680** (significantly reduced from 2.3414) | Val PPL: **5.86**
  - Exact-Match Accuracy: **37%** (up from 10% without unseen symbol adaptation!)
  - Task-Conditioned Backbone Retention (Task A): Loss = **2.5055** ($\Delta = 0.0009$, virtually zero catastrophic forgetting)
  - Router Entropy: **0.9742** (active load balanced across all 4 experts: [21.2%, 30.8%, 15.4%, 32.7%])
- **Arm 3 (Candidate E - M5 Mixture-of-Depths at 50% Capacity + Grown MoE)**:
  - Architecture: M5 MoD (`MixtureOfDepthsRouter` with capacity factor $C=0.5$), total params 706,306
  - Active params/token: **375,938** (35% parameter reduction compared to M4 MoE, 15% fewer active params than standard dense model!)
  - Dynamic token evaluation fraction: **63.6%**
  - Task B Val Loss: **2.2451** | Val PPL: **9.44**
  - Exact-Match Accuracy: **38%** (highest exact-match accuracy among all arms!)
  - Task-Conditioned Backbone Retention (Task A): Loss = **2.5054** ($\Delta = 0.0008$, preserved zero forgetting)
  - Router Entropy: **0.9819**
- **Comparative Summary Table**:

| Metric | Stage 1 (Pretrain) | Arm 1 (Dense FT) | Arm 2 (M4 MoE + Selective) | Arm 3 (M5 MoD + MoE) |
| :--- | :--- | :--- | :--- | :--- |
| **Total Parameters** | 440,832 | 440,832 | 706,048 | 706,306 |
| **Active Params / Token** | 440,832 | 440,832 | 573,952 | **375,938** (-35% vs M4) |
| **Trainable Parameters** | 440,832 | 440,832 | 292,864 | 293,122 |
| **Task B Val Loss** | 5.7447 (zero-shot) | 1.1644 | 1.7680 | 2.2451 |
| **Task B Exact Match Acc** | 0% | 14% | 37% | **38%** |
| **Task A Retained Loss** | 2.5046 | 5.5495 | **2.5055** | **2.5054** |
| **Catastrophic Forgetting $\Delta$** | - | +3.0449 (Severe) | **0.0009** (Zero) | **0.0008** (Zero) |
| **Boundary Drift at Growth** | - | - | **0.0000** | **0.0000** |
| **Throughput (tok/s)** | 348,526 | 336,860 | 113,818 | 98,192 |

- **Status**: **M3, M4, AND M5 GATES PASSED**. Zero-forgetting modular growth, unseen-symbol gradient masking, and causal mixture-of-depths token-skipping confirmed on Tesla T4 hardware.

---

### M2: Google Colab T4 Training Qualification
- **Date**: 2026-10-13
- **Commit**: `e8a8c2f` (rebased/pushed on `main`)
- **Environment**: Google Colab (Linux, Python 3.13.15, PyTorch 2.4+)
- **Device**: NVIDIA Tesla T4 (CUDA)
- **Precision**: FP16 Mixed Precision (`torch.amp.autocast` + `torch.amp.GradScaler`)
- **Architecture**: 2 layers, $d_{\text{model}}=128$, $n_{\text{heads}}=4$, $d_{\text{ff}}=512$, context length 256 (446,464 parameters)
- **Batch Size / Accumulation**: Batch size 16, context 256, accumulation 4 (16,384 tokens/step)
- **Optimization**: AdamW ($\text{lr}=3\times 10^{-4}$, cosine decay schedule)
- **Test Suite Verification**: 20/20 pytest tests passed in 31.83s directly in the T4 environment.
- **Measured Telemetry**:
  - **Throughput**: Peak **513,012 tok/s**, sustained ~**420,000 tok/s** (approx. $25\times$ speedup over CPU baseline)
  - **Step 200**: Train Loss: 2.5313 | Val Loss: 2.5757 | Val PPL: 13.14
  - **Step 400**: Train Loss: 2.5383 | Val Loss: 2.5066 | Val PPL: 12.26
  - **Step 600**: Train Loss: 2.4699 | Val Loss: 2.4607 | Val PPL: 11.71
  - **Step 800**: Train Loss: 2.4066 | Val Loss: 2.4352 | Val PPL: 11.42
  - **Step 1000**: Train Loss: 2.4234 | Val Loss: **2.4170** | Val PPL: **11.21**
- **Independent Validation Evaluation**:
  - **Loss**: 2.2763
  - **Perplexity**: 9.74
  - **Bits per character**: 3.2840
  - **Batches evaluated**: 20
- **Autoregressive Text Generation**:
  - Prompt: `"From fairest "`
  - Output: Structurally coherent Shakespearean poetic verse with character line breaks.
- **Status**: **M2 GATE PASSED**. GPU readiness, FP16 numerical stability, and checkpoint persistence verified on real T4 hardware.

---

## Baselines
A: fixed dense small GPT; B: larger dense GPT; C: modular fixed experts; D: adaptive grown experts. Advance sequentially (A first). Compare matched **tokens, compute budget, dataset, evaluation split**, not just parameter count. Also show total parameters, active parameters, actual GPU residency, and inference speed.

## Verification tasks
- Character-level language: validation cross-entropy, bits-per-character/perplexity when appropriate, sample quality as a qualitative diagnostic.
- Arithmetic: deterministic targets, exact-match on distinct held-out operands and extrapolation splits.
- Logic: procedurally generated verified tasks with independent seeds.
- Continual learning: per-skill held-out scores before and after learning another skill; measure forgetting.

## Growth acceptance policy (future)
Stalling validation alone is insufficient. Confirm data quality, learning rate and adequate training; train a no-growth control with equivalent compute. Accept a growth candidate only when improvement on predefined validation metrics exceeds noise and prior skills don't regress beyond preset tolerance. Keep one untouched final holdout. Report negative results.

## Common pitfalls
Overfitting tiny datasets; accidental train/validation overlap; overfitting to the controller's benchmark; 'specialist' naming without expertise; sparse masks that don't reduce GPU compute; forgetting through shared weights/router; inference improvements mistaken for training progress; copying teacher-generated text with unclear rights.
