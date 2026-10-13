# Experiment discipline and metrics

## Required experiment records
- Run ID, code SHA, dataset SHA/provenance, architecture/config and tokenizer fingerprint
- Train steps/tokens, wall clock, device, batch/context, precision, optimizer settings
- Train loss, held-out validation loss, accuracy by task (if verifiable), uncertainty across seeds
- Peak allocated/reserved VRAM, tokens/s, estimated FLOPs if justified, total training GPU-hours
- Growth decisions, candidate/baseline comparison, accepted/rejected status and why

## Qualification Run Log

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
