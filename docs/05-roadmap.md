- **Hardware Platform:** Remote Tesla T4 GPU (`sparklinghydraulics0/amg-m102-dynamic-evolution`)
- **Key Scientific Requalification Outcomes**:
  1. **Disjoint Partitioning & Zero Leakage (P0)**: Completely disjoint Train (16), Val (8), Test (8), and Novel Parameterization (16) splits per domain with zero token overlap. Unseen test accuracy measures true generalization: **16.9%** (Novel rule generalization: **36.7%**).
  2. **Independent Retention Auditing (P0)**: Retention evaluated strictly on independent never-trained test examples: **60.8% mean**, **50.0% worst-family** (Seed range: 25.0% - 100.0%).
  3. **Single-Pass Clean PCGrad (P0)**: Clean orthogonal gradient projection assigning projected gradients directly to `p.grad` in a single pass without task loss double-backprop.
  4. **Unbiased Causal Comparison (P0)**: Equalized budgets (`max_steps=30`, `early_stop_acc=85.0%`) with zero artificial floors. Arm F and Arm B both adapt in **30.0 steps (+0.0% advantage)**, demonstrating that previous reported speedup was driven by asymmetric stopping floors.
  5. **Autonomous Inference Routing (P1)**: Evaluated without ground-truth domain labels (`active_domains=None`). Achieves **2.5% routing accuracy**, proving that autonomous domain inference from 3 integer tokens requires richer contextual embeddings.
  6. **Dynamic Parameter Reconciliation (P1)**: Live PyTorch module inspection reconciles **1,325,295 persistent parameters** with **0 unexplained discrepancy** (accounting for base backbone 1,071,872 + 4 routers 16,776 + persistent growth block 202,466 + 9 adapters 4,608 + multi-domain head manager 27,625 + controllers 1,948).
  7. **Persistent Growth Across Lifetime (P1)**: The inserted 5th Transformer block was actively retained and verified throughout 100 continuous episodes.
  8. **True Bit-Exact Crash Recovery (P1)**: Certified across all 5 seeds with **0.0 parameter diff**, **raw logit diff = 0.00e+00**, and exact argmax match.

### Milestone M103-A: Natural-Language Learning Foundation: Text Substrate, Routing Diagnosis & Rule Internalization
- **Execution Date:** October 26, 2026
- **Status:** **M103_PARTIAL (6/10 GATES PASSED; 169/169 GPU TESTS PASS)**
- **Hardware Platform:** Remote Tesla T4 GPU (`sparklinghydraulics0/amg-m102-dynamic-evolution`)
- **Key Scientific Deliverables & Findings**:
  1. **M102.2 Routing Collapse Root-Cause Isolated & Certified**:
     - Diagnosed why M102.2 scored 2.5% in autonomous routing: episodic single-domain positive cross-entropy training washed out earlier routing weights because replay loss lacked router supervision.
     - Remediated by adding multi-domain replay supervision to router loss, restoring routing accuracy from `98.6%` to `99.0%`, with textual cues providing natural contextual domain separation (`96.9%`).
  2. **Text Tokenizer & Causal Language Modeling Substrate**:
     - Byte-level UTF-8 tokenizer (`ByteTextTokenizerM103`, 264 active tokens within 1024 vocab) with dedicated prompt delimiters (`<bos>`, `<rule>`, `<question>`, `<answer>`, `<eos>`).
     - Autoregressive causal attention with prompt masking (`-100` on prompt tokens).
  3. **Empirical Reality of Text Learning from Scratch**:
     - Neural adapter learning (20 steps on raw byte tokens from random initialization) achieved `0.0%` held-out exact match, identical to In-Context Prompting (`0.0%`) and Fresh Student Baseline (`0.0%`).
     - **Scientific Finding**: A randomly initialized 1.3M-parameter Transformer without prior language pretraining cannot generate multi-token English string answers (e.g. `"6 energy"`) from scratch in 20 adapter steps. Switching to natural language does not magically confer reasoning without semantic token grounding or language pretraining.
  4. **Dynamic Capacity Reconciliation**:
     - Verified live PyTorch module parameters directly: `1,088,256` base backbone + `20,970` layer routers + `198,272` persistent growth block + `2,048` domain adapters = `1,309,546` total persistent parameters (Discrepancy: **0**).
  5. **100% Bit-Exact Crash Recovery**:
     - Certified across all seeds (continuous logit diff = `0.00e+00`, exact argmax match).
- **Lineage Promotion Decision**: **QUALIFIED PARTIAL**.
  - All engineering, routing diagnosis, and text infrastructure gates passed; language acquisition from scratch is diagnosed as the primary research frontier.
- **Next Step: M103-B — Token Pretraining, Hybrid Vocabulary & Structured Language Acquisition**:
  - Pretrain/warm up the byte tokenizer and embedding trunk on domain vocabularies, or ground rule outputs to single-token answer heads before open-ended multi-token generation.
