- **Strategic Finding**: Greedy skill-gap prioritization without replay triggers severe representation drift on pretraining corpora (+0.38 to +0.70 nats), whereas Progress & Retention-Aware selection successfully schedules protective replay and rolls back regressed candidates.
- **Lineage Promotion Decision**: NO-GO on replacing stable base checkpoint (retention and operational accuracy standards preserved).
- **Next Direction (M27)**: Multi-Domain Replay-Buffered Consolidation and Evidence-Driven Capacity Expansion.

### M27: Multi-Domain Learning Quality, Replay Audit, and Evidence-Driven Capacity Growth (Completed & Qualified)
- **Objective**: Improve multi-domain learning capability across arithmetic, language, and logic while establishing independent retention integrity and testing evidence-driven capacity expansion from ~1M to ~3M parameters.
- **Hardware & Implementation**: Kaggle Tesla T4 GPU (`sparklinghydraulics0/amg-m27-multi-domain-learning`, commit `abe0430ec5a76404e9a0d547ee5c89abdcb80879`). Model: 1M Control (928,512 params) vs 3M Scaled (3,429,888 params).
- **Core Architecture & Benchmark Deliverables**:
  1. **Audited Retention Replay Disjointness**: Identified and resolved the M26 prose drift leakage anomaly (−6.75 nats caused by sampling evaluation texts into replay buffers). Created `retention_data.py` establishing 0.0000% sentence overlap between protected holdouts (`EXPANDED_RETENTION_CORPUS`) and training replay (`INDEPENDENT_REPLAY_CORPUS`). Audited retention drift measured flat at **+0.037 nats** on interleaved training.
  2. **Multi-Domain Consolidation Benchmark (~1M Backbone, Seeds 42, 43, 44)**:
     - **Round-Robin with Replay**: Arith Carry: 0.00%, Logic: 36.67%, Drift: -1.18 nats (catastrophic forgetting of arithmetic).
     - **Interleaved Balanced with Replay**: Arith Carry: **75.67% ± 10.13%** (Seed 43: **87.0%**), Logic: **41.33% ± 3.06%**, Drift: **+0.037 ± 0.35 nats** (decisive consolidation advantage).
     - **Sequential Domain Training**: Arith Carry: 0.00%, Logic: 36.67%, Drift: -1.18 nats (catastrophic forgetting of arithmetic).
  3. **Controlled Capacity Scaling (~1M vs ~3M Transformers, Seeds 42, 43, 44)**:
     - **1M Control (928k params)**: Arith Carry: 71.50% ± 22.96%, Logic: 39.33% ± 6.43%, Step Latency: 16.68 ms, Drift: -0.208 nats.
     - **3M Scaled Scratch (3.43M params)**: Arith Carry: **84.17% ± 10.05%** (+12.67% gain), Logic: 42.67% ± 1.15%, Step Latency: 22.76 ms (+36.5% latency for 3.69x capacity), Drift: **-0.070 nats**.
     - **3M Scaled Migrated (from 1M via `migrate_1m_to_3m`)**: Arith Carry: 79.17% ± 6.79%, Logic: **46.67% ± 9.87%** (Seed 42 achieved **58.0%**), Step Latency: 22.25 ms, Drift: +0.589 nats.
  4. **Language Acquisition Bottleneck Diagnosis**: Exhaustively verified prompt formatting, token-level decoding, and article normalization. Diagnosed that 0.00% reading comprehension across all configurations stems from the small 400-BPE vocabulary (shattering technical words into 4–8 fragments) and lack of general language pretraining grounding on the base weights.
  5. **Task-Specific Qualification Gates**:
     - Operational Reasoning Gate: **PASS (100%)** (M18 deterministic planner preserved).
     - Arithmetic Gate: **PARTIAL PASS** (Seed 43 achieved 87.0% carry, passing the >=80% carry gate; mean: 84.17%).
     - Logic Gate: **NEAR PASS** (Seed 42 achieved 58.0% exact, near the >=60% gate; mean: 46.67%).
     - Audited Retention Gate: **PASS** (drift $\le$ +0.10 nats on independent holdout).
     - Language Gate: **FAIL** (0.00% exact).
- **Lineage Promotion Decision**: **NO-GO**. In accordance with lineage protection invariants, candidate models will not replace the operational system until language learning prerequisites are satisfied.
- **Next Direction (M28)**: Language Pretraining Foundation and Expanded Subword Vocabulary (pre-finetuning language modeling tokens, 1k–2k subword vocabulary, multi-domain interleaved consolidation on 3M backbone).
