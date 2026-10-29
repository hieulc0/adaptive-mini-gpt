# Foundation Model Defect Register (Stages A.1 – A.12)

This register tracks all critical architectural defects, numerical failure modes, and statistical design limits identified during the foundational Stage A line of development.

---

## 1. Defect Index

| Defect ID | Discovered In | Category | Description | Status | Resolved In |
|---|---|---|---|---|---|
| **FDR-001** | Stage A.4 | Capacity | Rank collapse in frozen attention adapters | **Resolved** | Stage A.6 |
| **FDR-002** | Stage A.5 | Optimization | Loss oscillation under unclipped gradient norms | **Resolved** | Stage A.6 |
| **FDR-003** | Stage A.6 | Architecture | Coordinate overfitting under absolute wpe | **Resolved** | Stage A.9 |
| **FDR-004** | Stage A.7 | Curriculum | Catastrophic forgetting under single-frame exposure | **Resolved** | Stage A.8 |
| **FDR-005** | Stage A.8 | Plasticity | Lower-block freezing impedes syntactic induction | **Resolved** | Stage A.9 |
| **FDR-006** | Stage A.9 | Pretraining | Semantic void in un-preconditioned embeddings | **Resolved** | Stage A.9 |
| **FDR-007** | Stage A.10 | Generalization | Causal attention bleeding across distractor clauses | **Identified** | Stage A.11-R2 |
| **FDR-008** | Stage A.10 | Migration | Downstream head destruction during wpe->RoPE migration | **Identified** | Stage A.12 |
| **FDR-009** | Stage A.11 | Evaluation | Relational interference between single & minimal pairs | **Resolved** | Stage A.11-R2 |
| **FDR-010** | Stage A.11-R | Implementation | Head unpacking mismatch (`state_logits` vs `tag_logits`) | **Resolved** | Stage A.11-R |
| **FDR-011** | Stage A.11-R | Architecture | Multi-entity binding failure in causal last-token readout | **Resolved** | Stage A.11-R2 |
| **FDR-012** | Stage A.11-R2 | Implementation | Evaluator secondary forward strategy bypass (`strategy=None` -> ECA) | **Resolved** | Stage A.11-R2 |
| **FDR-013** | Stage A.11-R2 | Optimization | Single-token causal readout cross-seed variance on multi-entity queries | **Characterized** | Stage A.12 |
| **FDR-014** | Stage A.12 | Architecture | Query blindness & attention shortcut learning in unparameterized cross-readout | **Resolved** | Stage A.13 |
| **FDR-015** | Stage A.13 | Architecture | Causal byte-level unidirectional prefix bottleneck on trailing entity queries | **Characterized** | Stage A.13 |
| **FDR-016** | Stage B.0 | Exploration | REINFORCE sample inefficiency under tight practice horizons in continual learning | **Identified** | Stage B.0 |

---

## 2. Historical Stage Findings & Errata

Stage A.9 empirical findings (`bac611e` / `2070532`, 838 full GPU regressions pass; [report](STAGE_A9.md)):
- Absolute positional embeddings (`wpe`) bind properties to absolute spatial indices, causing 100.0% counterfactual flip failures when sentence syntax shifts.
- Relative position encoding (`RoPE`) combined with 1,500 steps of synthetic language pretraining eliminates coordinate overfitting, achieving 0.00% positional flip rate and 90.48% novel grammar generalization.

Stage A.10 empirical findings (`6bc8f10`, 842 full GPU regressions pass; [report](STAGE_A10.md)):
- 2x2 Factorial decomposition confirmed super-additive synergy between RoPE (+28.31%) and syntactic pretraining (+16.14%), with a +29.63% interaction term.
- All neural models collapsed to 47.84%–53.09% on the independent sealed test suite (`64daf317...`) due to unconstrained causal attention bleeding across distractor clauses. Sealed suite was permanently retired.

Stage A.11 empirical findings (`1a22fd6` / `0620a22`, 848 full GPU regressions pass; [report](STAGE_A11.md)):
- Canonical parent checkpoint digest reconciled to `3e93d72...` raw SHA256 / `aed7480...` PyTorch digest. Discrepant string `d404f2f...` isolated to documentation transcription typo.
- Supervised relational pairs force representations to separate linearly (63.5% probe flip rate), but end-to-end models reported 1.04% pair flip rate due to an evaluation head unpacking bug. Sealed suite untouched.

Stage A.11-R empirical findings (`77a7c3d`, 854 full GPU regressions pass; [report](STAGE_A11R.md)):
- **FDR-010 RESOLVED**: Fixed head unpacking bug in `evaluate_pair_exact_accuracy` (`state_logits` -> `tag_logits`), lifting ARM 4 pair-both accuracy from 1.04% to 57.64% (MP-1: 90.63%, MP-3: 82.29%, MP-2: 0.00%), strictly satisfying the Pigeonhole bound ($P_{\text{both}} \ge 40.96\%$).
- Readout candidates `R_ARM_2_LAST_TOKEN` and `R_ARM_3_QUERY_ATTN_POOL` reported 76.39% / 0.00% and 72.22% / 0.00% on minimal pairs, violating mathematical pigeonhole lower bounds ($52.78\%$ and $44.44\%$).

Stage A.11-R2 empirical findings (`832326b` / `2e1ed16`, 860 full GPU regressions pass; [report](STAGE_A11R2.md)):
- **FDR-012 / DEFECT-020 RESOLVED (Evaluator Secondary Forward Strategy Bypass)**:
  - *Mechanism*: In `scripts/run_stage_a11r.py`, `evaluate_with_paired_metrics()` ran `evaluate(model, batch, cases)` using the candidate's actual `default_strategy` (76.39% case exact), but then performed a second forward pass `model.forward_with_strategy(tokens, actions)` without passing `strategy=model.default_strategy`. In `semantic_bottleneck_m104_e.py`, `forward_with_strategy` had default `strategy: str = "entity_conditioned_attention"`. Because ECA was completely untrained in `R_ARM_2` and `R_ARM_3`, this second pass produced random noise (~25% case accuracy), recording `pair_both_correct_acc = 0.00%` while `result["exact"]` retained 76.39%!
  - *Correction*: `forward_with_strategy` now defaults to `strategy: str | None = None` resolving to `self.default_strategy`. Unified evaluation `compute_paired_metrics_from_predictions()` now computes case and pair metrics from the EXACT same prediction array, enforcing `pair_both_correct_acc >= max(0, 2 * case_exact_acc - 1)`.
  - *True Baseline*: On true student predictions, `R_ARM_2_LAST_TOKEN` actually achieved **76.39% case exact** and **66.67% pair both** (100.00% on MP-1, 100.00% on MP-3, 0.00% on MP-2, pigeonhole bound $66.67\% \ge 52.78\%$ verified).
- **FDR-009 RESOLVED (Training Gate & Batch Balancing)**:
  - By introducing balanced 24-case batches (6 single, 6 MP, 6 EB, 6 rehearsal) and two-timescale cosine annealing (`backbone_lr` 0.0005 $\rightarrow$ 0.00005, `readout_lr` 0.0015 $\rightarrow$ 0.00015), **all three candidates achieved 100.00% training exact match**, resolving the 94.8%–95.8% plateau and passing Gate 1.
- **FDR-011 / DEFECT-021 ADVANCEMENT & REMAINING LIMITATION (Multi-Entity Causal Routing Instability)**:
  - With balanced relational training, `R2_ARM_2_LAST_TOKEN` reached **83.33% overall minimal-pair both-correct accuracy** (clearing the $\ge 80.0\%$ gate), lifting MP-2 from **0.00%** to **54.17% cohort mean**!
  - In Seed 43, MP-2 reached **87.50%** (overall MP both = 95.83%). In Seed 42, MP-2 reached **75.00%** (overall MP both = 87.50%, EB suite both = 77.78%).
  - However, in Seed 44, MP-2 scored **0.00%**, demonstrating that single-token causal readout without specialized cross-clause routing is sensitive to weight initialization.
  - Residual query-gated pooling (`R2_ARM_3`) failed (MP-2 = 2.08%, MP both = 61.81%) because unweighted attention across question tokens dilutes the sharp query target signal.
- **PREREGISTERED SEALED ISOLATION PRESERVED**:
  - Because Gate 3 failed across the cohort (MP-2 = 54.17% vs $\ge 75.0\%$, EB-1..4 = 34.03% vs $\ge 80.0\%$), **`sealed_a11_v1` WAS NOT EVALUATED**. Strict test hygiene was preserved intact.
  - Checkpoint continuation verified ($\Delta \text{logit} = 0.00\text{e+}00$). 860 GPU regression tests passed.

Stage A.12 empirical findings (`96ffaad` / `9c67ce8`, 867 full GPU regressions pass; [report](STAGE_A12.md)):
- **FDR-013 CHARACTERIZED (Phase 1 Query Blindness Forensics)**:
  - Investigated saved immutable A.11-R2 checkpoints with 0 training updates.
  - Isolated that collapsing Seed 44 suffered from **Complete Query Blindness** (0.0% query sensitivity rate on MP-2 pairs): swapping `Query target: e1` to `Query target: e2` produced identical label predictions on 100% of minimal pairs.
  - Detached linear ridge probes on $h_{t_{\text{last}}}$ showed 83.33% accuracy in Seed 44 (vs 95.83% in Seed 43), proving target entity information was present in the hidden activations, but downstream readouts fell into static prefix-based local minima.
- **FDR-014 IDENTIFIED (Curriculum & Cross-Readout Bounds on Entity Binding)**:
  - Matched 1,500 pretrain + 7,200 supervised comparison across Seeds 42, 43, 44 showed:
    - Counterfactual paired curriculum (`A12_ARM_2`) maintained 100% training acquisition and 100% old-frame retention, but dispersed attention over the 28-name pool, dropping cohort MP-2 mean to 6.25%.
    - Query-span cross readout (`A12_ARM_3`) stabilized Seed 44 (MP-2 rose from 0.0% to 43.75%) and Seed 42 (MP-2 = 68.75%), but degraded Seed 43 (down to 9.38%), yielding a cohort mean of 40.62%.
  - Single-clause generalization (`MP-1` = 91.67%–100.0%, `MP-3` = 100.0%, dev familiar/novel = 98.15%–99.74%) and Gates 1, 2, 3A, and 4 passed cleanly.
  - Gates 3B, 3C, and 3D failed due to seed variance ($\ge 75.0\%$ MP-2 per-seed requirement).
- **PREREGISTERED SEALED ISOLATION PRESERVED**:
  - `sealed_a11_v1` was not evaluated and remains 100% unviewed and pristine.
  - Overall Verdict: **`FAIL`**. No checkpoint is promoted. Continuation verified ($\Delta \text{logit} = 0.00\text{e+}00$). 867 GPU regression tests passed.

Stage A.13 & Stage B.0 empirical findings (`edb225b` / `8567836`, 873 full GPU regressions pass; [report](STAGE_A13_B0.md)):
- **FDR-014 RESOLVED & DOCUMENTATION DISCREPANCY RECONCILED (Stage A.13 Phase A13.1)**:
  - Reconciled A.11-R2 (`832326b`) vs. A.12 (`96ffaad`) documentation discrepancy: `docs/STAGE_A12.md` accurately reported `0.00% / 100.00% / 18.75%` (mean `39.58%`), matching `stage_a12_v1.json`, whereas `README.md` in `96ffaad` cited intermediate draft numbers (`87.50% / 100.00% / 0.00%`). Formal erratum documented.
  - Baseline divergence mechanism isolated to `scripts/run_stage_a11r2.py` constructing `balanced_plan` before `seed_all(seed)` vs. `scripts/run_stage_a12.py` invoking it after `seed_all(seed)`.
  - Fixed Phase 1 Table 2 indexing bug in `diagnose_stage_a12_seeds.py` (`tag_logits[:, [6, 7]]` -> `TAG_IDS = (10, 15)`), proving familiar training names achieve **100.00% MP-2 Both across all seeds**.
  - Continuous logit sensitivity proved Seed 44 suffers true representation-level query blindness on unseen names (hidden shift $\Delta_h = 0.31$, cosine distance $0.0004$, OOD ridge probe $50.00\%$ chance).
- **FDR-015 CHARACTERIZED (Track A Architectural Bottleneck on Causal Byte Encodings)**:
  - Controlled comparison between `A13_ARM_1_LAST_TOKEN_CONTROL` and `A13_ARM_2_LEARNED_ENTITY_CROSS_ATTENTION` (using student's registered ECA MultiheadAttention with 0 parameter growth):
    - In `A13_ARM_2`, Seed 43 achieved **100.00% MP-2 Both**, proving cross-attention projections ($W_Q, W_K, W_V$) can route target attributes when the prefix representation supports it.
    - However, Seed 42 collapsed to **0.00%** and Seed 44 to **21.88%** on `MP-2` (mean `40.62%`), replicating extreme seed bifurcation.
    - Architectural root cause: in unidirectional causal byte models, prefix tokens are encoded before the trailing entity query target is observed; attention easily falls into local prefix shortcuts. Multi-pass or bidirectional entity indexing is structurally necessary.
  - Preregistered stop rule executed: `sealed_a11_v1` remained strictly unviewed and untouched.
- **FDR-016 IDENTIFIED & CONTINUAL LEARNING CERTIFIED (Track B Stage B.0 Persistent Self-Learning Pilot)**:
  - Successfully demonstrated that **one continuous `LearningState`** (0 weight/optimizer resets) acquires sequential concepts `A -> B -> C` under tight budgets (160 queries, 400 updates per concept) with 92.5%–100% relative knowledge retention via 25% episodic replay.
  - Isolated comparative deltas:
    - Forward Transfer Delta: $-2.70\%$ AULC ($0.4154$ vs $0.4424$, slight negative transfer under tight budgets).
    - Active Heuristic Delta: $+0.79\%$ AULC ($0.4233$ vs $0.4154$, fixed entropy heuristic outperforms random practice).
    - Learned Policy Delta: $-1.59\%$ AULC ($0.4074$ vs $0.4233$). Under 10 rounds $\times$ 16 queries per concept, REINFORCE exploration variance cannot surpass deterministic entropy heuristics without longer training horizons.
- **CONTINUATION & REGRESSION CERTIFIED**:
  - Deterministic continuation verified bit-perfectly ($\Delta \text{logit} = 0.00\text{e+}00$).
  - Full remote regression suite: **873 passed, 0 failed** in 630.80s on Kaggle Tesla T4 GPU.
