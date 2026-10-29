# Foundation Model Defect Register (Stages A.1 – A.11-R2)

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
| **FDR-013** | Stage A.11-R2 | Optimization | Single-token causal readout cross-seed variance on multi-entity queries | **Identified** | Stage A.12 |

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
