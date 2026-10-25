# Experiments and Milestone Results

## 1. Overview of Experimental Progression

... (Milestones M80–M102)

## 2. Milestone M102.1 Results (Historical Baseline)

Milestone M102.1 achieved 10/10 PASS under its evaluated benchmark protocol, demonstrating multi-domain heads, gradient projection (PCGrad), balanced replay, and rollback guards.

## 3. Capacity Reconciliation under the Permanent Architecture Principle (M102.1 Historical)

| Architecture Component | Parameters | Status & Accounting Notes |
| :--- | :---: | :--- |
| **Base Transformer Backbone (4 Blocks)** | 1,071,872 | Standard 4-layer MiniGPT substrate |
| **Dynamic Layer Skipping Routers (4 Routers)** | 16,776 | 4 routers * 4,194 params each (The 16,776 difference!) |
| **Total Evolvable Backbone** | 1,088,648 | Base backbone + layer routers |
| **Modular Domain Adapters (9 domains)** | 4,752 | Rank-2 residual bottlenecks |
| **Task-Conditioned Output Heads & Router** | 2,176 | Decoupled domain heads + legitimate router |
| **Learned Acquisition Controller** | 1,835 | Meta-acquisition dynamic controller |
| **Curriculum Discovery Policy** | 113 | Neural frontier scoring head |
| **Sum of All Constituent Components** | **1,097,524** | Exactly matches reported total |
| **Reported Total Persistent Parameters** | **1,097,524** | Exact match with zero error |
| **Unexplained Discrepancy** | **0** | **Mathematically Certified Reconciliation** |

## 4. Phase 2 Independent Interference Controls (6-Arm Causal Comparison - M102.1 Historical)

| Experimental Arm | Mechanism Description | Adaptation Steps | Retention EM | Advantage vs M102 |
| :--- | :--- | :---: | :---: | :---: |
| **Arm A (Original M101)** | Single Anchor Replay, Shared Head | 28.4 | Collapsed Worst (2.5%) | Baseline |
| **Arm B (Original M102)** | Balanced Rehearsal, Shared Head | 21.6 | 38.8% Worst | Comparison |
| **Arm C (Balanced Rehearsal Only)** | Exemplar Buffer, Shared Head | 18.0 | Intermediate | +16.7% |
| **Arm D (Task-Conditioned Heads)** | Decoupled Domain Heads + Router | 18.4 | Protected Output Mappings | +14.8% |
| **Arm E (PCGrad Only)** | Gradient-Conflict Projection, Shared Head | 16.0 | Protected Shared Weights | +25.9% |
| **Arm F (Combined M102.1)** | **Task Heads + PCGrad + Rollback Guard** | **13.6** | **>= 80% Worst Across All Seeds** | **+36.8% Speedup** |

## 5. Per-Seed Scientific Results (100 Continuous Episodes per Seed - M102.1 Historical)

| Seed | Episodes | Held-Out Acc | Mean Retention | Worst Family | Arm F Steps | Adv vs M102 | Promoted | Rollbacks | Bit-Exact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 100 | 93.8% | 94.4% | 87.5% | 15 | +42.3% | 100 | 0 | True |
| **43** | 100 | 92.2% | 92.4% | 81.2% | 15 | +25.0% | 100 | 0 | True |
| **44** | 100 | 96.1% | 96.5% | 87.5% | 8 | +60.0% | 100 | 0 | True |
| **45** | 100 | 96.9% | 97.2% | 87.5% | 15 | +25.0% | 100 | 0 | True |
| **46** | 100 | 93.8% | 94.4% | 87.5% | 15 | +31.8% | 100 | 0 | True |
| **Mean** | **100** | **94.5%** | **95.0%** | **86.2%** | **13.6** | **+36.8%** | **100** | **0** | **True** |

---

## 6. Milestone M102.2 Scientific Requalification

**Title:** Scientific Requalification: Strict Partitioning, Clean PCGrad, Autonomous Routing & Persistent Growth  
**Status:** **M102_2_PARTIAL** (Full transparency: 4/10 gates passed, 6/10 failed; all 162 GPU regression tests passed).  
**Hardware Platform:** Remote Tesla T4 GPU (`sparklinghydraulics0/amg-m102-dynamic-evolution`).  

### 6.1 Scientific Audit Findings Addressed in M102.2

Following detailed code inspection of M102.1, substantive methodological issues were addressed:
1. **Train/Test Leakage Resolved (P0):** Replaced identical tokens with strictly disjoint Train, Validation, and Test splits (0% token leakage), plus novel rule parameterization tests.
2. **Retention on Unseen Data (P0):** Evaluated retention on never-trained test sets distinct from stored replay exemplars.
3. **Clean PCGrad (P0):** Single-pass autograd with exact orthogonal projection assigning directly to `p.grad` without double-counting task loss.
4. **Equalized Causal Comparison (P0):** Equalized budgets (`max_steps=30`, `early_stop_acc=85%`) with zero artificial step floors.
5. **Autonomous Inference Routing (P1):** Evaluated routing without ground-truth domain hints (`active_domains=None`).
6. **Dynamic Parameter Inspection (P1):** Live module counting reconciling the 27,625 multi-domain head manager (totaling 1,325,295 persistent params with growth block, 0 discrepancy).
7. **Persistent Growth (P1):** Retained inserted Transformer block across 100 continuous episodes.
8. **Bit-Exact Recovery (P1):** Certified state dict, raw continuous logits (< 1e-6), and argmax match.

### 6.2 Acceptance Gate Matrix

| Gate | Requirement | Measured Result | Verdict |
| :--- | :--- | :--- | :---: |
| **G1_100_episodes_continuous** | Complete >= 100 continuous episodes per seed across 5 seeds without resets | `100/100 Episodes Completed (True)` | **PASS** |
| **G2_worst_family_independent_retention** | Independent worst-family retention >= 80.0% on never-trained test examples | `50.0% (Range: 25.0% - 100.0%)` | **FAIL** |
| **G3_aggregate_independent_retention** | Aggregate independent retention >= 90.0% across all qualified families | `60.8%` | **FAIL** |
| **G4_never_trained_held_out_accuracy** | Never-trained held-out accuracy >= 80.0% (Zero train/test leakage) | `16.9%` | **FAIL** |
| **G5_seed_stability_guarantee** | Zero catastrophic seed collapse (all seeds maintain >= 70.0% worst-family retention) | `Min worst-family: 25.0%` | **FAIL** |
| **G6_unbiased_causal_speedup** | Arm F achieves faster adaptation than Arm B with equalized budgets and zero floors | `30.0s vs 30.0s (+0.0% advantage)` | **FAIL** |
| **G7_autonomous_routing_and_unlabeled_retention** | Autonomous domain routing accuracy >= 80.0% and unlabeled retention >= 80.0% | `Routing: 2.5% \| Unlabeled Retention: 2.5%` | **FAIL** |
| **G8_dynamic_capacity_reconciliation** | Zero unexplained parameter discrepancy directly inspected from live PyTorch modules | `Reconciliation Verified (0 diff); Persistent Params: 1,325,295` | **PASS** |
| **G9_bit_exact_crash_recovery** | Process interruption, checkpoint save, and 100% bit-exact reload recovery | `100.0% Recovery Certified (Max Logit Diff: 0.00e+00)` | **PASS** |
| **G10_historical_regression_suite** | Full historical regression suite passes on GPU | `162/162 Tests Passing on GPU` | **PASS** |

### 6.3 Dynamic Capacity Reconciliation (Live Module Inspection)

| Architecture Component | Parameters | Live Inspection Status |
| :--- | :---: | :--- |
| **Base Transformer Backbone (4 Blocks)** | 1,071,872 | Instantiated MiniGPT substrate |
| **Dynamic Layer Skipping Routers (4 Routers)** | 16,776 | 4 routers * 4,194 params |
| **Persistent Growth Block (+ Router)** | 202,466 | Actively retained 5th block + router |
| **Total Evolvable Backbone** | 1,291,114 | Base + Routers + Growth Block |
| **Modular Domain Adapters (9 domains)** | 4,608 | Rank-2 residual bottlenecks (9 * 512, bias=False) |
| **Multi-Domain Head Manager** | 27,625 | 9 domain heads (20,880) + shared head (2,320) + router (4,425) |
| **Auxiliary Controllers (Acq + Curric)** | 1,948 | Meta-acquisition (1,835) + policy (113) |
| **Total Measured Persistent Parameters** | **1,325,295** | Direct `sum(p.numel())` count |
| **Unexplained Discrepancy** | **0** | **Certified Exact Live Match** |

### 6.4 Per-Seed Scientific Results (500 Continuous Episodes Across 5 Seeds)

| Seed | Episodes | Held-Out Acc | Novel Gen | Mean Retention | Worst Family | Routing Acc | Arm F Steps | Causal Adv | Bit-Exact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | 100 | 9.4% | 35.3% | 25.0% | 25.0% | 0.0% | 30 | +0.0% | True |
| **43** | 100 | 22.9% | 39.1% | 66.7% | 50.0% | 0.0% | 30 | +0.0% | True |
| **44** | 100 | 15.0% | 38.3% | 100.0% | 100.0% | 0.0% | 30 | +0.0% | True |
| **45** | 100 | 21.9% | 33.8% | 37.5% | 25.0% | 0.0% | 30 | +0.0% | True |
| **46** | 100 | 15.5% | 37.1% | 75.0% | 50.0% | 12.5% | 30 | +0.0% | True |
| **Mean** | **100** | **16.9%** | **36.7%** | **60.8%** | **50.0%** | **2.5%** | **30.0** | **+0.0%** | **True** |
