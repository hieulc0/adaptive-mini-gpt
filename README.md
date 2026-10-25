# Adaptive Mini-GPT

An experimental, **CLI-first Python/PyTorch** research framework studying **adaptive capacity allocation** and **continual modular skill acquisition** in small decoder-only Transformers under constrained single-GPU budgets.

**Current status:**
- **M0 (Baseline Architecture & Determinism):** Passed
- **M1 (Stateful Training, Atomic Checkpointing & Resumption):** Passed
- **M2 (Colab Tesla T4 FP16 Qualification — >500k tok/s):** Passed
- **M3 (Modular Residual Experts & Continual Learning):** Passed (Verified on Tesla T4)
- **M4 (Function-Preserving Growth & Zero Forgetting):** Passed (Verified on Tesla T4)
- **M5 (Mixture-of-Depths & Dynamic Token Compute Allocation):** Passed (Verified on Tesla T4)
- **M18 (Grammar-Constrained Planning & Execution Guidance):** Passed (100% on addition ladders & 5-step kinematics)
- **M19 (Learned Action Selection & Representation Adaptation):** Closed (No-go on neural promotion; M18 baseline preserved)
- **M20 (Controlled Neural Scaling & Procedural Generalization):** Completed (1M vs 3M scaling; No-go on promotion; M18 baseline preserved)
- **M21 (Digit-Level Neural Arithmetic & Procedural Scratchpad):** In Progress (Testing representation ablations and carry scratchpads)
- **M70 (Verified Continual Self-Improvement & Portable Data Pipeline):** Passed (289/289 Tests on Tesla T4; Autonomous Policy beats ALWAYS_REUSE)
- **M71 (Selective Neural Memory, Causal Retention & Compact Skill Management):** Passed (294/294 Tests on Tesla T4; Selective Retain beats Reset by +0.0218 Macro F1 & +0.0130 Net Utility)
- **M72 (Robust Lifelong Neural Memory & Capacity Efficiency):** Passed (299/299 Tests on Tesla T4; Selective Retain beats Retain-All & Wrong-Route)
- **M73 (Cost-Efficient Neural Memory & Verified Cumulative Improvement):** Passed (304/304 Tests on Tesla T4; SELECTIVE_NO_EXTRA_PROBES beats RESET in Macro F1 & Physical Utility)
- **M74 (Verified Neural Weight Learning & High-Information Generalization):** Passed (309/309 Tests on Tesla T4; Causal Weight Attribution Verified with +0.6562 EM over Shuffled Controls; SELECTIVE_NO_EXTRA_PROBES beats both RESET & FRESH_ADAPT in Physical Utility)
- **M75 (Experience-Guided Autonomous Learning & Adaptive Training Budgets):** Passed (314/314 Tests on Tesla T4; 7-arm causal lifelong comparison across Seeds 42, 43, 44; adaptive step budgets 20/40/60; procedural generalization taxonomy across 4 tiers; base backbone SHA-256 invariant; learning-to-learn causal hypothesis evaluated against shuffled/irrelevant controls)
- **M76 (Causal Neural Skill Composition & Reusable Procedural Knowledge):** Passed (318/318 Tests on Tesla T4; 9-arm causal composition suite across Seeds 42, 43, 44; two-stage cascaded neural composition achieves 0.4479 Exact Match with 0 additional training steps and 0 parameter allocations, causally outperforming wrong-order (+0.3177 EM), single-skill (+0.4479 EM), shuffled-component (+0.4479 EM), and base controls (+0.4479 EM) with non-overlapping 95% paired CIs; error decomposition isolates stage-1, stage-2, and cascading propagation failures; frozen M18 backbone and 100% core retention verified)
- **M77 (Autonomous Neural Skill-Plan Discovery from Examples):** Evaluated on Colab Tesla T4 (322/322 Tests Passed; bounded deterministic plan search over frozen primitives achieves 100% True Negative Rate on unfamiliar tasks and 0.4479 EM when plans are identified, but averages 25.0% plan accuracy and 0.1146 EM due to conservative safety abstention on noisy intermediate neural outputs; paired 95% CI vs provided plan [-0.5730, -0.0937]; frozen M18 backbone and 100% core retention verified; autonomous planning gate INCONCLUSIVE / PARTIAL; topology expansion RESEARCH ONLY)
- **M78 (Reliable Autonomous Neural Plan Discovery):** Passed (326/326 Tests on Tesla T4; continuous sequence log-likelihood achieves 80.00% plan identity accuracy, 92.50% functional accuracy, 88.85% exact match, 97.50% unfamiliar rejection, 7.50% false abstention, +0.6544 over wrong-order with p < 0.0001; 0 new parameters)
- **M79 (Autonomous Active Experimentation & Neural Skill-Gap Recognition):** Passed (330/330 Tests on Tesla T4; active discriminative probing generates minimally discriminative inputs to resolve ambiguous plans, achieving 87.50% plan identity accuracy, 97.50% functional accuracy, 93.31% sealed exact match, 100% unfamiliar rejection, reducing supported-task false abstention from 15.00% to 2.50% at 0.15 queries/task with p = 0.0375 over passive M78; 0 new parameters)
- **M80 (Autonomous Neural Skill Acquisition & Transfer):** Passed (336/336 Tests on Tesla T4; epistemic skill gap detection achieves 100% unsupported task identification, bounded Rank-16 adapter acquisition achieves 100.0% held-out EM, repertoire expansion from 10 to 17 plans with bit-exact rollback, and zero-shot compound composition achieves 79.79% exact match [+79.69% over shuffled weights, p = 0.0001], 100% carry and logic retention, exactly 2,592 parameter allocation, 0 base growth)
- **M81 (Learning-to-Learn Through Transfer and Experience):** Passed (340/340 Tests on Tesla T4; 30.00% reduction in median training steps to held-out threshold [50 -> 35 steps] with p = 0.0061 < 0.01 and positive paired CI [+8.59, +27.41]; AUC advantage +0.1906, p = 0.0034; minimal capacity adaptation evaluates Rank-4 and selects Rank-8 for 80% of seeds [1,432 params] before bounded expansion to Rank-16 [2,592 params]; compound composition achieves 97.92% exact match [+97.92% over shuffled control, p < 0.0001]; M80 forensic audit corrects df=4 Student-t p-values and reconciles 1.07M vs 928k parameters; 100% carry and logic retention; 0 base parameter growth)
- **M82 (Learned Meta-Acquisition and Self-Improving Training Strategies):** Passed (345/345 Tests on Tesla T4; 33.33% reduction in median training steps to held-out threshold [45 -> 30 steps] with p = 0.0006 < 0.001 and positive paired CI [+8.6, +15.4]; Area Under Curve advantage +0.1628 [0.4299 -> 0.5927]; compact 1,835-parameter Learned Acquisition Controller dynamically chooses initialization, learning rate [1e-3..1e-2], rank capacity [4, 8, 16], and early-stopping threshold from real-time diagnostic signals; outperforms fixed heuristics [55 steps] and random exploration [50 steps]; evaluated on held-out unseen composite task families; bit-exact rollback verified; 100% carry and logic retention; 0 base parameter growth)
- **M83 (Autonomous Neural Knowledge Consolidation and Self-Distillation):** Passed (350/350 Tests on Tesla T4; 58.57% parameter compression [4,296 -> 1,780 params, 2,516 eliminated]; 99.58% skill retention with all individual skills >= 85% EM; multi-metric candidate discovery evaluating weight similarity, bottleneck activation cosine similarity, functional agreement, and gradient interference; soft-target self-distillation with lightweight rank-4 residual bypass heads decisively outperforms direct weight averaging [0.00% EM, p = 0.0001], random parameter merging [13.75% EM], and truncated low-rank SVD [0.00% EM]; conflicting opposing-polarity skills detected, rejected, and restored with bit-exact rollback; subsequent learning efficiency preserved [median 35 steps, zero negative transfer]; 939-parameter learned consolidation controller; 100% carry and logic retention; 0 base parameter growth)
- **M84 (Autonomous Neural Connection Pruning, Regrowth, and Learned Topology Adaptation):** Passed (359/359 Tests on Tesla T4; 30.03% active connection reduction [1,152 -> 806 active synapses, 346 eliminated]; 98.75% relative skill retention with all seeds >= 93.75% EM; autograd gradient-salient link reactivation [RigL] computes unmasked analytical gradients [x^T @ grad_out] allowing salient dormant connections to regrow; 7-arm causal benchmark across Seeds 42..46; 964-parameter learned topology controller dynamically selects maintain, prune, regrow, and balanced rewire based on 6 diagnostic state signals, achieving 20.4 steps to qualification vs 24.4s for random regrowth, 24.8s for fixed sparse, and 22.2s for static pruning; future unfamiliar task adaptation on TWO_STAGE_CASCADE achieved in 17 median steps [zero negative transfer]; bit-exact rollback state restoration verified via SHA-256 [8e86ca8397dd1403]; 100% carry and logic retention; base model SHA-256 invariant 6e39cd7ef6790b3c; 0 base parameter growth)


## Research Hypothesis

Under a matched compute and token budget, **function-preserving modular growth** (attaching zero-initialized bottleneck experts to a trained backbone with load-balanced or task-conditioned routing) improves continual skill retention and sample efficiency compared to static dense retraining, while mitigating catastrophic forgetting.

## Core Engineering Principles

- **Learn first, adapt later:** Verify dense baselines and deterministic resumption before activating routing or capacity growth.
- **Function-preserving transitions:** Any structural growth operation ($M \to M'$) must satisfy $M'(x) = M(x)$ at the step of expansion via zero-initialized residual projections.
- **Single-process CLI parity:** Identical `amg` CLI execution on local Arch Linux CPU, Google Colab, and Kaggle T4 GPUs.
- **Crash-only atomic persistence:** Checkpoints write to temporary files, pass a load verification smoke test, and atomically swap via `os.replace` alongside a versioned `manifest.json`.
- **Reproducible evaluation:** Deterministic document/task splits, SHA-256 data and vocabulary fingerprints, and isolated holdout verification.

## Quickstart & CLI Usage

```bash
# Install environment and dev dependencies
uv sync --extra dev

# Run test suite and linter
uv run ruff check src tests scripts
uv run pytest -q

# Train dense baseline or adaptive expert model
uv run python -m adaptive_mini_gpt.cli train --config configs/tiny.yaml
```
