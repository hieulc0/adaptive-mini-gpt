# Approved audit repair phase

Baseline: branch `m104-e-and-m105-a`, `a6d8e11eda1ecfb7067f1552b7c8e7eb13d2eb11`, clean tree. This is repair work, not a new milestone. Historical reports and failed runs remain immutable.

Plan before implementation:

1. Repair complete-input boundary failures without increasing model capacity: reject padding overflow, use structured markers and model-derived context budgets, and validate that observable descriptions entail the experiment's labels. Preserve the 256-position model as this experiment's configuration; allow other declared budgets without silently resizing weights.
2. Diagnose M90 with a fixed backbone and fixed data while perturbing ambient RNG; compare historical and explicitly seeded initialization, record every rank/task score and preserve the >=85% gate. Isolate reference/joint/sequential initialization streams and seed the backbone before construction. Do not pick seeds based on scores.
3. Inventory loss/target/optimizer/causal interfaces across source; repair only demonstrated defects and add focused tests. Explicit masks must retain causal independence. Audit optimizer membership and real gradients remotely.
4. Reconcile fixed rotary/router plumbing with remaining expert-growth, sequence-routing and transactional-state limitations. No structural growth or learning module is added.
5. Run one finite complete-input experiment: same existing neural student, frozen/selective/full plasticity, mixed training descriptions and disjoint unseen names/descriptions, majority and simple bag-of-words controls, fixed budgets and thresholds. Record full token sequences, label evidence, gradients, checkpoints, quality and cost independently. Stop after negative generalization rather than adding capacity.

LOCAL-SAFE: reads/edits, syntax/lint/type checks and inspected mock/tiny-fixture tests only. REMOTE-REQUIRED: actual neural loss/gradient/causal tests, M90 diagnosis, training and the full suite on Kaggle T4. Rollback: revert the focused commits; no existing checkpoint is overwritten or automatically promoted. GPU budget: one bounded diagnosis/study job plus full regression; fix only observed failures before any justified rerun.
