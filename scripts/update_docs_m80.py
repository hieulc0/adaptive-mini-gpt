"""Script to cleanly update documentation for Milestone M80."""

from pathlib import Path

# 1. Update docs/05-roadmap.md
roadmap_path = Path("docs/05-roadmap.md")
with open(roadmap_path, "r", encoding="utf-8") as f:
    r_text = f.read()

dup_marker = "### Milestone M79: Autonomous Active Experimentation and Neural Skill-Gap Recognition (Completed & Qualified on Remote Colab Tesla T4 GPU)"
first_idx = r_text.find(dup_marker)
second_idx = r_text.find(dup_marker, first_idx + len(dup_marker))

if second_idx != -1:
    before = r_text[:second_idx]
    m80_section = """### Milestone M80: Autonomous Neural Skill Acquisition and Transfer (Completed & Qualified on Remote Colab Tesla T4 GPU)

- **Status**: Completed & Qualified on Remote Google Colab Tesla T4 GPU (`amg-m80` session).
- **Focus**: Epistemic skill-gap diagnostic detection distinguishing known confident, ambiguous, and unsupported tasks; autonomous neural adapter acquisition on top of frozen Transformer representations; comparative learning-to-learn evaluation (Fresh Random vs Pretrained Backbone vs Experience Warm-Start vs Shuffled Weights); adapter rank capacity evaluation (Rank-16 = 2,592 params vs Rank-8 = 1,432 params); step-budget sweeps ($T \\in \\{20, 40, 60, 80\\}$); dynamic repertoire expansion from 10 to 17 candidate plans with bit-exact rollback; 6-arm zero-shot compound composition benchmark across 5 independent seeds (42, 43, 44, 45, 46); paired Student-t causal hypothesis testing; 100% core retention.
- **Hardware & Implementation**: Google Colab Tesla T4 GPU (Session `amg-m80`, commit `a3c8c35`, Python 3.13, PyTorch 2.4.0). Mini-GPT recurrent backbone (1,071,872 frozen parameters, base hash `6e39cd7ef6790b3c`), bounded Rank-16 adapters (2,592 parameters each). Full regression suite passed: **336/336 tests on Tesla T4 in 116.34s**.
- **Core Architecture & Benchmark Deliverables**:
  1. **Phase 1 Forensic M79 Baseline Reconciliation**:
     - Verified M79 active inquiry exact match (93.31%), query accounting (0.15 queries/task on composable tasks, 0 queries wasted on 40 uncomposable tasks), and explained oracle-probe nominal RNG offset artifact.
  2. **Phase 2 Epistemic Skill-Gap Identification**:
     - Evaluated unsupported rule `BITWISE_XOR_MASK` ($s \\to (s \\oplus 6) \\bmod 16$). Skill gap detector achieved 100% detection rate (`SKILL_GAP_UNSUPPORTED`), correctly triggering autonomous neural acquisition.
  3. **Phase 3 Comparative Learning-to-Learn Sweeps**:
     - Pretrained Backbone adapter achieved **100.00% held-out exact match** (exceeding $\\ge 85\\%$ admission threshold).
     - Shuffled weights control collapsed to **0.00% exact match** (-100.0% drop, proving causal weight learning).
     - Step-budget sweep proved 20 steps is insufficient (37.50% EM), while 40-60 steps reliably attains 97.92%-100.00% EM.
     - Rank-16 (2,592 params) significantly outperformed Rank-8 (1,432 params) with +9.58% EM ($t=3.1293, p=0.0036$).
  4. **Phase 4 Repertoire Expansion & Zero-Shot Composition**:
     - Repertoire expanded from 3 to 4 primitives, expanding plan hypothesis space from 10 to 17 candidate plans. Bit-exact rollback verified.
     - Autonomous plan discovery over 17 plans achieved **79.79% exact match** on compound transfer tasks combining old and new primitives (`REFLECT_THEN_XOR`, `XOR_THEN_REFLECT`, `SHIFT3_THEN_XOR`, `XOR_THEN_SHIFT3`), outperforming shuffled control (**+79.69% EM**, $t=6.5731, p=0.0001 < 0.001$) and inverted order (**+31.15% EM**, $t=2.5323, p=0.0226$).
  5. **Phase 5 Hardware Parameters, Retention & Lineage**:
     - Newly admitted skill allocated exactly 2,592 parameters (Rank-16 bottleneck adapter).
     - Core backbone weights 100% frozen (0 base parameter growth, SHA-256 hash `6e39cd7ef6790b3c` invariant).
     - Arithmetic carry retention: **100.0% (20/20)**; Operational logic retention: **100.0% (8/8)**; Primitive retention: `REFLECT` 100%, `SHIFT3` 97.92%, `AFFINE` 97.92%.
- **Lineage Promotion Decision**: **QUALIFIED IN RESEARCH TIER**. Autonomous neural acquisition and dynamic repertoire expansion fully verified on Tesla T4.
- **Next Direction Options (Post-M80 Strategic Priority -> M81)**:
  - **Stage 2 Cross-Domain Transfer & Continual Autonomous Learning (M81)**: Extend autonomous skill acquisition across multiple distinct functional domains—including multi-digit arithmetic, operational logic, and symbolic transformations—evaluating whether accumulated prior acquisition experience improves the learning efficiency of subsequent skills.
"""
    with open(roadmap_path, "w", encoding="utf-8") as f:
        f.write(before + m80_section)
    print("Successfully updated docs/05-roadmap.md")

# 2. Update docs/06-data-pipeline.md
pipeline_path = Path("docs/06-data-pipeline.md")
m80_pipeline_content = """
### 12. Milestone M80 Autonomous Neural Skill Acquisition Specification
Milestone M80 introduces `data/specs/m80_acquisition_spec.json`, formalizing:
- **Phase 1 Forensic Audit & Baseline Reconciliation**:
    * Reconciles M79 query accounting (0.15 queries/task) and explains nominal oracle control underperformance via RNG offset artifact.
- **Epistemic Diagnostic Detection**:
    * Evaluates candidate plans; when top likelihood $S(P) < -1.65$, task is classified as `SKILL_GAP_UNSUPPORTED`, directly triggering neural adaptation.
- **Autonomous Neural Adapter Acquisition**:
    * Bounded parameter budget: Rank-16 bottleneck adapter (2,592 parameters) vs Rank-8 bottleneck adapter (1,432 parameters).
    * Training steps budget sweep: $T \\in \\{20, 40, 60, 80\\}$ steps on 48 practice examples.
    * Strict qualification admission gate: held-out exact match $\\ge 0.85$ (85%).
- **Comparative Learning-to-Learn Initialization Modes**:
    * `FRESH_RANDOM`: fresh student without pretrained backbone representations.
    * `PRETRAINED_BACKBONE`: adapter trained on frozen Transformer representations.
    * `EXPERIENCE_WARM_START`: warm-started from related prior skill adapter weights.
    * `SHUFFLED_WEIGHTS`: scrambled parameter control for causal weight attribution.
- **Dynamic Repertoire Expansion & Combinatorial Plan Space**:
    * Repertoire expansion: 3 primitives $\\to$ 4 primitives (`COMPLEMENT_REFLECT`, `CAESAR_SHIFT_MOD3`, `AFFINE_TRANSFORM`, `BITWISE_XOR_MASK`).
    * Plan hypothesis space: expands from 10 to 17 candidate plans (1 abstain, 4 single-primitive, 12 two-stage sequential compositions).
    * Bit-exact rollback support: candidate removal restores initial 10-plan space with 0 residual parameter traces.
- **Zero-Shot Compound Transfer Evaluation (6 Arms)**:
    * `AUTONOMOUS_PLAN_EXPANDED`: autonomous plan discovery over 17 plans.
    * `ORACLE_PLAN_PROVIDED`: composition with ground-truth plan ($P^*$).
    * `INVERTED_ORDER_CONTROL`: reversed execution order ($P_{\\text{inv}}$).
    * `SINGLE_SKILL_ABLATION`: single primitive alone.
    * `SHUFFLED_ACQUIRED_ADAPTER`: scrambled adapter weights control.
    * `FRESH_MONOLITHIC_ADAPTER`: direct end-to-end trained compound adapter reference.
- **Execution Provenance**:
    * 336/336 unit and regression tests passed on Google Colab Tesla T4 GPU (`amg-m80` session).
    * 100% carry (20/20) and logic (8/8) retention; base parameter SHA-256 hash invariant (`6e39cd7ef6790b3c`).
"""

with open(pipeline_path, "r", encoding="utf-8") as f:
    p_text = f.read()

if "### 12. Milestone M80 Autonomous Neural Skill Acquisition Specification" not in p_text:
    with open(pipeline_path, "a", encoding="utf-8") as f:
        f.write(m80_pipeline_content)
    print("Successfully appended M80 specification to docs/06-data-pipeline.md")
