#!/bin/bash
set -euo pipefail

echo "=== System Preflight Check ==="
python3 -c "import torch; assert torch.cuda.is_available(); print('CUDA Device:', torch.cuda.get_device_name(0), 'PyTorch:', torch.__version__)"

# 1. Checkpoints (optional)
mkdir -p /content/checkpoints
if [ -f /content/m18-checkpoints.tar.gz ]; then
  tar -xzf /content/m18-checkpoints.tar.gz -C /content/checkpoints
fi

# 2. Setup repo
rm -rf /content/adaptive-mini-gpt
echo "Cloning branch m83-knowledge-consolidation..."
git clone --branch m83-knowledge-consolidation https://github.com/hieulc0/adaptive-mini-gpt.git /content/adaptive-mini-gpt

cd /content/adaptive-mini-gpt
git log -n 1 --oneline || true

# 3. Install dependencies
pip install -e . pytest

# 4. Run Pytest Regression Suite on Tesla T4 (all milestones up through M83)
echo "=== Running Pytest Regression Suite on Tesla T4 ==="
python3 -m pytest -v \
  tests/test_m83_knowledge_consolidation.py \
  tests/test_m82_meta_acquisition.py \
  tests/test_m81_learning_to_learn.py \
  tests/test_m80_neural_skill_acquisition.py \
  tests/test_m79_active_experimentation.py \
  tests/test_m78_reliable_plan_discovery.py \
  tests/test_m77_neural_plan_discovery.py \
  tests/test_m76_neural_skill_composition.py \
  tests/test_m75_experience_guided_learning.py \
  tests/test_m74_neural_weight_learning.py \
  tests/test_m73_causal_neural_memory.py \
  tests/test_m72_robust_lifelong_memory.py \
  tests/test_m71_selective_neural_memory.py \
  tests/test_m70_continual_self_improvement.py \
  tests/test_m70_data_pipeline.py \
  tests/test_m69_budgeted_autonomous_learning.py \
  tests/test_m68_autonomous_research_loop.py \
  tests/test_m67_utility_calibrated_selection.py \
  tests/test_m66_modular_skill_learning.py \
  tests/test_m65_three_valued_reasoning.py \
  tests/test_m64_three_class_meta.py \
  tests/test_m63_stable_few_shot.py \
  tests/test_m62_neural_meta_learning.py \
  tests/test_m61_safe_meta_learning.py \
  tests/test_m60_causal_meta_learning.py \
  tests/test_m59_generalizable_learning.py \
  tests/test_m58_adaptive_optimization.py \
  tests/test_m57_capacity_evolution.py \
  tests/test_m56_reliable_transfer.py \
  tests/test_m55_skill_composition.py \
  tests/test_m54_skill_reuse.py \
  tests/test_m53_causal_learning.py \
  tests/test_m52_learning_driven_experiment.py \
  tests/test_m51_guided_experiment.py \
  tests/test_m50_experiment_design.py \
  tests/test_m49_active_learning.py \
  tests/test_m48_rule_induction.py \
  tests/test_m47_stable_relational.py \
  tests/test_m46_relational_representations.py \
  tests/test_m45_epistemic_uncertainty.py \
  tests/test_m44_structural_analogy.py \
  tests/test_m43_selective_memory.py \
  tests/test_m42_learning_to_learn.py \
  tests/test_m41_directional_reasoning.py \
  tests/test_m40_compositional_knowledge.py \
  tests/test_m39_knowledge_integration.py \
  tests/test_m38_structural_evolution.py \
  tests/test_m37_independent_research.py \
  tests/test_m36_curriculum_discovery.py \
  tests/test_m35_metacognitive_policy.py \
  tests/test_m34_modular_continual_learning.py \
  tests/test_m33_cumulative_learning.py \
  tests/test_m32_recurrent_reasoning.py \
  tests/test_m31_intermediate_reasoning.py \
  tests/test_m30_fast_adaptation.py \
  tests/test_m29_closed_loop.py \
  tests/test_m28_language_foundation.py \
  tests/test_m27_learning_quality.py \
  tests/test_m26_curriculum.py \
  tests/test_m25_multi_domain.py

# 5. Run M83 Benchmark
echo "=== Running M83 Knowledge Consolidation Benchmark on Tesla T4 ==="
mkdir -p /content/m83_output
python3 scripts/run_m83_knowledge_consolidation_benchmark.py \
  --seeds 42 43 44 45 46 \
  --checkpoint-dir /content/checkpoints \
  --output-dir /content/m83_output \
  --device cuda

echo "=== Milestone M83 Job Complete ==="
