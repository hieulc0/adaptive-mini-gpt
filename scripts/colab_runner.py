import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path("/tmp/adaptive-mini-gpt")
if REPO.exists():
    shutil.rmtree(REPO)

print("=== Cloning latest repository branch on Colab T4 ===", flush=True)
subprocess.run([
    "git", "clone", "--depth", "1", "--branch", "m103-g1-m104a-autonomous-concept-learning",
    "https://github.com/hieulc0/adaptive-mini-gpt.git", str(REPO)
], check=True)

env = dict(os.environ)
env["PYTHONPATH"] = f"{REPO / 'src'}:{env.get('PYTHONPATH', '')}".rstrip(":")

# 1. GPU Feasibility Unit Test
print("\n=== Running M103-G.1 Unit Tests on Colab T4 ===", flush=True)
subprocess.run([sys.executable, "-m", "pytest", "-v", "tests/test_m103_g1_autonomous_concept_learning.py"], cwd=REPO, env=env, check=True)

# 2. 1-Seed GPU Feasibility Benchmark
print("\n=== Running 1-Seed GPU Feasibility Benchmark (Seed 42) ===", flush=True)
out_dir = Path("/tmp/runs/m103_g1_feasibility")
if out_dir.exists():
    shutil.rmtree(out_dir)

subprocess.run([
    sys.executable, "scripts/run_m103_g1_benchmark.py",
    "--seeds", "42",
    "--steps", "120",
    "--output-dir", str(out_dir)
], cwd=REPO, env=env, check=True)

# Print markdown summary
rep_md = out_dir / "m103_g1_report.md"
if rep_md.exists():
    print("\n=== 1-Seed Feasibility Report ===\n", flush=True)
    print(rep_md.read_text(), flush=True)
