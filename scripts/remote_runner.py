import subprocess
import sys

print(">>> Starting Remote M87 Execution on Tesla T4 GPU <<<", flush=True)
res = subprocess.run(["bash", "/content/run_job.sh"], check=False)
print(f">>> Remote M87 Execution Finished with returncode {res.returncode} <<<", flush=True)
sys.exit(res.returncode)
