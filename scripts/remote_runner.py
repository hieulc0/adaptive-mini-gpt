import subprocess
import sys

print(">>> Starting Remote M88 Execution on Tesla T4 GPU <<<", flush=True)
res = subprocess.run(["bash", "/content/run_job.sh"], check=False)
print(f">>> Remote M88 Execution Finished with returncode {res.returncode} <<<", flush=True)
sys.exit(res.returncode)
