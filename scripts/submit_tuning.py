import os
import subprocess
import time
from pathlib import Path

experiments_folder = "configs/experiment/tune"

max_concurrent_jobs = 2

train_script = "train.py"

def count_running_jobs():
    result = subprocess.run(["squeue", "-u", os.getenv("USER")], capture_output=True, text=True)
    return len(result.stdout.strip().split("\n")) - 1

def submit_job(config_name):
    job_command = f"python3 {train_script} experiment=tune/{config_name}"
    slurm_script = f"#!/bin/bash\n#SBATCH -p performance\n#SBATCH --job-name=dlbs_{config_name}\n#SBATCH --gpus=1\n#SBATCH --output=output/{config_name}_%j.out\n#SBATCH --error=output/{config_name}_%j.err\n#SBATCH --time=02:00:00\n\nsource ./venv/bin/activate\n{job_command}"

    with open("scripts/submit_job.sh", "w") as f:
        f.write(slurm_script)

    subprocess.run(["sbatch", "scripts/submit_job.sh"])
    os.remove("scripts/submit_job.sh")

experiment_configs = [f.name for f in Path(experiments_folder).glob("*.yaml")]

for config in experiment_configs:
    while count_running_jobs() >= max_concurrent_jobs:
        print("Maximum jobs running. Waiting...")
        time.sleep(30)
    print(f"Submitting job for config: {config}")
    submit_job(config)

print("All jobs submitted.")