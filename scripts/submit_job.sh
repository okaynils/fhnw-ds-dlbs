#!/bin/bash

# Usage: ./submit_jobs.sh <experiment_name>
# Example: ./submit_jobs.sh unet_overfit_1

experiment=$1
time=$2

if [ -z "$experiment" ]; then
  echo "Error: No experiment name provided."
  echo "Usage: $0 <experiment_name>"
  exit 1
fi

# Submit the job to Slurm
sbatch \
  --time="$time" \
  --partition=performance \
  --gpus=1 \
  --exclusive \
  --mem=18G \
  --job-name="dlbs_${experiment}" \
  --output="output/${experiment}_%j.out" \
  --error="output/${experiment}_%j.err" \
  --wrap="source ./venv/bin/activate; python3 train.py experiment=${experiment}"