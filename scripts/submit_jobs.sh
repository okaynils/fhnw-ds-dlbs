#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -p performance
#SBATCH --gpus=1
#SBATCH --job-name=unet_overfit_1
#SBATCH --output=unet_overfit_1_%j.out
#SBATCH --error=unet_overfit_1_%j.err

experiment=$1

source ./venv/bin/activate

echo Running Experiment $experiment ...
python3 train.py experiment=$experiment