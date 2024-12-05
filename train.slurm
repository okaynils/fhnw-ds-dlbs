#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --partition=performance
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

source ./venv/bin/activate

python3 train.py experiment=${experiment_name}
