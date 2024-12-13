#!/bin/bash
#SBATCH -t 02:00:00
#SBATCH -p performance
#SBATCH --gpus=1
#SBATCH --mem 18G
#SBATCH --job-name=dlbs_unet_fl_lr_0.0001
#SBATCH --output=output/unet_fl_lr_0.0001_%j.out
#SBATCH --error=output/unet_fl_lr_0.0001_%j.err

experiment=$1

source ./venv/bin/activate

echo Running Experiment $experiment ...
python3 train.py experiment=$experiment