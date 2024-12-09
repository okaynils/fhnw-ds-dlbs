#!/bin/bash

job1_id=$(sbatch --export=experiment_name=unet_overfit_1 train.slurm | awk '{print $4}')
echo "Submitted job 1 (unet_overfit_1) with ID: $job1_id"

job2_id=$(sbatch --export=experiment_name=unet_overfit_2 train.slurm | awk '{print $4}')
echo "Submitted job 2 (unet_overfit_2) with ID: $job2_id"
