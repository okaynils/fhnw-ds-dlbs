## Submitting a new run to slurm:
`sbatch --job-name=unet_overfit_1 --export=experiment_name=unet_overfit_1 --output=logs/unet_overfit_1_%j.out --error=logs/unet_overfit_1_%j.err train.slurm`