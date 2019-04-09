#!/bin/bash

#SBATCH --job-name=sim_test
#SBATCH --output=log_simulate_test_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u simulate.py -n 10000 --name test${SLURM_ARRAY_TASK_ID} --test --point --dir /scratch/jb6504/StrongLensing-Inference
