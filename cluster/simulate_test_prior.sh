#!/bin/bash

#SBATCH --job-name=sim-epr
#SBATCH --output=log_simulate_testprior_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u simulate.py -n 10000 --name test_prior${SLURM_ARRAY_TASK_ID} --test --dir /scratch/jb6504/StrongLensing-Inference
