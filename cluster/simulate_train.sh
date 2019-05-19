#!/bin/bash

#SBATCH --job-name=sim-tr
#SBATCH --output=log_simulate_train_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=1-00:00:00
# #SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u simulate.py -n 1000 --name train_${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/StrongLensing-Inference
