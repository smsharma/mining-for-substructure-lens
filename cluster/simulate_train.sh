#!/bin/bash

#SBATCH --job-name=sim_train
#SBATCH --output=log_simulate_train_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=2-00:00:00
# #SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u simulate.py -n 10000 --name train${SLURM_ARRAY_TASK_ID} --dir /scratch/jb6504/StrongLensing-Inference
