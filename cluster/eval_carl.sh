#!/bin/bash

#SBATCH --job-name=e-c
#SBATCH --output=log_eval_carl.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u test.py carl test carl_grid --grid --dir /scratch/jb6504/StrongLensing-Inference
python -u test.py carl test_prior carl_prior --dir /scratch/jb6504/StrongLensing-Inference
