#!/bin/bash

#SBATCH --job-name=e-c-d
#SBATCH --output=log_eval_carl_deep.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

# python -u test.py carl_deep test_prior carl_deep_prior --dir /scratch/jb6504/StrongLensing-Inference
python -u test.py carl_deep test_prior carl_deep_shuffledprior --shuffle --dir /scratch/jb6504/StrongLensing-Inference
python -u test.py carl_deep test carl_deep_grid --grid --dir /scratch/jb6504/StrongLensing-Inference
