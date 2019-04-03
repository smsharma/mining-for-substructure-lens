#!/bin/bash

#SBATCH --job-name=e-a
#SBATCH --output=log_eval_alice.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u test.py alice test alice_grid --grid --dir /scratch/jb6504/StrongLensing-Inference
# python -u test.py alice test_prior alice_prior --dir /scratch/jb6504/StrongLensing-Inference
python -u test.py alice test_prior alice_shuffledprior --shuffled --dir /scratch/jb6504/StrongLensing-Inference
