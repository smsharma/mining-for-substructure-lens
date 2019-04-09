#!/bin/bash

#SBATCH --job-name=e-as
#SBATCH --output=log_eval_alices.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

# python -u test.py alices test_prior alices_prior --dir /scratch/jb6504/StrongLensing-Inference
python -u test.py alices test_prior alices_shuffledprior --shuffle --dir /scratch/jb6504/StrongLensing-Inference
python -u test.py alices test alices_grid --grid --dir /scratch/jb6504/StrongLensing-Inference
