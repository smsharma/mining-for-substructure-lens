#!/bin/bash

#SBATCH --job-name=train_alices_alpha_-4
#SBATCH --output=log_train_alices_alpha_-4.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u train.py alices --alpha 1.e-4 --name alices_alpha_-4 --dir /scratch/jb6504/StrongLensing-Inference
