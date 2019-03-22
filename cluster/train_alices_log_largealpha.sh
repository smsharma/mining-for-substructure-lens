#!/bin/bash

#SBATCH --job-name=train_alices_log_alpha_-3
#SBATCH --output=log_train_alices_log_alpha_-3.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u train.py alices --log --alpha 1.e-3 --name alices_log_alpha_-3 --dir /scratch/jb6504/StrongLensing-Inference
