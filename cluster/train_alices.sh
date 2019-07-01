#!/bin/bash

#SBATCH --job-name=tr-a
#SBATCH --output=log_train_alices.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

# python -u train.py alices train_fix alices_fix --dir /scratch/jb6504/StrongLensing-Inference
# python -u train.py alices train_mass alices_mass --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alices train_align alices_align --dir /scratch/jb6504/StrongLensing-Inference
