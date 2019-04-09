#!/bin/bash

#SBATCH --job-name=c
#SBATCH --output=log_train_carl.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u train.py carl --name carl --dir /scratch/jb6504/StrongLensing-Inference
