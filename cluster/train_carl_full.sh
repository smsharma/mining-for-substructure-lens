#!/bin/bash

#SBATCH --job-name=tr-c-f
#SBATCH --output=log_train_carl_full.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u train.py carl train_full carl_full --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py carl train_full carl_full_aux -z --dir /scratch/jb6504/StrongLensing-Inference
