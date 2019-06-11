#!/bin/bash

#SBATCH --job-name=tr-c
#SBATCH --output=log_train_carl.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u train.py carl train_fix carl_fix --dir /scratch/jb6504/StrongLensing-Inference
# python -u train.py carl train_mass carl_mass --dir /scratch/jb6504/StrongLensing-Inference
# python -u train.py carl train_align carl_align --dir /scratch/jb6504/StrongLensing-Inference
# python -u train.py carl train_full carl_full --dir /scratch/jb6504/StrongLensing-Inference
# python -u train.py carl train_full carl_full_aux -z --dir /scratch/jb6504/StrongLensing-Inference
