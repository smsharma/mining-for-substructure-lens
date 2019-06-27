#!/bin/bash

#SBATCH --job-name=tr-a-f
#SBATCH --output=log_train_alices_full.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u train.py alices train_full alices_full --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alices train_full alices_full_aux -z --dir /scratch/jb6504/StrongLensing-Inference

python -u train.py alices train_full alices_full_pre --load alices_fix --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alices train_full alices_full_pre_aux --load alices_fix --dir /scratch/jb6504/StrongLensing-Inference
