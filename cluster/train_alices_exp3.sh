#!/bin/bash

#SBATCH --job-name=tr-a-e3
#SBATCH --output=log_train_alices_exp3.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u train.py alices train_fix alices_fix_log --log --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alices train_mass alices_mass_log --log --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alices train_align alices_align_log --log --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alices train_full alices_full_log --log --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alices train_full alices_full_log_aux --log -z --dir /scratch/jb6504/StrongLensing-Inference
