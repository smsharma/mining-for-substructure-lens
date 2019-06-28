#!/bin/bash

#SBATCH --job-name=tr-a-e
#SBATCH --output=log_train_alices_exp.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u train.py alices train_fix alices_fix_deep --deep --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alices train_mass alices_mass_deep --deep --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alices train_align alices_align_deep --deep --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alices train_full alices_full_deep --deep --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alices train_full alices_full_deep_aux --deep -z --dir /scratch/jb6504/StrongLensing-Inference
