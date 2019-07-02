#!/bin/bash

#SBATCH --job-name=tr-a-m
#SBATCH --output=log_train_alices_monster.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u train.py alices train_fix alices_fix_monster --log --epochs 200 --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alices train_mass alices_mass_monster --log --epochs 200 --load alices_fix_monster --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alices train_align alices_align_monster --log --epochs 200 --load alices_fix_monster --dir /scratch/jb6504/StrongLensing-Inference
