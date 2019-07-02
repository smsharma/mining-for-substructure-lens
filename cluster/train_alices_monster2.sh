#!/bin/bash

#SBATCH --job-name=tr-a-m2
#SBATCH --output=log_train_alices_monster2.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u train.py alices train_full alices_full_monster_aux -z --log --epochs 200 --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alices train_full alices_full_monster --log --epochs 200 --load alices_fix_monster --dir /scratch/jb6504/StrongLensing-Inference
