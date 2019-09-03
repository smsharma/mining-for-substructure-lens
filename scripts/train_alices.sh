#!/bin/bash

#SBATCH --job-name=slr-t-a
#SBATCH --output=log_train_alices.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
base=/scratch/jb6504/recycling_strong_lensing/
cd $base/

python -u train.py alices train_fix alices_fix --dir $base
python -u train.py alices train_mass alices_mass --load alices_fix --dir $base
python -u train.py alices train_align alices_align --load alices_fix --dir $base
python -u train.py alices train_full alices_full --load alices_fix --dir $base
