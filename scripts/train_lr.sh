#!/bin/bash

#SBATCH --job-name=slr-t-lr
#SBATCH --output=log_train_lr.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
base=/scratch/jb6504/recycling_strong_lensing/
cd $base/

python -u train.py alices train_full alices_full_lr1e3 --load alices_fix --lr 1.e-3 --dir $base
python -u train.py alices train_full alices_full_lr3e4  --load alices_fix --lr 3.e-4 --dir $base
python -u train.py alices train_full alices_full_lr3e5  --load alices_fix --lr 3.e-5 --dir $base
python -u train.py alices train_full alices_full_lr1e5  --load alices_fix --lr 1.e-5 --dir $base
