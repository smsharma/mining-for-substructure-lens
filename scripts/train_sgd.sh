#!/bin/bash

#SBATCH --job-name=slr-t-sgd
#SBATCH --output=log_train_sgd.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
base=/scratch/jb6504/recycling_strong_lensing/
cd $base/

python -u train.py alices train_full alices_full_sgd1e1 --load alices_fix --optimizer sgd --lr 0.1 --dir $base
python -u train.py alices train_full alices_full_sgd1e2 --load alices_fix --optimizer sgd --lr 0.01 --dir $base
