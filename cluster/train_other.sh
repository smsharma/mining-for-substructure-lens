#!/bin/bash

#SBATCH --job-name=slr-t-other
#SBATCH --output=log_train_other2.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
base=/scratch/jb6504/recycling_strong_lensing/
cd $base/

# python -u train.py alices train_full alices_full_fromscratch --dir $base
# python -u train.py alices train_full alices_full_deep --load alices_fix --deep --dir $base
python -u train.py alices train_full alices_full_batchsize64 --load alices_fix --batchsize 64 --dir $base
python -u train.py alices train_full alices_full_batchsize256 --load alices_fix --batchsize 256 --dir $base
