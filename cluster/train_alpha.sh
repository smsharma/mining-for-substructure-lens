#!/bin/bash

#SBATCH --job-name=slr-t-a
#SBATCH --output=log_train_alpha.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
base=/scratch/jb6504/recycling_strong_lensing/
cd $base/

python -u train.py alices train_full alices_full_alpha2e2 --load alices_fix --alpha 2.e-2 --dir $base
python -u train.py alices train_full alices_full_alpha2e3 --load alices_fix --alpha 2.e-3 --dir $base
python -u train.py alices train_full alices_full_alpha2e5 --load alices_fix --alpha 2.e-5 --dir $base
python -u train.py alices train_full alices_full_alpha2e6 --load alices_fix --alpha 2.e-6 --dir $base
