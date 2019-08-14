#!/bin/bash

#SBATCH --job-name=slr-t-c
#SBATCH --output=log_train_carl.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
base=/scratch/jb6504/recycling_strong_lensing/
cd $base/

python -u train.py carl train_fix carl_fix --dir $base-Inference
python -u train.py carl train_mass carl_mass --load carl_fix --dir $base
python -u train.py carl train_align carl_align --load carl_fix --dir $base
python -u train.py carl train_full carl_full --load carl_fix --dir $base
