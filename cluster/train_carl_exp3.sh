#!/bin/bash

#SBATCH --job-name=tr-c-e3
#SBATCH --output=log_train_carl_exp3.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u train.py carl train_fix carl_fix_log --log --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py carl train_mass carl_mass_log --log --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py carl train_align carl_align_log --log --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py carl train_full carl_full_log --log --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py carl train_full carl_full_log_aux --log -z --dir /scratch/jb6504/StrongLensing-Inference
