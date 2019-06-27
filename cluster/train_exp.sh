#!/bin/bash

#SBATCH --job-name=tr-exp
#SBATCH --output=log_train_exp.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

# python -u train.py alice train_mass alice_mass_pre --load alices_fix --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alice train_mass alice_mass_zb --zerobias --dir /scratch/jb6504/StrongLensing-Inference

python -u train.py alice train_align alice_align_pre --load alices_fix --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alice train_align alice_align_zb --zerobias --dir /scratch/jb6504/StrongLensing-Inference
