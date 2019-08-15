#!/bin/bash

#SBATCH --job-name=tr-a2
#SBATCH --output=log_train_alice2.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

# python -u train.py alice train_fix alice_fix --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alice train_mass alice_mass --dir /scratch/jb6504/StrongLensing-Inference
# python -u train.py alice train_align alice_align --dir /scratch/jb6504/StrongLensing-Inference
# python -u train.py alice train_full alice_full --dir /scratch/jb6504/StrongLensing-Inference
# python -u train.py alice train_full alice_full_aux -z --dir /scratch/jb6504/StrongLensing-Inference

python -u train.py alice train_mass alice_mass_pre --load alice_fix --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alice train_mass alice_mass_zb --zerobias --dir /scratch/jb6504/StrongLensing-Inference