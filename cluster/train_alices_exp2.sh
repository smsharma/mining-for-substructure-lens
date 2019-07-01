#!/bin/bash

#SBATCH --job-name=tr-a-e2
#SBATCH --output=log_train_alices_exp2.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u train.py alices train_fix alices_fix_sgd --optimizer sgd --lr 0.001 --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alices train_mass alices_mass_sgd --optimizer sgd --lr 0.001 --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alices train_align alices_align_sgd --optimizer sgd --lr 0.001 --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alices train_full alices_full_sgd --optimizer sgd --lr 0.001 --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alices train_full alices_full_sgd_aux --optimizer sgd --lr 0.001 -z --dir /scratch/jb6504/StrongLensing-Inference
