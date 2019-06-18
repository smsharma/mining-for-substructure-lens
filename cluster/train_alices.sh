#!/bin/bash

#SBATCH --job-name=tr-as
#SBATCH --output=log_train_alices.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u train.py alices train_fix alices_fix --alpha 1.e-4  --dir /scratch/jb6504/StrongLensing-Inference
# python -u train.py alices train_mass alices_mass --alpha 1.e-4 --dir /scratch/jb6504/StrongLensing-Inference
python -u train.py alices train_align alices_align --alpha 1.e-4 --dir /scratch/jb6504/StrongLensing-Inference
# python -u train.py alices train_full alices_full --alpha 1.e-4 --dir /scratch/jb6504/StrongLensing-Inference
# python -u train.py alices train_full alices_full_aux -z --alpha 1.e-4 --dir /scratch/jb6504/StrongLensing-Inference
