#!/bin/bash

#SBATCH --job-name=as-ad
#SBATCH --output=log_train_alices_aux_deep.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

python -u train.py alices --aux z --alpha 0.1 --name alices_aux_deep --deep --dir /scratch/jb6504/StrongLensing-Inference
