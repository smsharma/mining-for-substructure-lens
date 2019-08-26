#!/bin/bash

#SBATCH --job-name=slr-c
#SBATCH --output=log_calibrate.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate lensing
base=/scratch/jb6504/recycling_strong_lensing/
cd $base

for tag in full
do
    if [ "tag" = "fix" ]; then
        modeltag=${tag}
    else
        modeltag=${tag}_pre
    fi
    python -u calibrate.py carl_${modeltag}_grid carl_calibrate_${tag} --dir $base
    python -u calibrate.py alices_${modeltag}_grid carl_calibrate_${tag} --dir $base
done
