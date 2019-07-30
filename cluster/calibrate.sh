#!/bin/bash

#SBATCH --job-name=cal
#SBATCH --output=log_calibrate.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

# DIR=/Users/johannbrehmer/work/projects/strong_lensing/StrongLensing-Inference
DIR=/scratch/jb6504/StrongLensing-Inference/

source activate lensing
cd $DIR

for tag in fix align full mass
do
    if [ "$tag" = "fix" ]; then
        modeltag=${tag}
    else
        modeltag=${tag}_pre
    fi
    python -u calibrate.py carl_${modeltag}_grid carl_calibrate_${tag} --dir $DIR
    python -u calibrate.py alices_${modeltag}_grid carl_calibrate_${tag} --dir $DIR
done
