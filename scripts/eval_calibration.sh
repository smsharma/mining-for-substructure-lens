#!/bin/bash

#SBATCH --job-name=slr-e-c
#SBATCH --output=log_eval_calibration.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
base=/scratch/jb6504/recycling_strong_lensing/
cd $base

tag=full
for modeltag in full full_sgd1e2
do
    for i in {0..624}
    do
        echo ""
        echo ""
        echo ""
        echo "Evaluating ${modeltag} on calibration sample $i"
        echo ""
        python -u test.py alices_${modeltag} calibrate_${tag}_theta_$i alices_${modeltag}_calibrate_theta_$i --dir $base --igrid $i
    done
done
