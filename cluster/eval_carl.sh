#!/bin/bash

#SBATCH --job-name=slr-e-c
#SBATCH --output=log_eval_carl.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
base=/scratch/jb6504/recycling_strong_lensing/
cd $base/

# What to do
for tag in fix mass align full
do
    modeltag=${tag}
    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on prior sample"
    echo ""
    python -u test.py carl_${modeltag} test_${tag}_prior carl_${modeltag}_prior --dir $base

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on shuffled prior sample"
    echo ""
    python -u test.py carl_${modeltag} test_${tag}_prior carl_${modeltag}_shuffledprior --shuffle --dir $base

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on point sample / param grid"
    echo ""
    python -u test.py carl_${modeltag} test_${tag}_point carl_${modeltag}_grid --grid --dir $base

#    echo ""
#    echo ""
#    echo ""
#    echo "Evaluating ${modeltag} on point sample / fine param grid"
#    echo ""
#    python -u test.py carl_${modeltag} test_${tag}_point carl_${modeltag}_finegrid --grid --finegrid --dir $base

done

#for tag in full
#do
#    for i in {0..624}
#    do
#        echo ""
#        echo ""
#        echo ""
#        echo "Evaluating ${modeltag} on calibration sample $i"
#        echo ""
#        python -u test.py carl_${modeltag} calibrate_${tag}_theta$i carl_${modeltag}_calibrate_theta$i --dir $base
#    done
#
#    echo ""
#    echo ""
#    echo ""
#    echo "Evaluating ${modeltag} on reference calibration sample"
#    echo ""
#    python -u test.py carl_${modeltag} calibrate_${tag}_ref carl_${modeltag}_calibrate_ref --dir $base
#
#done
