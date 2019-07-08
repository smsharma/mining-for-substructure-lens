#!/bin/bash

#SBATCH --job-name=e-c
#SBATCH --output=log_eval_carl.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

# What to do
for tag in fix mass align full
do
    if [ "tag" = "fix" ]; then
        modeltag=${tag}
    else
        modeltag=${tag}_pre
    fi

    for i in {0..624}
    do
        echo ""
        echo ""
        echo ""
        echo "Evaluating ${modeltag} on calibration sample $i"
        echo ""
        python -u test.py carl_${modeltag} calibrate_${tag}_theta$i carl_${modeltag}_calibrate_theta$i --dir /scratch/jb6504/StrongLensing-Inference
    done

    echo ""
    echo ""
    echo ""
        echo "Evaluating ${modeltag} on reference calibration sample"
    echo ""
    python -u test.py carl_${modeltag} calibrate_${tag}_ref carl_${modeltag}_calibrate_ref --dir /scratch/jb6504/StrongLensing-Inference

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on big-prior sample"
    echo ""
    # python -u test.py carl_${modeltag} test_${tag}_prior carl_${modeltag}_prior --dir /scratch/jb6504/StrongLensing-Inference

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on shuffled big-prior sample"
    echo ""
    # python -u test.py carl_${modeltag} test_${tag}_prior carl_${modeltag}_shuffledprior --shuffle --dir /scratch/jb6504/StrongLensing-Inference

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on point sample / param grid"
    echo ""
    # python -u test.py carl_${modeltag} test_${tag}_point carl_${modeltag}_grid --grid --dir /scratch/jb6504/StrongLensing-Inference

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on point sample / fine param grid"
    echo ""
    # python -u test.py carl_${modeltag} test_${tag}_point carl_${modeltag}_finegrid --grid --finegrid --dir /scratch/jb6504/StrongLensing-Inference

done
