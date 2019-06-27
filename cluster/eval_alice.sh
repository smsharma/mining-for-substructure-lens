#!/bin/bash

#SBATCH --job-name=e-a
#SBATCH --output=log_eval_alice.log
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
    modeltag=${tag}

    # for i in {0..624}
    # do
    #     echo ""
    #     echo ""
    #     echo ""
    #     echo "EVALUATING CALIB $i"
    #     echo ""
    #     python -u test.py alice_${modeltag} calibrate_${tag}_theta$i alice_${modeltag}_calibrate_theta$i --dir /scratch/jb6504/StrongLensing-Inference
    # done
    #
    # echo ""
    # echo ""
    # echo ""
    # echo "EVALUATING CALIB REF"
    # echo ""
    # # python -u test.py alice_${modeltag} calibrate_${tag}_ref alice_${modeltag}_calibrate_ref --dir /scratch/jb6504/StrongLensing-Inference

    echo ""
    echo ""
    echo ""
    echo "EVALUATING PRIOR SAMPLE"
    echo ""
    python -u test.py alice_${modeltag} test_${tag}_prior alice_${modeltag}_prior --dir /scratch/jb6504/StrongLensing-Inference

    echo ""
    echo ""
    echo ""
    echo "EVALUATING PRIOR SAMPLE (SHUFFLED)"
    echo ""
    python -u test.py alice_${modeltag} test_${tag}_prior alice_${modeltag}_shuffledprior --shuffle --dir /scratch/jb6504/StrongLensing-Inference

    echo ""
    echo ""
    echo ""
    echo "EVALUATING POINT SAMPLE ON PARAM GRID"
    echo ""
    python -u test.py alice_${modeltag} test_${tag}_point alice_${modeltag}_grid --grid --dir /scratch/jb6504/StrongLensing-Inference

done