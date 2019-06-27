#!/bin/bash

#SBATCH --job-name=e-a-a
#SBATCH --output=log_eval_alice_aux.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

# What to do
for tag in full full_pre
do
    modeltag=${tag}_aux

    echo ""
    echo ""
    echo ""
    echo "EVALUATING PRIOR SAMPLE"
    echo ""
    python -u test.py alice_${modeltag} test_${tag}_prior alice_${modeltag}_prior -z --dir /scratch/jb6504/StrongLensing-Inference

    echo ""
    echo ""
    echo ""
    echo "EVALUATING PRIOR SAMPLE (SHUFFLED)"
    echo ""
    python -u test.py alice_${modeltag} test_${tag}_prior alice_${modeltag}_shuffledprior -z --shuffle --dir /scratch/jb6504/StrongLensing-Inference

    echo ""
    echo ""
    echo ""
    echo "EVALUATING POINT SAMPLE ON PARAM GRID"
    echo ""
    python -u test.py alice_${modeltag} test_${tag}_point alice_${modeltag}_grid -z --grid --dir /scratch/jb6504/StrongLensing-Inference

done
