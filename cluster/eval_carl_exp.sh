#!/bin/bash

#SBATCH --job-name=e-c-e
#SBATCH --output=log_eval_carl_exp.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

for tag in fix mass align
do
    modeltag=${tag}_log

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on prior sample"
    echo ""
    python -u test.py carl_${modeltag} test_${tag}_prior carl_${modeltag}_prior --dir /scratch/jb6504/StrongLensing-Inference

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on shuffled prior sample"
    echo ""
    python -u test.py carl_${modeltag} test_${tag}_prior carl_${modeltag}_shuffledprior --shuffle --dir /scratch/jb6504/StrongLensing-Inference

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on point sample / param grid"
    echo ""
    python -u test.py carl_${modeltag} test_${tag}_point carl_${modeltag}_grid --grid --dir /scratch/jb6504/StrongLensing-Inference

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on point sample / fine param grid"
    echo ""
    python -u test.py carl_${modeltag} test_${tag}_point carl_${modeltag}_finegrid --grid --fine --dir /scratch/jb6504/StrongLensing-Inference

done

for tag in mass align
do
    modeltag=${tag}_pre

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on prior sample"
    echo ""
    python -u test.py carl_${modeltag} test_${tag}_prior carl_${modeltag}_prior --dir /scratch/jb6504/StrongLensing-Inference

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on shuffled prior sample"
    echo ""
    python -u test.py carl_${modeltag} test_${tag}_prior carl_${modeltag}_shuffledprior --shuffle --dir /scratch/jb6504/StrongLensing-Inference

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on point sample / param grid"
    echo ""
    python -u test.py carl_${modeltag} test_${tag}_point carl_${modeltag}_grid --grid --dir /scratch/jb6504/StrongLensing-Inference

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on point sample / fine param grid"
    echo ""
    python -u test.py carl_${modeltag} test_${tag}_point carl_${modeltag}_finegrid --grid --fine --dir /scratch/jb6504/StrongLensing-Inference

done
