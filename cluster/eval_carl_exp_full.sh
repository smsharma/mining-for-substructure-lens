#!/bin/bash

#SBATCH --job-name=e-c-fe
#SBATCH --output=log_eval_carl_full_exp.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

tag=full
for modeltag in full_log full_pre full_log_aux
do:

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
