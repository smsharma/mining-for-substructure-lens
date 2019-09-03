#!/bin/bash

#SBATCH --job-name=slr-e-alpha
#SBATCH --output=log_eval_alpha.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
base=/scratch/jb6504/recycling_strong_lensing/
cd $base

tag=full
for variation in alpha2e2 alpha2e3 alpha2e5 alpha2e6
do
    modeltag=${tag}_${variation}
    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on prior sample"
    echo ""
    python -u test.py alices_${modeltag} test_${tag}_prior alices_${modeltag}_prior --dir $base

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on shuffled prior sample"
    echo ""
    python -u test.py alices_${modeltag} test_${tag}_prior alices_${modeltag}_shuffledprior --shuffle --dir $base

    echo ""
    echo ""
    echo ""
    echo "Evaluating ${modeltag} on point sample / param grid"
    echo ""
    python -u test.py alices_${modeltag} test_${tag}_point alices_${modeltag}_grid --grid --dir $base

done
