#!/bin/bash

#SBATCH --job-name=e-c-f
#SBATCH --output=log_eval_carl_full.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=7-00:00:00
#SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

# What to do
tag=full
modeltag=full
echo ""
echo ""
echo ""
echo "EVALUATING PRIOR SAMPLE"
echo ""
python -u test.py carl_${modeltag} test_${tag}_prior carl_${modeltag}_prior --dir /scratch/jb6504/StrongLensing-Inference

echo ""
echo ""
echo ""
echo "EVALUATING PRIOR SAMPLE (SHUFFLED)"
echo ""
python -u test.py carl_${modeltag} test_${tag}_prior carl_${modeltag}_shuffledprior --shuffle --dir /scratch/jb6504/StrongLensing-Inference

echo ""
echo ""
echo ""
echo "EVALUATING POINT SAMPLE ON PARAM GRID"
echo ""
python -u test.py carl_${modeltag} test_${tag}_point carl_${modeltag}_grid --grid --dir /scratch/jb6504/StrongLensing-Inference

echo ""
echo ""
echo ""
echo "EVALUATING POINT SAMPLE ON PARAM GRID (FINE)"
echo ""
python -u test.py carl_${modeltag} test_${tag}_point carl_${modeltag}_finegrid --grid --fine --dir /scratch/jb6504/StrongLensing-Inference


modeltag=full_aux
echo ""
echo ""
echo ""
echo "EVALUATING PRIOR SAMPLE"
echo ""
python -u test.py carl_${modeltag} test_${tag}_prior carl_${modeltag}_prior -z --dir /scratch/jb6504/StrongLensing-Inference

echo ""
echo ""
echo ""
echo "EVALUATING PRIOR SAMPLE (SHUFFLED)"
echo ""
python -u test.py carl_${modeltag} test_${tag}_prior carl_${modeltag}_shuffledprior -z --shuffle --dir /scratch/jb6504/StrongLensing-Inference

echo ""
echo ""
echo ""
echo "EVALUATING POINT SAMPLE ON PARAM GRID"
echo ""
python -u test.py carl_${modeltag} test_${tag}_point carl_${modeltag}_grid -z --grid --dir /scratch/jb6504/StrongLensing-Inference

echo ""
echo ""
echo ""
echo "EVALUATING POINT SAMPLE ON PARAM GRID (FINE)"
echo ""
python -u test.py carl_${modeltag} test_${tag}_point carl_${modeltag}_finegrid -z --grid --fine --dir /scratch/jb6504/StrongLensing-Inference