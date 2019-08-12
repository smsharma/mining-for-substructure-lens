#!/bin/bash

#SBATCH --job-name=sim-cal
#SBATCH --output=log_simulate_calibration_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=7-00:00:00
# #SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

# python -u simulate.py --calibrate --theta ${SLURM_ARRAY_TASK_ID} --fixz --fixm --fixalign -n 10000 --name $filename --dir /scratch/jb6504/StrongLensing-Inference
# python -u simulate.py --calibrate --theta ${SLURM_ARRAY_TASK_ID} --fixz --fixalign -n 10000 --name $filename --dir /scratch/jb6504/StrongLensing-Inference
# python -u simulate.py --calibrate --theta ${SLURM_ARRAY_TASK_ID} --fixz --fixm -n 10000 --name $filename --dir /scratch/jb6504/StrongLensing-Inference
python -u simulate.py --calibrate --theta ${SLURM_ARRAY_TASK_ID} -n 10000 --name $filename --dir /scratch/jb6504/StrongLensing-Inference
