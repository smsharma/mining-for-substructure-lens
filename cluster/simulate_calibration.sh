#!/bin/bash

#SBATCH --job-name=sim-cal
#SBATCH --output=log_simulate_calibration_%a.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=5-00:00:00
# #SBATCH --gres=gpu:1

source activate lensing
cd /scratch/jb6504/StrongLensing-Inference/

filename=calibrate_fix_theta_${SLURM_ARRAY_TASK_ID}
if test -f "/scratch/jb6504/StrongLensing-Inference/data/samples/x_$filename.npy"; then
    echo "$filename exists, skipping simulation"
else
    python -u simulate.py --calibrate --theta ${SLURM_ARRAY_TASK_ID} --fixz --fixm --fixalign -n 10000 --name $filename --dir /scratch/jb6504/StrongLensing-Inference
fi

filename=calibrate_mass_theta_${SLURM_ARRAY_TASK_ID}
if test -f "/scratch/jb6504/StrongLensing-Inference/data/samples/x_$filename.npy"; then
    echo "$filename exists, skipping simulation"
else
    python -u simulate.py --calibrate --theta ${SLURM_ARRAY_TASK_ID} --fixz --fixalign -n 10000 --name $filename --dir /scratch/jb6504/StrongLensing-Inference
fi

filename=calibrate_align_theta_${SLURM_ARRAY_TASK_ID}
if test -f "/scratch/jb6504/StrongLensing-Inference/data/samples/x_$filename.npy"; then
    echo "$filename exists, skipping simulation"
else
    python -u simulate.py --calibrate --theta ${SLURM_ARRAY_TASK_ID} --fixz --fixm -n 10000 --name $filename --dir /scratch/jb6504/StrongLensing-Inference
fi

filename=calibrate_full_theta_${SLURM_ARRAY_TASK_ID}
if test -f "/scratch/jb6504/StrongLensing-Inference/data/samples/x_$filename.npy"; then
    echo "$filename exists, skipping simulation"
else
    python -u simulate.py --calibrate --theta ${SLURM_ARRAY_TASK_ID} -n 10000 --name $filename --dir /scratch/jb6504/StrongLensing-Inference
fi
