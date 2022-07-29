#!/bin/bash

#SBATCH --job-name=LibTorchTest
#SBATCH --account=raise-ctp1
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:15:00
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --gres=gpu:1

ml NVHPC/22.3 cuDNN CMake

echo "DEBUG: $(date)"
echo "DEBUG: SLURM_JOB_ID: $SLURM_JOB_ID"
echo "DEBUG: SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

export CUDA_VISIBLE_DEVICES="0"
export OMP_NUM_THREADS=1

srun ./mnist
