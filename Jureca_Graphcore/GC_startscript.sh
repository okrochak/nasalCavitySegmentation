#!/bin/bash

# general configuration of the job
#SBATCH --job-name=GC_test
#SBATCH --account=zam
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=01:00:00

# configure node and process count on the CM
#SBATCH --partition=dc-ipu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive

srun apptainer run pytorch.sif -- python3 \
        ./GC_pytorch_mnist.py \
        --data-dir /p/scratch/raise-ctp1/data_MNIST/ \
        --nworker $SLURM_CPUS_PER_TASK \
        --concM 100

# eof
