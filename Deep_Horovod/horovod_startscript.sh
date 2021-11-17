#!/usr/bin/env bash

# slurm job configuration
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:15:00
#SBATCH --job-name=test
#SBATCH --gres=gpu:1 
#SBATCH --partition=dp-esb-ib

# load modules
module --force purge
module use $OTHERSTAGES 
ml Stages/2020 GCC ParaStationMPI/5.4.7-1-mt Python

# source env
source /p/project/prcoe12/RAISE/envAI_deepv/bin/activate

# cuda flags
export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export CUDA_LAUNCH_BLOCKING=1 # blocks launch, no need at this moment

# execute
srun --cpu-bind=none python -u pytorch_mnist.py

# eof
