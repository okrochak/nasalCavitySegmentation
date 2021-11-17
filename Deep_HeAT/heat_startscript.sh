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
ml Stages/2020 GCC ParaStationMPI/5.4.7-1-mt 
ml Python CMake NCCL mpi4py netCDF parallel-netcdf SciPy-Stack

# command to exec
bs=32
epochs=1 
cm=2 
dataDir="/p/project/prcoe12/RAISE/data_MNIST/"
COMMAND="heat_pytorch_mnist.py --batch-size $bs --epochs $epochs --concM $cm --data-dir $dataDir" 
echo "DEBUG: EXECUTE=$COMMAND"

# source env
source /p/project/prcoe12/RAISE/envAI_deepv/bin/activate

# cuda flags
export CUDA_VISIBLE_DEVICES="0,1,2,3"
unset UCX_TLS
export PSP_CUDA=0

# execute
srun python -u $COMMAND 

# eof
