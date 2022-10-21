#!/bin/bash

# general configuration of the job
#SBATCH --job-name=HorTest
#SBATCH -D .
#SBATCH --qos=bsc_case
#SBATCH --time=03:00:00
#SBATCH --output=job.out
#SBATCH --error=job.err

# configure node and process count on the CM
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=2
#SBATCH --exclusive

# gres options
#SBATCH --gres=gpu:2

# parameters
debug=false # do nccl debug
exp=false   # turn on experimental features 
bs=2        # batch-size
epochs=1    # epochs
lr=0.001    # learning rate

# MNIST
#dataDir='/gpfs/scratch/bsc21/bsc21163/RAISE_Dataset/data_MNIST/'
#COMMAND="Hor_pytorch_mnist.py --concM 100"

# ATBL - small
#dataDir='/gpfs/scratch/bsc21/bsc21163/RAISE_Dataset/T31/'
#COMMAND="Hor_pytorch_AT.py"

# ATBL - large
dataDir='/gpfs/scratch/bsc21/bsc21163/RAISE_Dataset/T31_LD/'
COMMAND="Hor_pytorch_AT_LD_mod.py"

EXEC=$COMMAND" --batch-size $bs \
  --epochs $epochs \
  --lr $lr \
  --concM $cm \
  --use-adasum \
  --shuff \
  --nworker $SLURM_CPUS_PER_TASK \
  --data-dir $dataDir"

# set modules
ml bsc gcc/10.2.0 openmpi/4.0.5 rocm/5.1.1 python/3.9.1

# set env
source /gpfs/projects/bsc21/bsc21163/envAI_BSC/bin/activate
export LD_LIBRARY_PATH=/gpfs/projects/bsc21/bsc21163/envAI_BSC/lib:$LD_LIBRARY_PATH

# sleep a sec
sleep 1

# job info 
echo "DEBUG: TIME: $(date)"
echo "DEBUG: EXECUTE: $EXEC"
echo "DEBUG: SLURM_JOB_ID: $SLURM_JOB_ID"
echo "DEBUG: SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_NNODES: $SLURM_NNODES"
echo "DEBUG: SLURM_NTASKS: $SLURM_NTASKS"
echo "DEBUG: SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "DEBUG: SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "DEBUG: SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "DEBUG: SLURMD_NODENAME: $SLURMD_NODENAME"
echo "DEBUG: SLURM_NODEID: $SLURM_NODEID"
echo "DEBUG: CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
if [ "$debug" = true ] ; then
  export NCCL_DEBUG=INFO
fi
if [ "$exp" = true ] ; then
  export MIOPEN_FIND_ENFORCE=4
  export MIOPEN_DISABLE_CACHE=1 
  export MIOPEN_ENABLE_LOGGING=1 
  export MIOPEN_LOG_LEVEL=7
  export MIOPEN_ENABLE_LOGGING_CMD=1
  export MIOPEN_FIND_MODE=1
fi
echo
export MIOPEN_DEBUG_DISABLE_FIND_DB=1

# launch
srun --cpu-bind=none python -u $EXEC

# eof
