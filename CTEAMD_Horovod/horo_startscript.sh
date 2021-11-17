#!/bin/bash

# general configuration of the job
#SBATCH --job-name=HT
#SBATCH -D .
#SBATCH --qos=bsc_case
#SBATCH --time=01:00:00
#SBATCH --output=job.out
#SBATCH --error=job.err

# configure node and process count on the CM
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

# gres options
#SBATCH --gres=gpu:2

# command to exec
debug=false # do nccl debug
bs=32       # batch-size
epochs=1    # epochs
cm=10       # data-size (concat dataset for MNIST)

# MNIST
dataDir='/gpfs/projects/bsc21/bsc21163/data_MNIST/'
COMMAND="horo_pytorch_mnist.py"

#dataDir='/gpfs/projects/bsc21/bsc21163/T31/'
#COMMAND="horo_pytorch_AT.py"

EXEC=$COMMAND" --batch-size $bs 
  --epochs $epochs
  --concM $cm
  --data-dir $dataDir"

# debug? 
debug=false

# set modules
ml gcc openmpi rocm python

# set env
source /gpfs/projects/bsc21/bsc21163/envAI_BSC/bin/activate
export LD_LIBRARY_PATH=/gpfs/projects/bsc21/bsc21163/envAI_BSC/lib:$LD_LIBRARY_PATH

# sleep a sec
sleep 1

# job info 
echo "COMMAND: $COMMAND"
echo "DEBUG: SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_NNODES: $SLURM_NNODES"
echo "DEBUG: SLURM_NTASKS: $SLURM_NTASKS"
echo "DEBUG: SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "DEBUG: SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "DEBUG: SLURMD_NODENAME: $SLURMD_NODENAME"
echo "DEBUG: SLURM_NODEID: $SLURM_NODEID"
echo "DEBUG: SLURM_LOCALID: $SLURM_LOCALID" 
echo "DEBUG: SLURM_PROCID: $SLURM_PROCID"
echo "DEBUG: CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
if [ "$debug" = true ] ; then
  export NCCL_DEBUG=INFO
fi
echo

# fix ROCm bug
export MIOPEN_DEBUG_DISABLE_FIND_DB=1

# launch
srun python -u $EXEC

# eof
