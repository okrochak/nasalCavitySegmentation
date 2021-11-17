#!/bin/bash

# general configuration of the job
#SBATCH --job-name=HeAT
#SBATCH -D .
#SBATCH --qos=bsc_case
#SBATCH --time=00:10:00
#SBATCH --output=job.out
#SBATCH --error=job.err

# configure node and process count on the CM
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

# gres options
#SBATCH --gres=gpu:2

# parameters
debug=false
bs=32
epochs=10
cm=1 
dataDir='/gpfs/projects/bsc21/bsc21163/data_MNIST/'
COMMAND="heat_pytorch_mnist.py 
  --batch-size $bs --epochs $epochs --concM $cm --data-dir $dataDir" 

# command to exec
echo "DEBUG: EXECUTE=$COMMAND"

# set modules
ml gcc openmpi rocm python

# set env
source /gpfs/projects/bsc21/bsc21163/envAI_BSC/bin/activate
export LD_LIBRARY_PATH=/gpfs/projects/bsc21/bsc21163/envAI_BSC/lib:$LD_LIBRARY_PATH

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
export MIOPEN_DEBUG_DISABLE_FIND_DB=1

# execute
srun python -u $COMMAND 

# eof
