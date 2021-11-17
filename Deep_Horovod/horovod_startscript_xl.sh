#!/bin/bash

# general configuration of the job
#SBATCH --job-name=TorchTest
#SBATCH --account=deepext
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=06:00:00

# configure node and process count on the CM
#SBATCH --partition=dp-esb-ib
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --exclusive

# command to exec
bs=32
epochs=1 
cm=2
dataDir="/p/project/prcoe12/RAISE/data_MNIST/"
COMMAND="horo_pytorch_mnist.py --batch-size $bs --epochs $epochs --concM $cm --data-dir $dataDir" 
echo "DEBUG: EXECUTE=$COMMAND"

# debug? 
debug=false

# set modules 
module --force purge
module use $OTHERSTAGES 
ml Stages/2020 GCC ParaStationMPI/5.4.7-1-mt Python

# source env
source /p/project/prcoe12/RAISE/envAI_deepv/bin/activate

# sleep a sec
# sleep 1

# job info 
echo "DEBUG: SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_NNODES: $SLURM_NNODES"
echo "DEBUG: SLURM_NTASKS: $SLURM_NTASKS"
echo "DEBUG: SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "DEBUG: SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "DEBUG: SLURM_NODEID: $SLURM_NODEID"
echo "DEBUG: SLURM_LOCALID: $SLURM_LOCALID" 
echo "DEBUG: SLURM_PROCID: $SLURM_PROCID"

# set comm
PSP_CUDA=1
PSP_UCP=1 
export NCCL_SOCKET_IFNAME=ib
export NCCL_IB_HCA=ipogif0 
export NCCL_IB_CUDA_SUPPORT=1
export CUDA_VISIBLE_DEVICES="0,1,2,3"
if [ "$debug" = true ] ; then
  export NCCL_DEBUG=VERSION
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=ALL
fi

# launch
srun --cpu-bind=none python -u $COMMAND

# eof
