#!/bin/bash

# general configuration of the job
#SBATCH --job-name=ddp_run
#SBATCH --account=slfse
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job_ddp.out
#SBATCH --error=job_ddp.err
#SBATCH --time=00:30:00

# configure node and process count on the CM
#SBATCH --partition=develbooster
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive

# command to exec
bs=64
epochs=90
be='nccl'
COMMAND="ddp_resnet.py --batch-size $bs --epochs $epochs --backend $be"
echo "DEBUG: EXECUTE=$COMMAND"

# set modules 
module --force purge
ml GCC ParaStationMPI cuDNN NCCL Python

source /p/home/jusers/aach1/juwels/aach1/virtual_env/torch_cuda/bin/activate
# sleep a sec
sleep 1

debug=false
# Echo job configuration
if [ "$debug" = true ] ; then
   echo "DEBUG: SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
   echo "DEBUG: SLURM_NNODES=$SLURM_NNODES"
   echo "DEBUG: SLURM_NTASKS=$SLURM_NTASKS"
   echo "DEBUG: SLURM_TASKS_PER_NODE=$SLURM_TASKS_PER_NODE"
   echo "DEBUG: SLURM_SUBMIT_HOST=$SLURM_SUBMIT_HOST"
   echo "DEBUG: SLURM_NODEID=$SLURM_NODEID"
   echo "DEBUG: SLURM_LOCALID=$SLURM_LOCALID" 
   echo "DEBUG: SLURM_PROCID=$SLURM_PROCID"
fi

export PSP_CUDA=1
export PSP_UCP=1
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OMP_NUM_THREADS=1

# launch
srun python -m torch.distributed.run \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$SLURMD_NODENAME.juwels:29500 \
    $COMMAND


# eof

