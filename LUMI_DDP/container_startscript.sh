#!/bin/bash

# general configuration of the job
#SBATCH --job-name=TorchTest
#SBATCH --account=project_465000280
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=0-00:15:00

# configure node and process count
# SBATCH --partition=dev-g
#SBATCH --partition=small-g
# SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --exclusive

# parameters

## MNIST
#dataDir='/scratch/project_465000280/RAISE_datasets/data_MNIST/'
#COMMAND="DDP_pytorch_mnist.py"
#EXEC="$COMMAND --backend nccl --epochs 10 --concM 10 --nworker $SLURM_CPUS_PER_TASK\
# --data-dir $dataDir"

# AT
dataDir='/scratch/project_465000280/RAISE_datasets/actuated_tbl/'
COMMAND="DDP_pytorch_AT.py"
EXEC="$COMMAND --epochs 1 --batch-size 1 --concM 1 --benchrun \
       --nworker $SLURM_CPUS_PER_TASK --data-dir $dataDir"

### do not modify below ###

# set modules
ml LUMI/22.08 partition/G rocm/5.0.2 ModulePowerUser/LUMI buildtools cray-python/3.9.12.1

# set env vars 
unset GREP_OPTIONS
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
export LD_LIBRARY_PATH=$HIP_LIB_PATH:$LD_LIBRARY_PATH
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_TASK" > 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

# miopen
# check: https://rocmsoftwareplatform.github.io/MIOpen/doc/html/install.html
export MIOPEN_USER_DB_PATH="/tmp/sam-miopen-cache-$SLURM_PROCID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
#rm -rf $MIOPEN_USER_DB_PATH
mkdir -p $MIOPEN_USER_DB_PATH
#export MIOPEN_DEBUG_DISABLE_FIND_DB=1
#export MIOPEN_DISABLE_CACHE=1
export MIOPEN_FIND_ENFORCE=5

# nccl
# check: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn
#export NCCL_SOCKET_IFNAM=hsn0,hsn1,hsn2,hsn3

# job info 
echo "DEBUG: TIME: $(date)"
echo "DEBUG: EXECUTE: $EXEC"
echo "DEBUG: SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "DEBUG: SLURM_JOB_ID: $SLURM_JOB_ID"
echo "DEBUG: SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_NNODES: $SLURM_NNODES"
echo "DEBUG: SLURM_NTASKS: $SLURM_NTASKS"
echo "DEBUG: SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "DEBUG: SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "DEBUG: SLURMD_NODENAME: $SLURMD_NODENAME"
echo "DEBUG: HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"

# required if --cpu-bind is required, bind with --cpu-bind=mask_cpu:$MASKS
MASKS="ff000000000000,ff00000000000000,ff0000,ff000000,ff,ff00,ff00000000,ff0000000000"

# singularity does not bind SCRATCH at this moment!
export SCRATCH_P="/scratch/project_465000280"

# launch
srun --cpu-bind=none bash -c \
    "singularity exec --rocm -B $SCRATCH_P torch_rocm_docker.sif torchrun \
    --log_dir='logs' \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint='$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)':34567 \
    $EXEC"

# eof
