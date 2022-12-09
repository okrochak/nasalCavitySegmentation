#!/bin/bash

# general configuration of the job
#SBATCH --job-name=TorchTest
#SBATCH --account=project_465000280
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:15:00

# configure node and process count
#SBATCH --partition=eap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive

# parameters

## MNIST
#dataDir='/scratch/project_465000280/RAISE_datasets/data_MNIST/'
#COMMAND="DDP_pytorch_mnist.py"
#EXEC="$COMMAND --backend nccl --epochs 10 --concM 10 --nworker $SLURM_CPUS_PER_TASK\
# --data-dir $dataDir"

# AT
dataDir='/scratch/project_465000280/RAISE_datasets/actuated_tbl_small/'
COMMAND="DDP_pytorch_AT.py"
EXEC="$COMMAND --epochs 1 --batch-size 1 --concM 1 --benchrun \
       --nworker $SLURM_CPUS_PER_TASK --data-dir $dataDir"

### do not modify below ###

# set modules
ml LUMI/22.08 partition/EAP rocm/5.0.2 ModulePowerUser/LUMI 
ml buildtools cray-python/3.9.12.1 craype-accel-amd-gfx90a

# set env
source /scratch/project_465000280/inancera/envAI_LUMI/bin/activate

# set env vars 
unset GREP_OPTIONS
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export LD_LIBRARY_PATH=$HIP_LIB_PATH:$LD_LIBRARY_PATH
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_TASK" > 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

# miopen
# check: https://rocmsoftwareplatform.github.io/MIOpen/doc/html/install.html
mkdir -p tmp
export MIOPEN_USER_DB_PATH="tmp"
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_FIND_ENFORCE=5

# nccl
# check: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=hsn

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
echo "DEBUG: CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# launch 
srun --cpu-bind=none bash -c "torchrun \
    --log_dir='logs' \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint='$(hostname --ip-address)':29500 \
    $EXEC"

# eof
