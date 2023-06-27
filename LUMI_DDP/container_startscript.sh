#!/bin/bash

# general configuration of the job
#SBATCH --job-name=TorchTest
#SBATCH --account=project_465000625
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=0-00:30:00

# configure node and process count
#SBATCH --partition=dev-g
# SBATCH --partition=small-g
# SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --exclusive

# parameters
dataDir='/scratch/${SBATCH_ACCOUNT}/RAISE_datasets/actuated_tbl/'
COMMAND="DDP_ATBL_CDM.py"
EXEC="$COMMAND \
        --batch-size 2 \
        --epochs 10 \
        --lr 0.0001 \
        --nworker $SLURM_CPUS_PER_TASK \
        --shuff \
        --scale-lr \
        --schedule \
        --synt \
        --synt-dpw 100 \
        --benchrun \
        --data-dir $dataDir"


### do not modify below ###


# set modules
ml LUMI/22.08 partition/G

# set env vars
unset GREP_OPTIONS
#-hip
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
export LD_LIBRARY_PATH=$HIP_LIB_PATH:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1
export HIP_LAUNCH_BLOCKING=1
#-threads
export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_TASK" > 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi
#-nccl
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=3
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10
#-miopen
export MIOPEN_DISABLE_CACHE=1
export MIOPEN_USER_DB_PATH=/tmp/miopen-userdb-${USER}
export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen-cache-${USER}
#-rocm
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1
#-slurm
export SLURM_MPI_TYPE=pmi2
#-container
export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/lib:/opt/cray/libfabric/1.15.2.0/lib64

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

# launch
export SCRATCH="/scratch/${SBATCH_ACCOUNT}"
export SING_FLAGS="-B $SCRATCH:$SCRATCH"
srun --cpu-bind=none singularity exec $SING_FLAGS \
    torch_rocm.sif bash -c ". torch_rocm_env/bin/activate; torchrun \
    --log_dir='logs' \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint='$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)':24900 \
    $EXEC"

# eof
