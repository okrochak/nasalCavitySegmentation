#!/bin/bash

# general configuration of the job
#SBATCH --job-name=TorchTest
#SBATCH --account=project_465000280
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
#SBATCH --exclusive

# parameters
dataDir='/scratch/${SBATCH_ACCOUNT}/RAISE_datasets/actuated_tbl/'
COMMAND="DDP_pytorch_AT_LD_mod.py"
EXEC="$COMMAND --epochs 1 --batch-size 1 --concM 1 --benchrun \
       --nworker $SLURM_CPUS_PER_TASK --data-dir $dataDir"

### do not modify below ###

# set modules
#-standard
#ml LUMI/22.08 partition/G rocm/5.0.2 ModulePowerUser/LUMI buildtools cray-python/3.9.12.1
#-preconfig by LUMI support
module use /pfs/lustrep2/projappl/project_462000125/samantao-public/mymodules
module load aws-ofi-rccl/sam-rocm-5.3.3.lua
#-CSC https://docs.lumi-supercomputer.eu/software/local/csc/
#ml use /appl/local/csc/modulefiles
#ml pytorch

# set venv
source /scratch/${SCRATCH_ACCOUNT}/inancera/envAI_LUMI/bin/activate

# set env vars
unset GREP_OPTIONS
#-hip
export HIP_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
export LD_LIBRARY_PATH=$HIP_LIB_PATH:$LD_LIBRARY_PATH
#-threads
export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_TASK" > 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi
#-nccl
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=3
#-miopen
mkdir -p /tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
#-rocm
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1

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
srun --cpu-bind=none bash -c "torchrun \
    --log_dir='logs' \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint='$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)':39591 \
    $EXEC"

# eof
