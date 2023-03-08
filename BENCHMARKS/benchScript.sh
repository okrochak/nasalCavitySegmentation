#!/bin/bash

# general configuration of the job
#SBATCH --job-name=TorchJUBE
#SBATCH --account=#ACC#
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=#TIMELIM#

# configure node and process count on the CM
#SBATCH --partition=#QUEUE#
#SBATCH --nodes=#NODES#
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=#NW#
#SBATCH --gpus-per-node=#NGPU#
#SBATCH --exclusive

# gres options have to be disabled for deepv
#SBATCH --gres=#GRES#

# command to exec
COMMAND="#SCRIPT#"
EXEC="$COMMAND \
        --batch-size #BS# \
        --epochs #EPCS# \
        --lr #LR# \
        --nworker #NW# \
        --shuff \
        --scale-lr \
        --schedule \
        --synt \
        --synt-dpw 100 \
        --benchrun \
        --data-dir #DATADIR#"

# setup
#MODULES#

#ENVS#

#DEVICES#
export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_TASK" > 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# debug 
sleep 1
echo "DEBUG: TIME: $(date)"
echo "DEBUG: EXECUTE: $EXEC"
echo "DEBUG: SLURM_JOB_ID: $SLURM_JOB_ID"
echo "DEBUG: SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_NNODES: $SLURM_NNODES"
echo "DEBUG: SLURM_NTASKS: $SLURM_NTASKS"
echo "DEBUG: SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "DEBUG: SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "DEBUG: SLURMD_NODENAME: $SLURMD_NODENAME"
echo "DEBUG: SLURM_NODEID: $SLURM_NODEID"
echo "DEBUG: SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE"
echo "DEBUG: CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# launch
srun --cpu-bind=none bash -c "torchrun \
    --log_dir='logs' \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_conf=is_host=$((($SLURM_NODEID)) && echo 0 || echo 1) \
    --rdzv_backend=c10d \
    --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)'i:29500 \
    $EXEC"

touch #READY#

# eof
