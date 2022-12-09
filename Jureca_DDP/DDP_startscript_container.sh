#!/bin/bash

# general configuration of the job
#SBATCH --job-name=AMDTorchTest
#SBATCH --account=zam
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:15:00

# configure node and process count on the CM
#SBATCH --partition=dc-mi200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --exclusive

# parameters
debug=false # do debug
bs=32       # batch-size
epochs=5    # epochs
lr=0.01     # learning rate

# AT
dataDir="/p/scratch/raise-ctp1/T31_LD/"
COMMAND="DDP_pytorch_AT.py"
EXEC="$COMMAND \
  --batch-size $bs \
  --epochs $epochs \
  --lr $lr \
  --nworker $SLURM_CPUS_PER_TASK \
  --data-dir $dataDir"


### do not modify below ###


# set modules
ml Architecture/jureca_mi200
ml GCC/11.2.0 OpenMPI/4.1.4 ROCm/5.3.0 CMake/3.23.1 
ml UCX-settings/RC-ROCm

# set env variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
mkdir -p $SLURM_SUBMIT_DIR/tmp
export MIOPEN_USER_DB_PATH=$SLURM_SUBMIT_DIR/tmp
export NCCL_DEBUG=WARN

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
if [ "$debug" = true ] ; then
  export NCCL_DEBUG=INFO
fi
echo

# launch container
srun --cpu-bind=none bash -c "apptainer exec --rocm \
        torch_rocm_docker.sif \
        python -m fixed_torch_run \
        --log_dir='logs' \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=8 \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
        --rdzv_backend=c10d \
        --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)'i:29500 \
        $EXEC"

#eof
