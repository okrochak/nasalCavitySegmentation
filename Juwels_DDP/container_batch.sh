#!/bin/bash

# general configuration of the job
#SBATCH --job-name=TorchContTest
#SBATCH --account=slfse
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=0-00:15:00

# configure node and process count on the CM
#SBATCH --partition=develbooster
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=4
#SBATCH --exclusive

# gres options have to be disabled for deepv
#SBATCH --gres=gpu:4

# parameters
debug=false # do debug
dataDir='/p/scratch/raise-ctp2/inanc2/T31/'
COMMAND="DDP_ATBL_CAE_mod.py"

EXEC="$COMMAND \
  --batch-size 1 \
  --epochs 10 \
  --lr 0.001 \
  --nworker $SLURM_CPUS_PER_TASK \
  --shuff \
  --scale-lr \
  --schedule \
  --data-dir $dataDir" 


### do not modify below ###


# set modules and envs
ml GCC/11.3.0 OpenMPI/4.1.4 cuDNN/8.6.0.163-CUDA-11.7 Apptainer-Tools/2023
source $SLURM_SUBMIT_DIR/torch_env//bin/activate

# set env vars
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_TASK" > 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# job info 
echo "DEBUG: TIME: $(date)"
echo "DEBUG: EXECUTE: $EXEC"
echo "DEBUG: SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "DEBUG: SLURM_JOB_ID: $SLURM_JOB_ID"
echo "DEBUG: SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_NNODES: $SLURM_NNODES"
echo "DEBUG: SLURM_NTASKS: $SLURM_NTASKS"
echo "DEBUG: SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "DEBUG: SLURM_GPUS_PER_NODE: $SLURM_GPUS_PER_NODE" 
echo "DEBUG: SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "DEBUG: SLURMD_NODENAME: $SLURMD_NODENAME"
echo "DEBUG: SLURM_NODEID: $SLURM_NODEID"
echo "DEBUG: CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
if [ "$debug" = true ] ; then
  export NCCL_DEBUG=INFO
fi
ech

# launch container
srun --cpu-bind=none bash -c "apptainer exec --nv torch.sif \
        python -m fixed_torch_run \
        --log_dir='logs' \
        --nnodes=$SLURM_NNODES \
        --nproc_per_node=$SLURM_GPUS_PER_NODE \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
        --rdzv_backend=c10d \
        --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)'i:29500 \
        $EXEC"

#eof
