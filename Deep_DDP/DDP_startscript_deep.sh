#!/bin/bash

# general configuration of the job
#SBATCH --job-name=TorchTest
#SBATCH --account=deepext
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=0-01:00:00

# configure node and process count on the CM
#SBATCH --partition=dp-esb
# SBATCH --partition=dp-dam
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --exclusive

# parameters
debug=false # do debug
bs=96       # batch-size
epochs=1    # epochs
lr=0.01     # learning rate

# dataset
# MNIST
#dataDir="/p/project/prcoe12/RAISE/data_MNIST/"
#COMMAND="DDP_pytorch_mnist.py"

# AT
dataDir="/p/project/prcoe12/RAISE/T31/"
COMMAND="DDP_pytorch_AT.py"

EXEC="$COMMAND \
  --batch-size $bs \
  --epochs $epochs \
  --lr $lr \
  --nworker $SLURM_CPUS_PER_TASK \
  --data-dir $dataDir"

# set modules
ml --force purge
ml use $OTHERSTAGES
ml Stages/2022 GCC/11.2.0 OpenMPI/4.1.2 cuDNN/8.3.1.22-CUDA-11.5 NCCL/2.11.4-CUDA-11.5 Python/3.9.6

# recent bug: https://gitlab.jsc.fz-juelich.de/software-team/easybuild/-/wikis/Failed-to-initialize-NVML-Driver-library-version-mismatch-message
ml -nvidia-driver/.default

# set env - pip
source /p/project/prcoe12/RAISE/envAI_deepv/bin/activate

# set env - conda
#source /p/project/prcoe12/RAISE/miniconda3_deepv/etc/profile.d/conda.sh
#conda activate

# New CUDA drivers on the compute nodes
ln -s /usr/lib64/libcuda.so.1 .
ln -s /usr/lib64/libnvidia-ml.so.1 .
LD_LIBRARY_PATH=.:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# sleep a sec
sleep 1

# job info 
echo "TIME: $(date)"
echo "DEBUG: EXECUTE: $EXEC"
echo "DEBUG: SLURM_JOB_ID: $SLURM_JOB_ID"
echo "DEBUG: SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_NNODES: $SLURM_NNODES"
echo "DEBUG: SLURM_NTASKS: $SLURM_NTASKS"
echo "DEBUG: SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "DEBUG: SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "DEBUG: SLURMD_NODENAME: $SLURMD_NODENAME"
echo "DEBUG: SLURM_NODEID: $SLURM_NODEID"
echo "DEBUG: CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
if [ "$debug" = true ] ; then
  export NCCL_DEBUG=INFO
fi

# set comm, CUDA and OMP
#export PSP_CUDA=1 # not needed atm
#export PSP_UCP=1 # not needed atm
export CUDA_VISIBLE_DEVICES="0"
export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_TASK" > 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

# launch
srun bash -c "torchrun \
    --log_dir='logs' \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
    --rdzv_backend=c10d \
    --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)'i:29500 \
    $EXEC"

# eof
