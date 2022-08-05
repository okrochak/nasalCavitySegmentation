#!/bin/bash

# general configuration of the job
#SBATCH --job-name=DStest
#SBATCH --account=raise-ctp1
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=02:00:00

# configure node and process count on the CM
#SBATCH --partition=dc-gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive

# gres options have to be disabled for deepv
#SBATCH --gres=gpu:4

# parameters
debug=false # do nccl debug
epochs=1    # epochs
be='nccl'   # backend
lr=0.001    # learning rate
bs=2        # batch-size

# AT
dataDir='/p/scratch/raise-ctp1/inanc2/T31_LD/'
COMMAND="DS_pytorch_AT.py"

EXEC=$COMMAND" --batch-size $bs
  --epochs $epochs
  --backend $be
  --nworker $SLURM_CPUS_PER_TASK
  --benchrun
  --data-dir $dataDir"

# set modules
ml --force purge
ml Stages/2022 NVHPC/22.3 ParaStationMPI/5.5.0-1-mt NCCL/2.12.7-1-CUDA-11.5 cuDNN/8.3.1.22-CUDA-11.5
ml Python/3.9.6 libaio/0.3.112 HDF5/1.12.1-serial mpi-settings/CUDA

# set env
source /p/project/raise-ctp1/RAISE/envAI_jureca/bin/activate

# sleep a sec
sleep 1

# job info
echo "DEBUG: TIME: $(date)" 
echo "DEBUG: EXECUTE: $EXEC"
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

# set comm
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_TASK" > 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

#### do not change this part
# create node-list
sysN=$(eval "scontrol show hostnames")
for i in $sysN; do
  x+=\"$i\":[$CUDA_VISIBLE_DEVICES],
done
WID=`echo {${x::-1}} | base64 -w 0`

# modify config file with parameters
sed -i "2s|.*|  \"train_micro_batch_size_per_gpu\": ${bs},|" DS_config.json
sed -i "7s|.*|      \"lr\": ${lr}|" DS_config.json
####

# launch
srun python -m deepspeed.launcher.launch \
  --node_rank $SLURM_PROCID \
  --master_addr ${SLURMD_NODENAME}i \
  --master_port 29500 \
  --world_info $WID \
  $EXEC --deepspeed_mpi --deepspeed_config DS_config.json

# eof
