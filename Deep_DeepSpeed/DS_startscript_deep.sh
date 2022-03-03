#!/bin/bash

# general configuration of the job
#SBATCH --job-name=DSTest
#SBATCH --account=deepext
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=00:30:00

# configure node and process count on the CM
#SBATCH --partition=dp-esb
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --exclusive

# parameters
debug=false # do nccl debug
epochs=10   # epochs
lr=0.01     # learning rate
bs=96       # batch-size

# AT
dataDir="/p/project/prcoe12/RAISE/T31/"
COMMAND="DS_pytorch_AT.py"
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

# set env - pip
source /p/project/prcoe12/RAISE/envAI_deepv/bin/activate

# sleep a sec
sleep 1

# job info 
echo "DEBUG: EXECUTE: $EXEC"
echo "DEBUG: SLURM_JOB_ID: $SLURM_JOB_ID"
echo "DEBUG: SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_NNODES: $SLURM_NNODES"
echo "DEBUG: SLURM_NTASKS: $SLURM_NTASKS"
echo "DEBUG: SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "DEBUG: SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "DEBUG: SLURMD_NODENAME: $SLURMD_NODENAME"
echo "DEBUG: SLURM_NODEID: $SLURM_NODEID"
echo "DEBUG: SLURM_LOCALID: $SLURM_LOCALID" 
echo "DEBUG: SLURM_PROCID: $SLURM_PROCID"
echo "DEBUG: CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
if [ "$debug" = true ] ; then
  export NCCL_DEBUG=INFO
fi
echo

#### do not change this part
# create node-list
sysN=$(scontrol show hostnames)
for i in $sysN; do
  x+=\"$i\":[$CUDA_VISIBLE_DEVICES],
done
WID=`echo {${x::-1}} | base64 -w 0`

# modify config file with parameters
sed -i "2s|.*|  \"train_micro_batch_size_per_gpu\": ${bs},|" DS_config.json
####

srun python3 -m deepspeed.launcher.launch \
  --node_rank $SLURM_PROCID \
  --master_addr ${SLURMD_NODENAME}i \
  --master_port 29500 \
  --world_info $WID \
  $EXEC --deepspeed_mpi --deepspeed_config DS_config.json

# eof
