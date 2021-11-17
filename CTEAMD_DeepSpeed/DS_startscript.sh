#!/bin/bash

# general configuration of the job
#SBATCH --job-name=TT
#SBATCH -D .
#SBATCH --qos=bsc_case
#SBATCH --time=00:10:00
#SBATCH --output=job.out
#SBATCH --error=job.err

# configure node and process count on the CM
#SBATCH --nodes=8
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

# gres options
#SBATCH --gres=gpu:2

# parameters
debug=false # do nccl debug
epochs=10   # epochs
cm=1        # data-size (concat dataset for MNIST)
be='nccl'   # backend 
lr=0.01     # learning rate
ri=10       # do restart interval
li=1        # do log interval

# batch-size in DS has to be divisible to #GPGPUs for DS
bs=96       # batch-size

# MNIST
#dataDir='/gpfs/projects/bsc21/bsc21163/data_MNIST/'
#COMMAND="DS_pytorch_mnist.py"

# AT
dataDir='/gpfs/projects/bsc21/bsc21163/T31/'
COMMAND="DS_pytorch_AT.py"

EXEC=$COMMAND" --batch-size $bs
  --epochs $epochs
  --concM $cm
  --backend $be
  --lr $lr
  --restart-int $ri
  --log-int $li
  --benchrun
  --data-dir $dataDir"

# set modules
ml gcc openmpi rocm python

# set env
source /gpfs/projects/bsc21/bsc21163/envAI_BSC/bin/activate
export LD_LIBRARY_PATH=/gpfs/projects/bsc21/bsc21163/envAI_BSC/lib:$LD_LIBRARY_PATH

# sleep a sec
sleep 1

# job info 
echo "COMMAND: $COMMAND"
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
export MIOPEN_DEBUG_DISABLE_FIND_DB=1

#### do not change this part
# create node-list
sysN=$(eval "scontrol show hostnames")
for i in $sysN; do
  x+=\"$i\":[$CUDA_VISIBLE_DEVICES],
done
WID=`echo {${x::-1}} | base64 -w 0`

# modify config file with parameters
sed -i "2s|.*|  \"train_batch_size\": ${bs},|" DS_config.json
####

# launch
srun python -m deepspeed.launcher.launch \
  --node_rank $SLURM_PROCID \
  --master_addr $SLURMD_NODENAME \
  --master_port 29500 \
  --world_info $WID \
  $EXEC --deepspeed_mpi --deepspeed_config DS_config.json

# eof
