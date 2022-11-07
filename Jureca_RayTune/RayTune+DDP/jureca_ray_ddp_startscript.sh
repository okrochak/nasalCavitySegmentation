#!/bin/bash

# general configuration of the job
#SBATCH --job-name=RayTuneDDP
#SBATCH --account=raise-ctp2
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=01:00:00

# configure node and process count on the CM
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive

# gres options have to be disabled for deepv
#SBATCH --gres=gpu:4


ml --force purge
ml Stages/2022  GCC/11.2.0 CUDA/11.5 Python/3.9.6 PyTorch/1.11-CUDA-11.5 torchvision/0.12.0


num_gpus=4
# set env
source ddp_ray_env/bin/activate


# set comm
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_TASK" > 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

# launch
# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}

# __doc_head_ray_start__
port=8374

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node"i --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 4 --block &
# __doc_head_ray_end__

# __doc_worker_ray_start__

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$head_node"i:"$port" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 4 --block &
    sleep 5
done

echo "Ready"

python3 -u cifar_tune.py


# eof
