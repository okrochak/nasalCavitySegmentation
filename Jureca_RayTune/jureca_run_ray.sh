#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=RayTuneTest
#SBATCH --account=raise-ctp2
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=ray_tune.out
#SBATCH --error=ray_tune.err

#SBATCH --partition=dc-gpu
#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --exclusive

ml --force purge

ml Stages/2022  GCC/11.2.0  OpenMPI/4.1.2 PyTorch/1.11-CUDA-11.5 torchvision/0.12.0-CUDA-11.5

source ray_tune_env/bin/activate

sleep 1
# make sure CUDA devices are visible
export CUDA_VISIBLE_DEVICES="0,1,2,3"

num_gpus=4

## Limit number of max pending trials
export TUNE_MAX_PENDING_TRIALS_PG=$(($SLURM_NNODES * 4))

## Disable Ray Usage Stats
export RAY_USAGE_STATS_DISABLE=1

####### this part is taken from the ray example slurm script #####
set -x

# __doc_head_address_start__

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi
# __doc_head_address_end__

# __doc_head_ray_start__
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node"i --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $num_gpus  --block &
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
        ray start --address "$head_node"i:"$port" --redis-password='5241590000000000' \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $num_gpus --block &
    sleep 5
done

echo "Ready"

python3 -u  cifar_tune.py
