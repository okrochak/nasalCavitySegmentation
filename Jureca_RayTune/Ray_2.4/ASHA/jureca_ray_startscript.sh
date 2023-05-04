#!/bin/bash
# shellcheck disable=SC2206

#SBATCH --job-name=ray_cifar_test
#SBATCH --account=
#SBATCH --output=ray_test_cifar.out
#SBATCH --error=ray_test_cifar.err
#SBATCH --partition=dc-gpu
#SBATCH --nodes=2
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --exclusive


ml --force purge

ml Stages/2023  GCC/11.3.0  OpenMPI/4.1.4 PyTorch/1.12.0-CUDA-11.7 torchvision/0.13.1-CUDA-11.7

source ray_tune_env/bin/activate

COMMAND="cifar_tune_asha.py --scheduler ASHA --num-samples 12 --par-workers 2 --max-iterations 2 --data-dir /p/scratch/raise-ctp2/cifar10/data "

echo $COMMAND

sleep 1
# make sure CUDA devices are visible
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

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

port=7638

export ip_head="$head_node"i:"$port"
export head_node_ip="$head_node"i

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node"i --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $num_gpus  --block &

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

python -u $COMMAND
