#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --job-name=TFtest
#SBATCH --account=deepext
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job.out
#SBATCH --error=job.err

#SBATCH --partition=dp-esb
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00
#SBATCH --exclusive

ml --force purge
ml use $OTHERSTAGES
ml Stages/2022 GCC/11.2.0 OpenMPI/4.1.2 cuDNN/8.3.1.22-CUDA-11.5 NCCL/2.11.4-CUDA-11.5 Python/3.9.6

source /p/project/prcoe12/RAISE/testAI_deepv/bin/activate

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
echo

export CUDA_VISIBLE_DEVICES="0"
export OMP_NUM_THREADS=1

srun --cpu-bind=none --mpi=pspmix python3 -u tensorflow2_synthetic_benchmark.py
