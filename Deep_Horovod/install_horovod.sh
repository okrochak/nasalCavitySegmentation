#!/bin/sh
# author: MA
# version: 210701a

module --force purge
module use $OTHERSTAGES 
ml Stages/2020 GCC ParaStationMPI/5.4.7-1-mt Python CMake
# module load NCCL/2.8.3-1-CUDA-11.0 # needed?

# activate virtual python enviornment
source /p/project/prcoe12/RAISE/envAI_deepv/bin/activate

# std install
# HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_CUDA_HOME=/usr/local/software/skylake/Stages/Devel-2020/software/CUDA/11.0 pip install --no-cache-dir horovod

# w/ mpi
HOROVOD_WITH_MPI=1 pip install --no-cache-dir horovod


