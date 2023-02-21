#!/bin/bash
# -*- coding: utf-8 -*-
# author: EI
# version: 230214a
# pull and build containers on LUMI for PyTorch/ROCm
# following https://docs.lumi-supercomputer.eu/software/packages/pytorch/

# load modules and get aws-ofi-rccl plugin with EasyBuild
ml LUMI/22.08 partition/G EasyBuild-user
eb aws-ofi-rccl-66b3b31-cpeGNU-22.08.eb -r

# create Cache/TMP so that $HOME would not be used
mkdir -p Cache
mkdir -p TMP
export SINGULARITY_CACHEDIR=$(mktemp -d -p $PWD/Cache)
export SINGULARITY_TMPDIR=$(mktemp -d -p $PWD/TMP)
export SCRATCH="/scratch/project_465000280"

# official AMD container with Torch==1.10.0
#singularity pull torch_rocm.sif docker://amdih/pytorch:rocm5.0_ubuntu18.04_py3.7_pytorch_1.10.0

# docker AMD container with Torch==1.12.1
# info: https://hub.docker.com/r/rocm/pytorch
# alternative via DeepSpeed: https://hub.docker.com/r/rocm/deepspeed
#singularity pull torch_rocm.sif docker://rocm/pytorch

# CSC suggested container
singularity pull torch_rocm.sif docker://rocm/pytorch:rocm5.4.1_ubuntu20.04_py3.7_pytorch_1.12.1

# run bash to create envs
echo "running ./createDockerEnv.sh"
singularity exec -B $SCRATCH:$SCRATCH torch_rocm.sif bash -c "./createDockerEnv.sh"
