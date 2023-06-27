#!/bin/bash
# -*- coding: utf-8 -*-
# author: EI
# version: 230626a
# pull and build containers on LUMI for PyTorch/ROCm
# following https://docs.lumi-supercomputer.eu/software/packages/pytorch/

# load modules and get aws-ofi-rccl plugin with EasyBuild
ml LUMI/22.08 partition/G
ml

# create Cache/TMP so that $HOME would not be used
mkdir -p Cache
mkdir -p TMP
export SINGULARITY_CACHEDIR=$(mktemp -d -p $PWD/Cache)
export SINGULARITY_TMPDIR=$(mktemp -d -p $PWD/TMP)
export SCRATCH="/scratch/project_465000625"

# build container
if [ -f "torch_rocm.sif" ]; then
  echo
  echo 'container already exist'
  echo
else
  # official AMD container with Torch==1.10.0
  #singularity pull torch_rocm.sif docker://amdih/pytorch:rocm5.0_ubuntu18.04_py3.7_pytorch_1.10.0
  
  # docker AMD container with Torch==2.0.0
  # info: https://hub.docker.com/r/rocm/pytorch
  # alternative via DeepSpeed: https://hub.docker.com/r/rocm/deepspeed
  singularity pull torch_rocm.sif docker://rocm/pytorch
  
  # CSC suggested container
  #singularity pull torch_rocm.sif docker://rocm/pytorch:rocm5.4.1_ubuntu20.04_py3.7_pytorch_1.12.1
fi

export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/lib:/opt/cray/libfabric/1.15.2.0/lib64
export SING_FLAGS="-B /opt/cray:/opt/cray 
  -B /usr/lib64/libcxi.so.1:/usr/lib64/libcxi.so.1 
  -B /usr/lib64/libjson-c.so.3:/usr/lib64/libjson-c.so.3
  -B $SCRATCH:$SCRATCH"

# run bash to create envs
if [ -d "torch_rocm_env" ];then
  echo 'environment already exist'
else
  echo "running container_env.sh"
  singularity exec $SING_FLAGS torch_rocm.sif bash -c "bash container_env.sh"
fi
