#!/bin/bash
# -*- coding: utf-8 -*-
# author: EI
# version: 2212008a
# pull and build containers for PyTorch/ROCm

# load modules
ml LUMI/22.08 partition/EAP rocm/5.0.2 ModulePowerUser/LUMI 
ml buildtools cray-python/3.9.12.1 craype-accel-amd-gfx90a

# create Cache/TMP so that $HOME would not be used
mkdir -p Cache
mkdir -p TMP
export SINGULARITY_CACHEDIR=$(mktemp -d -p /scratch/project_465000280/inancera/Container/Cache)
export SINGULARITY_TMPDIR=$(mktemp -d -p /scratch/project_465000280/inancera/Container/TMP)
export APPTAINER_CACHEDIR=$(mktemp -d -p /scratch/project_465000280/inancera/Container/Cache)
export APPTAINER_TMPDIR=$(mktemp -d -p /scratch/project_465000280/inancera/Container/TMP)

# official AMD container with Torch==1.10.0
# singularity pull torch_rocm_amd.sif docker://amdih/pytorch:rocm5.0_ubuntu18.04_py3.7_pytorch_1.10.0

# docker AMD container with Torch==1.12.1
singularity pull torch_rocm_docker.sif docker://rocm/pytorch
