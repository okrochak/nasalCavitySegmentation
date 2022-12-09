#!/bin/bash
# -*- coding: utf-8 -*-
# author: EI
# version: 2212008a
# pull and build containers for PyTorch/ROCm

# load modules
ml Architecture/jureca_mi200
ml GCC/11.2.0 OpenMPI/4.1.2 ROCm/5.3.0 CMake/3.23.1 
ml UCX-settings/RC-ROCm

# create Cache/TMP so that $HOME would not be used
mkdir -p Cache
mkdir -p TMP 
export APPTAINER_CACHEDIR=$(mktemp -d -p $PWD/Cache)
export APPTAINER_TMPDIR=$(mktemp -d -p $PWD/TMP)

# official AMD container with Torch==1.10.0
# apptainer pull torch_rocm_amd.sif docker://amdih/pytorch:rocm5.0_ubuntu18.04_py3.7_pytorch_1.10.0

# docker AMD container with Torch==1.12.1
apptainer pull torch_rocm_docker.sif docker://rocm/pytorch

#eof
