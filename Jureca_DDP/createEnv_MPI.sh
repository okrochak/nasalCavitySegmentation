#!/bin/bash
# -*- coding: utf-8 -*-
# author: EI
# version: 221026a
# creates machine specific PyTorch with MPI support using Conda
# use compute node to compile!

# jureca modules
ml --force purge
ml Stages/2022 GCC/11.2.0 ParaStationMPI/5.5.0-1 NCCL/2.12.7-1-CUDA-11.5
ml cuDNN/8.3.1.22-CUDA-11.5 libaio/0.3.112 mpi-settings/CUDA CMake/3.21.1
ml Ninja-Python/1.10.2

# get CUDA version in the system
CUDA_ver="$(echo $EBVERSIONCUDA 2>&1 | tr -d .)"

# miniconda
download=false
if [ -d "$PWD/miniconda3" ];then
  echo "miniconda3 already installed!"
else
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -p $PWD/miniconda3 -b
  download=true
fi

if [ "$download" = true ] ; then
  # std libs
  conda install -y astunparse numpy pyyaml mkl mkl-include setuptools cffi \
          typing_extensions future six requests dataclasses Pillow --force-reinstall

  # cuda support (v11.5)
  conda install -c pytorch -y magma-cuda$CUDA_ver --force-reinstall
  conda install -y pkg-config libuv --force-reinstall

  # fix older library issue
  cp $EBROOTGCC/lib64/libstdc++.so.6.0.29 $CONDA_PREFIX/lib/
  pushd $CONDA_PREFIX/lib/
  rm -f libstdc++.so.6
  ln -s libstdc++.so.6.0.29 libstdc++.so.6
  popd
fi

# enable Conda env
source $PWD/miniconda3/etc/profile.d/conda.sh
conda activate

# pytorch with mpi support
if [ -d "$PWD/pytorch/build/test.dat" ];then
   echo 'pytorch already installed!'
else
   git clone --recursive https://github.com/pytorch/pytorch pytorch
   pushd pytorch
   rm -rf build
   git submodule sync
   git submodule update --init --recursive

   # install pytorch with custom flags
   export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

   mkdir tmp
   export TMPDIR=$PWD/tmp
   export CUDA_HOME=$CUDA_HOME
   python3 setup.py clean
   CMAKE_C_COMPILER=$(which mpicc) CMAKE_CXX_COMPILER=$(which mpicxx) \
           USE_DISTRIBUTED=ON USE_MPI=ON CUDA_ROOT_DIR=$EBROOTCUDA USE_CUDA=ON \
           NCCL_ROOT_DIR=$EBROOTNCCL USE_NCCL=ON USE_GLOO=ON \
           CUDNN_ROOT=$EBROOTCUDNN USE_CUDNN=ON \
           python3 setup.py install
   popd
fi

#eof
