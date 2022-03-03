#!/bin/sh
# author: EI
# version: 210709a

# get dir
iDir=$PWD

# set modules
module --force purge
module use $OTHERSTAGES 
ml Stages/2020 GCC/9.3.0 ParaStationMPI/5.4.7-1-mt CMake Ninja cuDNN NCCL mpi-settings/CUDA

# conda
if [ -d "${iDir}/miniconda3" ];then
   echo "miniconda3 already installed!" 
   source ${iDir}/miniconda3/etc/profile.d/conda.sh
   conda activate
else
   echo "miniconda3 will be compiled to ${iDir}/miniconda3!"
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh -p ${iDir}/miniconda3 -b
   source ${iDir}/miniconda3/etc/profile.d/conda.sh
   conda activate
   # std libs
   conda install -y astunparse numpy pyyaml mkl mkl-include setuptools cffi typing_extensions future six requests dataclasses Pillow --force-reinstall
   # cuda - check version with yours
   conda install -c pytorch -y magma-cuda110 --force-reinstall
   conda install -y pkg-config libuv --force-reinstall
   rm -f Miniconda3-latest-Linux-x86_64.sh
fi

# torch
if [ -d "${iDir}/pytorch/build" ];then
   echo 'pytorch already installed!'
else
   # clone pytorch
   if [ -d "${iDir}/pytorch" ];then
      echo 'pytorch repo is found!'
   else
      git clone --recursive https://github.com/pytorch/pytorch pytorch
   fi

   # update repos
   cd pytorch
   git submodule sync
   git submodule update --init --recursive

   # install pytorch
   export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
   export TMPDIR=${iDir}/tmp
   python setup.py clean
   CMAKE_C_COMPILER=$(which mpicc) CMAKE_CXX_COMPILER=$(which mpicxx) USE_DISTRIBUTED=ON USE_MPI=ON USE_CUDA=ON NCCL_ROOT_DIR=$EBROOTNCCL USE_NCCL=ON USE_GLOO=ON CUDNN_ROOT=$EBROOTCUDNN USE_CUDNN=ON python setup.py install
   cd ..
fi

# torchvision
if [ -d "${iDir}/torchvision/build" ];then
   echo 'torchvision already installed!'
else
   # clone torchvision
   if [ -d "${iDir}/torchvision" ];then
      echo 'torchvision repo is found!'
   else
      git clone --recursive https://github.com/pytorch/vision.git torchvision
   fi

   # update repos
   cd torchvision
   git submodule sync
   git submodule update --init --recursive

   # install torchvision
   export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
   export TMPDIR=${iDir}/tmp
   python setup.py clean
   CMAKE_C_COMPILER=$(which mpicc) CMAKE_CXX_COMPILER=$(which mpicxx) FORCE_CUDA=ON python setup.py install
fi

echo 'done!'
# eof
