#!/bin/bash
# -*- coding: utf-8 -*-
# author: EI
# version: 230626a
# creates python env with torch on LUMI

# parameters
nameE='envAI_LUMI'

# set modules
#ml LUMI/22.08 partition/G rocm

# module package suggested by LUMI support
ml use /pfs/lustrep2/projappl/project_462000125/samantao-public/mymodules
ml aws-ofi-rccl/rocm-5.5.0

# set env vars special for LUMI with rocm 5.3.0
export LD_LIBRARY_PATH=$HIP_LIB_PATH:$LD_LIBRARY_PATH
export MPICH_GPU_SUPPORT_ENABLED=1

# get python version
pver="$(python --version 2>&1 | awk {'print $2'} | cut -f1-2 -d.)"
echo "python version is ${pver}"

# create env
cont1=false
if [ -d "$PWD/$nameE" ];then
  echo 'env already exist'
else
  python${pver} -m venv $nameE
  cont1=true
fi
  
# activate env
source $nameE/bin/activate

# unset system python?? special for LUMI
unset PYTHONPATH
PYTHONPATH=$PWD/$nameE/lib/python3.9:/opt/cray/pe/python/3.9.13.1

# get wheel first
pip3 install --no-cache-dir wheel

# torch+torchvision
if [ -f "$PWD/$nameE/bin/torchrun" ]; then
  echo 'torch already installed'
else
  #pip${pver} install torch==1.12.1+rocm5.0 torchvision==0.13.1+rocm5.0 torchaudio==0.12.1+rocm5.0 \
  #        --extra-index-url https://download.pytorch.org/whl/rocm5.0/ --no-cache-dir
  
  #pip${pver} install --pre torch torchvision torchaudio \
  #        --extra-index-url https://download.pytorch.org/whl/nightly/rocm5.3 --no-cache-dir

  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
fi 

# install horovod
if [ -f "$PWD/$nameE/bin/horovodrun" ]; then
  echo 'Horovod already installed'
else
  # compiler vars
  export LDSHARED="$CC -shared" &&

  # CPU vars
  export HOROVOD_MPI_THREADS_DISABLE=1
  export HOROVOD_CPU_OPERATIONS=MPI

  # GPU vars
  export HOROVOD_GPU=ROCM
  export HOROVOD_GPU_OPERATIONS=NCCL

  # Host language vars
  export HOROVOD_WITH_PYTORCH=1
  export HOROVOD_WITHOUT_TENSORFLOW=1
  export HOROVOD_WITHOUT_MXNET=1

  pip3 install --no-cache-dir horovod
fi

# get rest of the libs
if [ "$cont1" = true ] ; then
  pip${pver} install -r reqs.txt 

  # modify l.4 of /torchnlp/_third_party/weighted_random_sampler.py
  var='int_classes = int'
  sed -i "4s|.*|$var|" \
    $PWD/$nameE/lib/python${pver}/site-packages/torchnlp/_third_party/weighted_random_sampler.py
fi

echo "unit tests:"
for item in 'torch' 'horovod';do
  python3 -c "import $item; print('$item version:',$item.__version__)"
done

#eof
