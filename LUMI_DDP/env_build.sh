#!/bin/bash
# -*- coding: utf-8 -*-
# author: EI
# version: 221208a
# creates python env with torch on LUMI

# parameters
nameE='envAI_LUMI'

# set modules
ml LUMI/22.08 partition/EAP rocm/5.0.2 ModulePowerUser/LUMI buildtools cray-python/3.9.12.1 craype-accel-amd-gfx90a

# set env vars special for LUMI
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
PYTHONPATH=$PWD/$nameE/lib/python3.9:/opt/cray/pe/python/3.9.12.1

# torch+torchvision
if [ -f "$PWD/$nameE/bin/torchrun" ]; then
  echo 'torch already installed'
else
  pip${pver} install torch==1.12.1+rocm5.0 torchvision==0.13.1+rocm5.0 torchaudio==0.12.1+rocm5.0 \
          --extra-index-url https://download.pytorch.org/whl/rocm5.0/ --no-cache-dir
fi 

# get rest of the libs
if [ "$cont1" = true ] ; then
  pip${pver} install -r reqs.txt 

  # modify l.4 of /torchnlp/_third_party/weighted_random_sampler.py
  var='int_classes = int'
  sed -i "4s|.*|$var|" \
    $PWD/$nameE/lib/python${pver}/site-packages/torchnlp/_third_party/weighted_random_sampler.py
fi

#eof
