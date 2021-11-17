#!/bin/bash
# -*- coding: utf-8 -*-
# author: EI
# version: 211025a
# creates python env

# get sys info
cDir=$PWD
sysN=$(eval "uname -n | cut -f2- -d.")
echo "system:${sysN}"
echo

# set modules
module --force purge
cont1=false
if [ "$sysN" = 'deepv' ] ; then
  ml GCC ParaStationMPI/5.4.9-1-mt Python cuDNN NCCL Python
  echo "modules loaded"
  echo
  cont1=true
else
  echo
  echo 'unknown system detected'
  echo 'canceling'
  echo
fi

cont2=false
if [ "$cont1" = true ] ; then
  if [ -d "${iDir}/env_AI${sysN}" ];then
    echo 'env already exist!'
  else
    # create env
    python3 -m venv envAI_${sysN}

    # activate env
    source envAI_${sysN}/bin/activate

    TMPDIR=${cDir} pip3 install \
       torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0 \
       -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir

    echo "a new env is created in ${cDir}"
    echo "activation is done via:"
    echo "source ${cDir}/envAI_${sysN}/bin/activate"

    cont2=true
  fi
fi

# get rest of the libraries
if [ "$cont2" = true ] ; then
  pip3 install -r reqs.txt --ignore-installed

  #modify l.4 of /torchnlp/_third_party/weighted_random_sampler.py
  var='int_classes = int'
  sed -i "4s|.*|$var|" \
    $cDir/envAI_${sysN}/lib/python3.8/site-packages/torchnlp/_third_party/weighted_random_sampler.py
fi


#eof
