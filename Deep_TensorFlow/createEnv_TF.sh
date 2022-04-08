#!/bin/bash
# -*- coding: utf-8 -*-
# author: EI
# version: 220302a
# creates machine specific python env

# set modules
ml --force purge

# get sys info
cDir=$PWD
sysN="$(uname -n | cut -f2- -d.)"
echo "system:${sysN}"
echo

cont1=false
if [ "$sysN" = 'deepv' ] ; then
  ml use $OTHERSTAGES
  ml Stages/2022 GCC OpenMPI cuDNN NCCL Python CMake
  cont1=true
elif [ "$sysN" = 'juwels' ] ; then
  ml Stages/2022 GCC ParaStationMPI Python CMake NCCL libaio cuDNN
  cont1=true
elif [ "$sysN" = 'jureca' ] ; then
  ml Stages/2022 GCC OpenMPI Python NCCL cuDNN libaio CMake
  cont1=true
else
  echo
  echo 'unknown system detected'
  echo 'canceling'
  echo
fi
echo "modules loaded"
echo

# get python version
pver="$(python --version 2>&1 | awk {'print $2'} | cut -f1-2 -d.)"
echo "python version is ${pver}"
echo

if [ "$cont1" = true ] ; then
  if [ -d "${cDir}/testAI_${sysN}" ];then
    echo 'env already exist'
    echo

    source testAI_${sysN}/bin/activate
  else
    # create env
    python3 -m venv testAI_${sysN}

    # get headers for pip
    if [ -f "${cDir}/testAI_${sysN}/bin/pip3" ]; then
      echo 'pip already exist'
    else
      cp "$(which pip3)" $cDir/testAI_${sysN}/bin/
      ln -s $cDir/testAI_${sysN}/bin/pip3 $cDir/testAI_${sysN}/bin/pip${pver}
      var="#!$cDir/testAI_${sysN}/bin/python${pver}"
      sed -i "1s|.*|$var|" $cDir/testAI_${sysN}/bin/pip3
    fi

    # activate env
    source testAI_${sysN}/bin/activate

    echo "a new env is created in ${cDir}"
    echo "activation is done via:"
    echo "source ${cDir}/testAI_${sysN}/bin/activate"
  fi
fi

# install TF 
if [ -f "${cDir}/testAI_${sysN}/bin/tensorboard" ]; then
  echo 'TF already installed'
  echo
else
  export TMPDIR=${cDir}

  pip3 install --upgrade tensorflow --no-cache-dir
fi

# install horovod
if [ -f "${cDir}/testAI_${sysN}/bin/horovodrun" ]; then
  echo 'Horovod already installed'
  echo
else
  export HOROVOD_GPU=CUDA
  export HOROVOD_GPU_OPERATIONS=NCCL
  export HOROVOD_WITH_TENSORFLOW=1
  export TMPDIR=${cDir}

  pip3 install --no-cache-dir horovod --ignore-installed
fi

# get rest of the libraries$
if [ "$cont1" = true ] ; then
  pip3 install -r reqs_TF.txt --ignore-installed
fi
 

# eof
