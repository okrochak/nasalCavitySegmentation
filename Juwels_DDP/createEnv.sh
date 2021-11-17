#!/bin/bash
# -*- coding: utf-8 -*-
# author: EI
# version: 210601a
# creates machine specific python env

# set modules
module --force purge

# get sys info
cDir=$PWD
sysN=$(eval "uname -n | cut -f2- -d.")
echo "system:${sysN}"
echo

cont1=false
if [ "$sysN" = 'deepv' ] ; then
   module use $OTHERSTAGES
   ml Stages/2020 GCCcore/.9.3.0 GCC/9.3.0 ParaStationMPI/5.4.7-1-mt Python
   cont1=true
elif [ "$sysN" = 'juwels' ] ; then
   ml GCC ParaStationMPI Python
   cont1=true
else
   echo
   echo 'unknown system detected'
   echo 'canceling'
   echo
fi
echo "modules loaded"
echo

if [ "$cont1" = true ] ; then
   pVer=$(eval "python --version")
   pDir=$(eval "which python")

   echo "python ver:${pVer}"
   echo "python dir:${pDir}"
   echo

   # create env
   python -m venv envAI_${sysN}

   # activate env
   source envAI_${sysN}/bin/activate

   pVer=$(eval "python --version")
   pDir=$(eval "which python")
   echo "python env ver:${pVer}"
   echo "python env dir:${pDir}"
   echo

   # install stuff
   TMPDIR=${cDir} pip install \
      torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 \
      -f https://download.pytorch.org/whl/torch_stable.html --no-cache-dir

   echo "a new env is created in ${cDir}"
   echo "activation is done via:"
   echo "source ${cDir}/envAI_${sysN}/bin/activate"
fi

#eof
