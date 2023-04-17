#!/bin/bash
# -*- coding: utf-8 -*-
# author: EI
# version: 220328a
# creates machine specific python env

# set modules
ml --force purge

# get sys info
cDir=$PWD
sysN="$(uname -n | cut -f2- -d.)"
echo "system:${sysN}"
echo

cont1=false
if [[ $sysN = 'deepv' || $sysN = 'dp-esb'* ]] ; then
  sysN=deepv
  ml use $OTHERSTAGES
  ml Stages/2022 NVHPC/22.1 OpenMPI/4.1.2 NCCL/2.15.1-1-CUDA-11.5 cuDNN/8.3.1.22-CUDA-11.5
  ml Python/3.9.6 HDF5 CMake
  ml -nvidia-driver/.default
  cont1=true
elif [ "$sysN" = 'juwels' ] ; then
  ml Stages/2023 StdEnv/2023 NVHPC/23.1 OpenMPI/4.1.4 cuDNN/8.6.0.163-CUDA-11.7
  ml Python/3.10.4 CMake HDF5 PnetCDF libaio/0.3.112
  cont1=true
elif [ "$sysN" = 'jureca' ] ; then
  #ml Stages/2023 StdEnv/2023 NVHPC/23.1 OpenMPI/4.1.4 cuDNN/8.6.0.163-CUDA-11.7
  ml Stages/2023 StdEnv/2023 GCC/11.3.0 OpenMPI/4.1.4 cuDNN/8.6.0.163-CUDA-11.7
  ml Python/3.10.4 CMake HDF5 PnetCDF libaio/0.3.112
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

# create env
if [ "$cont1" = true ] ; then
  if [ -d "${cDir}/envAI_${sysN}" ];then
    echo 'env already exist'
    echo

    source envAI_${sysN}/bin/activate
  else
    python3 -m venv envAI_${sysN}

    # get headers for pip
    if [ -f "${cDir}/envAI_${sysN}/bin/pip3" ]; then
      echo 'pip already exist'
    else
      cp "$(which pip3)" $cDir/envAI_${sysN}/bin/
      ln -s $cDir/envAI_${sysN}/bin/pip3 $cDir/envAI_${sysN}/bin/pip${pver}
      var="#!$cDir/envAI_${sysN}/bin/python${pver}"
      sed -i "1s|.*|$var|" $cDir/envAI_${sysN}/bin/pip3
    fi
    
    # activate env
    source envAI_${sysN}/bin/activate

    echo "a new env is created in ${cDir}"
    echo "activation is done via:"
    echo "source ${cDir}/envAI_${sysN}/bin/activate"
  fi
fi

# set tmp dir env var
export TMPDIR=${cDir}

# install torch
if [ -f "${cDir}/envAI_${sysN}/bin/torchrun" ]; then
  echo 'Torch already installed'
  echo
else
  # Stages/2023 - CUDA/11.7 - torch 2.0 stable
  pip3 install torch torchvision torchaudio --no-cache-dir
fi

# install horovod
if [ -f "${cDir}/envAI_${sysN}/bin/horovodrun" ]; then
  echo 'Horovod already installed'
  echo
else
  # compiler vars
  export LDSHARED="$CC -shared" &&
  
  # CPU vars
  export HOROVOD_WITH_MPI=1
  export HOROVOD_MPI_THREADS_DISABLE=1
  export HOROVOD_CPU_OPERATIONS=MPI

  # GPU vars
  #export HOROVOD_GPU=CUDA
  #export HOROVOD_CUDA_HOME=$EBROOTCUDA
  #export HOROVOD_GPU_OPERATIONS=MPI
  #export HOROVOD_GPU_OPERATIONS=NCCL
  export HOROVOD_GPU_ALLREDUCE=NCCL
  export HOROVOD_NCCL_LINK=SHARED 
  export HOROVOD_NCCL_HOME=$EBROOTNCCL

  # Host language vars
  export HOROVOD_WITH_PYTORCH=1
  export HOROVOD_WITHOUT_TENSORFLOW=1
  export HOROVOD_WITHOUT_MXNET=1

  pip3 install --no-cache-dir wheel
  pip3 install --no-cache-dir horovod
fi

# install deepspeed
if [ -f "${cDir}/envAI_${sysN}/bin/deepspeed" ]; then
  echo 'DeepSpeed already installed'
  echo
else
  # compile all opt. stuff - not needed & not working 
  #export DS_BUILD_OPS=1 
  # compile req. opt. stuff
  export DS_BUILD_FUSED_ADAM=1
  export DS_BUILD_UTILS=1
  if [ "$sysN" = 'deepv' ] ; then
    #fix libaio issues via:
    export DS_BUILD_AIO=0
  fi

  pip3 install --no-cache-dir DeepSpeed

  # add this to .../deepspeed/launcher/launch.py l.93
  var='    args.node_rank=int(os.environ.get("SLURM_PROCID",0))'
  sed -i "132s|.*|$var|" $cDir/envAI_${sysN}/lib/python${pver}/site-packages/deepspeed/launcher/launch.py
fi

# install heat
if [ -d "${cDir}/envAI_${sysN}/lib/python${pver}/site-packages/heat" ]; then
  echo 'HeAT already installed'
  echo
else
  export CFLAGS="-noswitcherror"
  export CXXFLAGS="-noswitcherror"

  # experimental
  # modify setup.py to accep torch>1.7 for heat
  git clone --recursive https://github.com/helmholtz-analytics/heat.git heat
  var='        "torch>=1.7.0",'
  sed -i "36s|.*|$var|" heat/setup.py

  # create tar ball 
  tar czf heat.tar.gz

  # install experimental heat
  pip3 install --no-cache-dir 'heat.tar.gz[hdf5,netcdf]'
fi

# get rest of the libraries$
if [ "$cont1" = true ] ; then
  # install rest
  pip3 install -r reqs.txt

  # modify l.4 of /torchnlp/_third_party/weighted_random_sampler.py
  var='int_classes = int'
  sed -i "4s|.*|$var|" \
    $cDir/envAI_${sysN}/lib/python${pver}/site-packages/torchnlp/_third_party/weighted_random_sampler.py
fi

# fix IB IP config
if [ -f "${cDir}/envAI_${sysN}/bin/torchrun" ]; then
  sed -i -e '3,8s/^/#/' ${cDir}/envAI_${sysN}/bin/torchrun
  echo """
import re
import sys
from torch.distributed.run import main
from torch.distributed.elastic.agent.server import api as sapi

def new_get_fq_hostname():
    return _orig_get_fq_hostname().replace('.', 'i.', 1)

if __name__ == '__main__':
    _orig_get_fq_hostname = sapi._get_fq_hostname
    sapi._get_fq_hostname = new_get_fq_hostname
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
""" >> ${cDir}/envAI_${sysN}/bin/torchrun
fi

#eof
