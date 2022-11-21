#!/bin/bash
# -*- coding: utf-8 -*-
# author: EI
# version: 221121a
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
  ml Stages/2022 GCC/11.2.0 ParaStationMPI/5.5.0-1 cuDNN/8.3.1.22-CUDA-11.5 NCCL/2.12.7-1-CUDA-11.5 Python/3.9.6 CMake/3.21.1
  #ml Stages/2022 GCC/11.2.0 OpenMPI/4.1.2 cuDNN/8.3.1.22-CUDA-11.5 NCCL/2.12.7-1-CUDA-11.5 Python/3.9.6 CMake/3.21.1
  cont1=true
elif [ "$sysN" = 'juwels' ] ; then
  ml GCC ParaStationMPI Python CMake
  cont1=true
elif [ "$sysN" = 'jureca' ] ; then
  #ml Stages/2022 GCC ParaStationMPI Python CMake NCCL libaio # Horovod iassues with pscom??
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
  if [ -d "${cDir}/envAI_${sysN}" ];then
    echo 'env already exist'
    echo

    source envAI_${sysN}/bin/activate
  else
    # create env
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

# install torch
if [ -f "${cDir}/envAI_${sysN}/bin/torchrun" ]; then
  echo 'Torch already installed'
  echo
else
  export TMPDIR=${cDir}

  #pip3 install \
  #   torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 -f \
  #           https://download.pytorch.org/whl/cu113/torch_stable.html --no-cache-dir

  ## Stages/2022 - CUDA/11.5
  #pip3 install \
  #   torch==1.11.0+cu115 torchvision==0.12.0+cu115 torchaudio==0.11.0+cu115 -f \
  #           https://download.pytorch.org/whl/cu115/torch_stable.html --no-cache-dir

  # Stages/2022 - CUDA/11.7
  pip3 install \
     torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 -f \
             https://download.pytorch.org/whl/cu117/torch_stable.html --no-cache-dir
fi

# install horovod
if [ -f "${cDir}/envAI_${sysN}/bin/horovodrun" ]; then
  echo 'Horovod already installed'
  echo
else
  pip3 install --no-cache-dir wheel
  #export HOROVOD_DEBUG=1
  export HOROVOD_GPU=CUDA
  export HOROVOD_CUDA_HOME=$EBROOTCUDA
  #export HOROVOD_WITH_MPI=1
  export HOROVOD_MPI_THREADS_DISABLE=1
  #export HOROVOD_GPU_OPERATIONS=MPI # only turn this off
  export HOROVOD_GPU_OPERATIONS=NCCL # only turn this off
  export HOROVOD_NCCL_HOME=$EBROOTNCCL
  export HOROVOD_WITH_PYTORCH=1
  export HOROVOD_WITHOUT_TENSORFLOW=1
  export HOROVOD_WITHOUT_MXNET=1
  export TMPDIR=${cDir}

  pip3 install --no-cache-dir horovod --ignore-installed
fi

# install deepspeed
if [ -f "${cDir}/envAI_${sysN}/bin/deepspeed" ]; then
  echo 'DeepSpeed already installed'
  echo
else
  #export DS_BUILD_OPS=1
  # if above not working?? recursion error use this
  export DS_BUILD_FUSED_ADAM=1
  export DS_BUILD_UTILS=1
  if [ "$sysN" = 'deepv' ] ; then
    #fix libaio issues via:
    export DS_BUILD_AIO=0
  fi
  export TMPDIR=${cDir}

  pip3 install --no-cache-dir DeepSpeed

  # add this to deepspeed/launcher/launch.py l.126
  var='    args.node_rank=int(os.environ.get("SLURM_PROCID",0))'
  sed -i "126s|.*|$var|" $cDir/envAI_${sysN}/lib/python${pver}/site-packages/deepspeed/launcher/launch.py
fi

# install heat
if [ -d "${cDir}/envAI_${sysN}/lib/python${pver}/site-packages/heat" ]; then
  echo 'HeAT already installed'
  echo
else
  export TMPDIR=${cDir}

  # need to modify setup.py to accep torch>1.9 for heat
  wget https://files.pythonhosted.org/packages/5d/3a/4781f1e6910753bfdfa6712c83c732c60e675d8de14983926a0d9306c7a6/heat-1.1.1.tar.gz
  tar xzf heat-1.1.1.tar.gz
  var='        "torch>=1.7.0",'
  sed -i "36s|.*|$var|" heat-1.1.1/setup.py
  var='        "torchvision>=0.8.0",'
  sed -i "39s|.*|$var|" heat-1.1.1/setup.py

  # create tar again!
  rm -rf heat-1.1.1.tar.gz
  tar czf heat-1.1.1.tar.gz heat-1.1.1
  rm -rf heat-1.1.1

  pip3 install --no-cache-dir 'heat-1.1.1.tar.gz[hdf5,netcdf]'

  rm -rf heat-1.1.1.tar.gz
fi

# get rest of the libraries$
if [ "$cont1" = true ] ; then
  # install rest
  pip3 install -r reqs.txt --ignore-installed

  # modify l.4 of /torchnlp/_third_party/weighted_random_sampler.py
  var='int_classes = int'
  sed -i "4s|.*|$var|" \
    $cDir/envAI_${sysN}/lib/python${pver}/site-packages/torchnlp/_third_party/weighted_random_sampler.py
fi

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
