#!/bin/bash
# -*- coding: utf-8 -*-
# author: EI
# version: 230120a
# creates machine specific python env
# env ONLY

# set modules
module --force purge

# get sys info
#sysN="$(uname -n | cut -f2- -d.)"
sysN="deepv"
echo "system:${sysN}"
echo

# create tmp dir
mkdir -p $PWD/tmp
export TMPDIR=$PWD/tmp

if [ "$sysN" = 'deepv' ] ; then
  module use $OTHERSTAGES
  # main
  ml Stages/2022 NVHPC/22.1 OpenMPI/4.1.2 NCCL/2.15.1-1-CUDA-11.5 cuDNN/8.3.1.22-CUDA-11.5

  # side
  ml Python/3.9.6 HDF5 CMake

  # version mismatch fix
  ml -nvidia-driver/.default

  # new cuda drivers in comp node, only use this if salloc
  ln -s /usr/lib64/libcuda.so.1 .
  ln -s /usr/lib64/libnvidia-ml.so.1 .
  LD_LIBRARY_PATH=.:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

elif [ "$sysN" = 'juwels' ] ; then
  ml GCC ParaStationMPI Python CMake
elif [ "$sysN" = 'jureca' ] ; then
  # main
  ml Stages/2022 NVHPC/22.1 ParaStationMPI/5.5.0-1-mt NCCL/2.14.3-1-CUDA-11.5 cuDNN/8.3.1.22-CUDA-11.5

  # side
  ml Python/3.9.6 libaio/0.3.112 HDF5/1.12.1 PnetCDF/1.12.2 mpi-settings/CUDA CMake/3.21.1
else
  echo 'unknown system detected'
  echo 'canceling'
  exit
fi
echo 'modules loaded'
echo

# get python version
pver="$(python --version 2>&1 | awk {'print $2'} | cut -f1-2 -d.)"
echo "python version is ${pver}"
echo

if [ -d "$PWD/envAI_${sysN}" ];then
  echo 'env already exist'
  echo

  source envAI_${sysN}/bin/activate
else
  # create env
  python3 -m venv envAI_${sysN}

  # get headers for pip
  if [ -f "$PWD/envAI_${sysN}/bin/pip3" ]; then
    echo 'pip already exist'
  else
    cp "$(which pip3)" $PWD/envAI_${sysN}/bin/
    ln -s $PWD/envAI_${sysN}/bin/pip3 $PWD/envAI_${sysN}/bin/pip${pver}
    var="#!$PWD/envAI_${sysN}/bin/python${pver}"
    sed -i "1s|.*|$var|" $PWD/envAI_${sysN}/bin/pip3
  fi

  # activate env
  source envAI_${sysN}/bin/activate

  echo "a new env is created in $PWD"
  echo "activation is done via:"
  echo "source $PWD/envAI_${sysN}/bin/activate"
fi

# install torch
if [ -f "$PWD/envAI_${sysN}/bin/torchrun" ]; then
  echo 'Torch already installed'
  echo
else
  pip3 install --no-cache-dir \
    torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 -f \
    https://download.pytorch.org/whl/cu117/torch_stable.html --no-cache-dir
fi

# install horovod
if [ -f "$PWD/envAI_${sysN}/bin/horovodrun" ]; then
  echo 'Horovod already installed'
  echo
else
  #export HOROVOD_DEBUG=1
  export HOROVOD_WITH_MPI=1
  export HOROVOD_MPI_THREADS_DISABLE=1
  export HOROVOD_GPU=CUDA
  #export HOROVOD_GPU_OPERATIONS=MPI
  export HOROVOD_CUDA_HOME=$EBROOTCUDA
  export HOROVOD_GPU_OPERATIONS=NCCL
  export HOROVOD_NCCL_HOME=$EBROOTNCCL
  export HOROVOD_WITH_PYTORCH=1
  export HOROVOD_WITHOUT_TENSORFLOW=1
  export HOROVOD_WITHOUT_MXNET=1

  pip3 install --no-cache-dir wheel --ignore-installed
  pip3 install --no-cache-dir horovod==0.25.0 --ignore-installed
fi

# install deepspeed
if [ -f "$PWD/envAI_${sysN}/bin/deepspeed" ]; then
  echo 'DeepSpeed already installed'
  echo
else
  # compile all opt. stuff - not needed & not working
  #export DS_BUILD_OPS=1
  # compile req. opt. stuff
  export DS_BUILD_FUSED_ADAM=1
  export DS_BUILD_UTILS=1

  pip3 install --no-cache-dir DeepSpeed=0.9.1

  # add this to .../deepspeed/launcher/launch.py l.219
  var='    args.node_rank=int(os.environ.get("SLURM_PROCID",0))'
  sed -i "219s|.*|$var|" $PWD/envAI_${sysN}/lib/python${pver}/site-packages/deepspeed/launcher/launch.py
fi

# install heat
if [ -d "$PWD/envAI_${sysN}/lib/python${pver}/site-packages/heat" ]; then
  echo 'HeAT already installed'
  echo
else
  export CFLAGS="-noswitcherror"
  export CXXFLAGS="-noswitcherror"

  pip3 install heat[hdf5,netcdf]
fi

# get rest of the libraries
# install rest
pip3 install -r reqs.txt --ignore-installed

# modify l.4 of /torchnlp/_third_party/weighted_random_sampler.py
var='int_classes = int'
sed -i "4s|.*|$var|" \
  $PWD/envAI_${sysN}/lib/python${pver}/site-packages/torchnlp/_third_party/weighted_random_sampler.py

# fix IB IP config
if [ -f "$PWD/envAI_${sysN}/bin/torchrun" ]; then
  sed -i -e '3,8s/^/#/' $PWD/envAI_${sysN}/bin/torchrun
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
""" >> $PWD/envAI_${sysN}/bin/torchrun
fi

#eof
