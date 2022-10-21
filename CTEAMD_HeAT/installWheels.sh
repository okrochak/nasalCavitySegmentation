#!/bin/bash
# -*- coding: utf-8 -*-
# author: EI
# version: 221019a
# install wheels, JUBE, torch, DDP, horovod, heat

# parameters
nameE=envAI_BSC

# get pwd 
cDir=$PWD
wDir=/gpfs/projects/bsc21/bsc21163/wheels

# get python version
pver="$(python --version 2>&1 | awk {'print $2'} | cut -f1-2 -d.)"
echo "python version is ${pver}"
echo

# install torch
if [ -f "${cDir}/${nameE}/bin/torchrun" ]; then
  echo 'Torch already installed'
  echo
else
  pip${pver} install --no-index --find-links $wDir -r reqs.txt --ignore-installed
fi

# install horovod/torch
if [ -f "${cDir}/${nameE}/bin/torchrun" ]; then
  echo 'Horovod already installed'
  echo
else
  # important fix for hipify (solved on Horovod 0.26.1 - but not stable) 
  # https://github.com/horovod/horovod/pull/3588/commits/141f41ea93731b9e3d2d7ecbac70fc10e0a8ec7e
  
  # install horovod/torch separately w/ these options
  export HOROVOD_WITH_MPI=1
  export HOROVOD_MPI_THREADS_DISABLE=1
  export HOROVOD_GPU=ROCM
  export HOROVOD_GPU_OPERATIONS=NCCL
  export HOROVOD_WITH_PYTORCH=1 
  export HOROVOD_WITHOUT_TENSORFLOW=1
  export HOROVOD_WITHOUT_MXNET=1
  pip${pver} install --no-cache-dir --no-index --find-links \
          $wDir $wDir/horovod-0.25.0_mod.tar.gz 
fi

# install deepspeed/torch 
if [ -f "$cDir/$nameE/bin/deepspeed" ];then
  echo 'Deepspeed is already installed!'
else
  # export DS_BUILD_OPS=1
  export DS_BUILD_FUSED_ADAM=1
  export DS_BUILD_UTILS=1

  pip${pver} install --no-cache-dir --no-index --find-links \
          $wDir $wDir/DeepSpeed_rocm.tar.gz 

  #add this to .../deepspeed/launcher/launch.py l.70
  var='    args.node_rank=int(os.environ.get("SLURM_PROCID",0))'
  sed -i "70s|.*|$var|" $cDir/$nameE/lib/python${pver}/site-packages/deepspeed/launcher/launch.py
fi

# install heat
if [ -d "$cDir/$nameE/lib/python${pver}/site-packages/heat" ];then
  echo 'HeAT is already installed!'
else
  pip${pver} install --no-cache-dir --no-index --find-links \
          $wDir $wDir/heat-1.2.0_mod.tar.gz[hdf5,netcdf] #--force-reinstall
fi

# fix issue with torchnlp 
if [ -f "$cDir/${nameE}/lib/python${pver}/site-packages/torchnlp/_third_party/weighted_random_sampler.py" ] ; then
  # modify l.4 of /torchnlp/_third_party/weighted_random_sampler.py
  var='int_classes = int'
  sed -i "4s|.*|$var|" \
    $cDir/${nameE}/lib/python${pver}/site-packages/torchnlp/_third_party/weighted_random_sampler.py
fi

# install JUBE 
if [ -f "$cDir/$nameE/bin/jube" ];then
  echo 'JUBE is already installed!'
else
  cd $wDir
  tar xzf JUBE-2.4.1.tar.gz
  cd JUBE-2.4.1
  python${pver} setup.py install
  cd ..
  rm -rf JUBE-2.4.1
fi

#eof
