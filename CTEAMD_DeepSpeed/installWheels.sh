#!/bin/bash
# -*- coding: utf-8 -*-
# author: EI
# version: 211025a
# install wheels, JUBE, torch and DeepSpeed

# parameters
nameE=envAI_BSC

# get pwd 
cDir=$PWD
wDir=/gpfs/projects/bsc21/bsc21163/wheels/

# set modules
ml bsc gcc openmpi rocm python cmake

# activate env
source $cDir/$nameE/bin/activate
export LD_LIBRARY_PATH=$cDir/$nameE/lib:$LD_LIBRARY_PATH

# install torch for horovod
pip3.9 install --no-index --find-links $wDir -r reqs.txt

# install JUBE 
if [ -f "$cDir/$nameE/bin/jube" ];then
   echo 'JUBE is already installed!'
else
   cd $wDir
   tar xzf JUBE-2.4.1.tar.gz
   cd JUBE-2.4.1
   python3.9 setup.py install
   cd ..
   rm -rf JUBE-2.4.1
fi

# install deepspeed/torch 
# export DS_BUILD_OPS=1 # not working?? recursion error
export DS_BUILD_FUSED_ADAM=1
export DS_BUILD_UTILS=1
pip3.9 install --no-cache-dir --no-index --find-links $wDir $wDir/DeepSpeed_rocm.tar.gz --force-reinstall

#add this to .../deepspeed/launcher/launch.py l.70
var='    args.node_rank=int(os.environ.get("SLURM_PROCID",0))'
sed -i "70s|.*|$var|" $cDir/$nameE/lib/python3.9/site-packages/deepspeed/launcher/launch.py

#eof
