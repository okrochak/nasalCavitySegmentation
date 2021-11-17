#!/bin/bash
# -*- coding: utf-8 -*-
# author: EI
# version: 210827a
# install wheels and JUBE

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

# install torch
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

#eof
