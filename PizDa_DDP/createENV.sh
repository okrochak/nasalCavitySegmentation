#!/bin/bash
# -*- coding: utf-8 -*-
# author: EI
# version: 211014a
# creates python env

# parameters
nameE='envAI_PD'

# set modules
module load daint-gpu
module load PyTorch
module load HDF5

# get pwd 
cDir=$PWD

# create env
python3 -m venv $nameE 

# activate env
source $nameE/bin/activate

# install torch+torchvision w/ cuda
pip3 install -r reqs.txt --force-reinstall

# install JUBE
if [ -f "$cDir/$nameE/bin/jube" ];then
  echo 'JUBE is already installed!'
else
  wget http://apps.fz-juelich.de/jsc/jube/jube2/download.php?version=2.4.1 -O JUBE-2.4.1.tar.gz
  tar xzf JUBE-2.4.1.tar.gz
  cd JUBE-2.4.1
  python3 setup.py install
  cd ..
  rm -rf JUBE-2.4.1 JUBE-2.4.1.tar.gz
fi

echo
echo
echo "a $nameE env is created in $cDir"
echo "source this env as:"
echo "source $nameE/bin/activate"

#eof
