#!/bin/bash
# -*- coding: utf-8 -*-
# author: EI
# version: 210827a
# creates python env

# parameters
nameE='envAI_BSC'

# set modules
ml bsc gcc openmpi rocm python

# get pwd 
cDir=$PWD

# create env
python -m venv $nameE 

# link missing libs
ln -s /usr/lib64/libtinfo.so.6 $nameE/lib/libtinfo.so.5
ln -s /apps/PYTHON/3.9.1/GCC/lib/libpython3.9.so $nameE/lib/libpython3.9.so

# link missing bins
cp /apps/PYTHON/3.9.1/GCC/bin/pip3.9 $cDir/$nameE/bin/
cp /apps/PYTHON/3.9.1/GCC/bin/easy_install-3.9 ./$nameE/bin/
ln -s $cDir/$nameE/bin/pip3.9 $cDir/$nameE/bin/pip3
ln -s $cDir/$nameE/bin/pip3.9 $cDir/$nameE/bin/pip
ln -s $cDir/$nameE/bin/easy_install-3.9 $cDir/$nameE/bin/easy_install

# modify headers 
var="#!$cDir/$nameE/bin/python3.9"
sed -i "1s|.*|$var|" $cDir/$nameE/bin/pip3.9
sed -i "1s|.*|$var|" $cDir/$nameE/bin/easy_install-3.9

# activate env
source $nameE/bin/activate
export LD_LIBRARY_PATH=${cDir}/$name/lib:$LD_LIBRARY_PATH

# install wheels
./installWheels.sh

echo
echo
echo "a $nameE env is created in $cDir"
echo "source this env as:"
echo "source $nameE/bin/activate"
echo export LD_LIBRARY_PATH=$cDir/$nameE/lib:'$LD_LIBRARY_PATH'

#eof
