#!/bin/sh
# author: EI
# version: 210701a

module --force purge
module use $OTHERSTAGES 
ml Stages/2020 GCC ParaStationMPI/5.4.7-1-mt 
ml Python CMake NCCL mpi4py netCDF parallel-netcdf SciPy-Stack

# activate virtual python enviornment
source /p/project/prcoe12/RAISE/envAI_deepv/bin/activate

# install
pip install 'heat[hdf5,netcdf]'

#eof
