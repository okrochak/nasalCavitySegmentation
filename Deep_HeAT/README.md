# DL using HeAT/PyTorch on deepv

# source
https://github.com/helmholtz-analytics/heat

# current isues
1. no concatenate dataset option

# to-do
1. find concatenate dataset option

# usage
add these commands to your batch script (on deepv):\
`module --force purge`\
`module use $OTHERSTAGES`\
`ml Stages/2020 GCC ParaStationMPI/5.4.7-1-mt Python NCCL mpi4py netCDF parallel-netcdf SciPy-Stack`\
`source /p/project/prcoe12/RAISE/envAI_deepv/bin/activate`

# install (opt.)
1. clone
2. run `./install_heat.sh`
3. submit `sbatch heat_startscript.sh`
