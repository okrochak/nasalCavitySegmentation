# DL using Horovod on deepv

# source
https://github.com/horovod/horovod

# current isues
1. 

# to-do
1. 

# usage
add these commands to your batch script (on deepv):\
`module --force purge`\
`module use $OTHERSTAGES`\
`ml Stages/2020 GCC ParaStationMPI/5.4.7-1-mt Python`\
`source /p/project/prcoe12/RAISE/envAI_deepv/bin/activate`

# install (opt.)
1. clone
2. run `./install_horovod.sh`
3. submit sbatch `horovod_startscript.sh`
