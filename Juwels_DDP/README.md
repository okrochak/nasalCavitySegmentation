# DL using DDP on juwels booster

# source
https://github.com/pytorch/pytorch#from-source

# current isues
1. Outdated -- please refer to Jureca 

# to-do
1.

# done
1. fixed local IPs for TCP

# usage
add these commands to your batch script (on juwels booster):\
`ml GCC ParaStationMPI cuDNN NCCL Python`\
`source /p/project/prcoe12/RAISE/envAI_juwels/bin/activate`

# install (opt.)
1. clone
2. run `./createEnv.sh` to create env and install torch
3. submit `sbatch DDP_startscript.sh`
