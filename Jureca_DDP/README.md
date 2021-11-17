# DL using DDP on jureca dc gpu

# DDP source
https://github.com/pytorch/pytorch#from-source

# jureca user documentation
https://apps.fz-juelich.de/jsc/hps/jureca/index.html

# current isues
1.

# to-do
1.

# done
1. 

# usage
add these commands to your batch script (on juwels booster):\
`ml GCC ParaStationMPI Python cuDNN NCCL libaio`\
`source /p/project/raise-ctp1/RAISE/envAI_jureca/bin/activate`

# install (opt.)
1. clone
2. run `./createEnv.sh` to create env and install torch
3. submit `sbatch DDP_startscript.sh`
