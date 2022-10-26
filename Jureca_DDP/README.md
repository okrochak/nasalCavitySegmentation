# DL using DDP on jureca dc gpu

# DDP source
https://github.com/pytorch/pytorch#from-source

# jureca user documentation
https://apps.fz-juelich.de/jsc/hps/jureca/index.html

# current isues
1. torchrun: Hostname/endpoint mismatch not handled\
workaround is to modify torchrun and use included batch script\
simply run `createEnv.sh` to install fixed torch\
discussion in: https://github.com/pytorch/pytorch/issues/73656

# to-do
1.

# done
1. 

# usage
1. clone
2. run `./createEnv.sh` to create env and install torch 
3. run `./createEnv_MPI.sh` to create Conda env and install torch with MPI support
4. submit `sbatch DDP_startscript.sh`
