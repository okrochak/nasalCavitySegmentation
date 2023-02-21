# DL using DDP on LUME partition-G / eap

# DDP source
https://github.com/pytorch/pytorch#from-source

# LUMI user documentation
https://docs.lumi-supercomputer.eu/software/packages/pytorch/

# CSC support
https://docs.csc.fi/apps/pytorch/

# current isues
1. SQL DB file lock?
2. no apptainer (singularity is too old)
3. all_reduce is not working on LUMI example in \
https://github.com/Lumi-supercomputer/lumi-reframe-tests/tree/main/checks/apps/deeplearning/pytorch/src

# to-do
1. further tests

# done
1. initial runs (with issues)

# usage - container
1. run `./container_build.sh` to use and build Torch/ROCm container
2. select a case from CASES folder 
3. modify and submit `sbatch container_startscript.sh`

# usage - Python ENV
1. run `./env_build.sh` to use and build Torch/ROCm container
2. select a case from CASES folder 
3. modify and submit `sbatch env_startscript.sh`
