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
2. for containers, instead of #1, use `fixed_torch_run.py` -- follow usage - containers.

# to-do
1. 

# done
1. tested containers (for both NVIDIA & AMD GPUs):\
https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch \
https://www.amd.com/en/technologies/infinity-hub/pytorch \
https://hub.docker.com/r/rocm/pytorch
 

# usage - Python Env
1. run `./createEnv.sh` to create env and install torch 
2. select a case from CASES folder 
3. submit `sbatch DDP_startscript.sh`

# usage - containers (note this for AMD partition - modify for NVIDIA)
1. run `./createContainer.sh` to use and build Torch/ROCm container
2. select a case from CASES folder 
3. submit `sbatch DDP_startscript_container.sh`

# usage - Source Code
1. run `./createEnv_MPI.sh` to create Conda env and install torch with MPI support
2. select a case from CASES folder 
3. submit `sbatch DDP_startscript.sh`
