# DL using DDP on juwels booster

### DDP source
https://github.com/pytorch/pytorch#from-source

### juwels documentation
https://apps.fz-juelich.de/jsc/hps/juwels/index.html

### current isues
1. torchrun: Hostname/endpoint mismatch not handled\
workaround is to modify torchrun and use included batch script\
simply run `createEnv.sh` to install fixed torch\
discussion in: https://github.com/pytorch/pytorch/issues/73656
2. for containers, instead of #1, use `fixed_torch_run.py` -- follow usage - containers.

### to-do
1.

### done
1. fixed local IPs for TCP
2. tested containers \
https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
3. Scale up to 2400 GPUs using NCCL backend

### usage - Python Env
1. run `./env_build.sh` to create env and install torch
2. select a case from CASES folder
3. submit `sbatch env_batch.sh`

### usage - containers
1. run `./container_build.sh` to build .sif
2. select a case from CASES folder
3. submit `sbatch container_batch.sh`
