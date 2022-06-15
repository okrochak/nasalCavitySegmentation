# DL using Graphcore IPU 

# Graphcore PyTorch documentation 
https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/pytorch_to_poptorch.html#

# jureca user documentation
https://apps.fz-juelich.de/jsc/hps/jureca/index.html

# current isues
1. no parallel training 

# to-do
1. implement parallelization 

# done
1. initial mnist tests show 8x better performance than A100 

# usage
apptainer is used for the containers
0. to use containers in Jureca, (if not done!) from JuDoor, click "Request access to restricted software", then "Access to other restricted software", and accept the agreement! ! finally, reset ssh
1. pull Graphcore SDK `apptainer pull poplar.sif docker://docker.io/graphcore/poplar:2.4.0` 
2. build Graphcore SDK with PyTorch `apptainer build pytorch.sif docker://docker.io/graphcore/pytorch` \
this comes with Torch-1.10.0
3. additional libraries are needed: \
`apptainer shell pytorch.sif`
`> pip3 install torchvision==1.11.0 tqdm h5py --user`
`> exit`
4. submit `sbatch GC_startscript.sh`
