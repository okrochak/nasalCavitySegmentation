# DL using HeAT/PyTorch on CTEAMD

# source
https://github.com/helmholtz-analytics/heat

# nodes
1. CTEAMD limits outgoing comm -- workaround is to link local machine with CTEAMD using\
`sshfs -o workaround=rename <user_name>@dt01.bsc.es: <local_folder>`\
use git commands in <local_folder>
2. To add new python libraries, download wheels (.whl) or tarbals from https://pypi.org/ (only cp39 works)\
from local machine to CTEAMD, use method above in #1\
copy the item to wheels folder (/gpfs/projects/bsc21/bsc21163/wheels)\
update the `regs.txt` file in the project folder (/gpfs/projects/bsc21/bsc21163)\
run ./installWheels.sh in the project folder
3. TBL datasets are moved to `/gpfs/scratch/bsc21/bsc21163/RAISE_Dataset/T31/`

# current isues
1. concat is not working 
2. parallel helpers are not working 
3. dataloader for .h5?

# to-do
1. 

# usage
add these commands to your batch script (on CTEAMD):\
`ml gcc openmpi rocm python`\
`source /gpfs/projects/bsc21/bsc21163/envAI_BSC/bin/activate`\
`export LD_LIBRARY_PATH=/gpfs/projects/bsc21/bsc21163/envAI_BSC/lib:$LD_LIBRARY_PATH`\

# includes
1. batch script `heat_startscript.sh`
3. example script 1 `heat_pytorch_mnist.py`

# install (opt.)
1. modify setup.py for HeAT to accept torch/torchvision with 1.9.0/1.10
2. run `./createENV.sh`
