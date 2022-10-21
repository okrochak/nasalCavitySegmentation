# DL using Horovod on CTEAMD

# source
https://github.com/horovod/horovod

# notes
1. CTEAMD limits outgoing comm -- workaround is to link local machine with CTEAMD using\
`sshfs -o workaround=rename <user_name>@dt01.bsc.es: <local_folder>`\
use git commands in <local_folder>
2. to setup Horovod \
`pip download -r reqHor.txt`\
 on your local machine to get the necessary wheels, and transfer them to CTEAMD with method in #1
3. TBL datasets are moved to `/gpfs/scratch/bsc21/bsc21163/RAISE_Dataset/T31/`

# current isues
1. Horovod 0.26.1 has issues when compiling, waiting for a fix

# to-do
1. 

# updates
1. updated to ROCm 5.1.1

# usage
add these commands to your batch script (on CTEAMD):\
`ml gcc openmpi rocm python`\
`source /gpfs/projects/bsc21/bsc21163/envAI_BSC/bin/activate`\
`export LD_LIBRARY_PATH=/gpfs/projects/bsc21/bsc21163/envAI_BSC/lib:$LD_LIBRARY_PATH`\
`export MIOPEN_DEBUG_DISABLE_FIND_DB=1`

# install (opt.)
1. apply this fix to untared Horovod: \
https://github.com/horovod/horovod/pull/3588/commits/141f41ea93731b9e3d2d7ecbac70fc10e0a8ec7e \
tar the modified Horovod and install
2. run `./createENV.sh`
