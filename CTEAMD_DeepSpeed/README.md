# DL using DeepSpeed on CTEAMD

# source for DS with ROCm
https://github.com/ROCmSoftwarePlatform/DeepSpeed

# notes
1. DS for ROCm is forked from stable branch, hence, few versions below (currently 0.4.6)
2. CTEAMD limits outgoing comm -- workaround is to link local machine with CTEAMD using\
`sshfs -o workaround=rename <user_name>@dt01.bsc.es: <local_folder>`\
use git commands in <local_folder>
3. to add new python libraries; (a) download wheels (.whl) or tarbals from https://pypi.org/ (only cp39 works) from local machine to CTEAMD with method above in #1, (b) copy the item to wheels folder (/gpfs/projects/bsc21/bsc21163/wheels), (c) update the `regs.txt` file in the project folder (/gpfs/projects/bsc21/bsc21163), and (d) run ./installWheels.sh in the project folder
4. TBL datasets are moved to `/gpfs/scratch/bsc21/bsc21163/RAISE_Dataset/T31/`
5. DS with ROCm support is 2 major versions behind (as 21.10.22) \
versions DS-ROCm: 0.5.8, whereas DS-master: 0.7.3

# current isues
1. compiling DS with all options gives recursion error! Current options are fine.
2. cannot get accurate ranks from batch script, a manual fix is needed to the source files (check item #install) 

# to-do
1. 

# updates
1. updated to ROCm 5.1.1

# usage
check `DS_startscript.sh` to see what should be included to the batch script (on CTEAMD) - there are a lot of changes:\
make sure `DS_config.json` exist in your submit directory.

# install (opt.)
1. run `./createENV.sh`
2. add `args.node_rank=int(os.environ.get("SLURM_PROCID",0))` to l.72 of \
`$<env>/lib/python3.9/site-packages/deepspeed/launcher/launch.py`
