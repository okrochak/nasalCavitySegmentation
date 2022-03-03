# DL using DDP on deepv

# source
https://github.com/pytorch/pytorch#from-source

# current isues
1. dirty fix to infiniband IPs\
https://github.com/pytorch/pytorch/issues/73656

# to-do
1.

# done
1. CUDA is back!
2. connection issues are solved
3. updated to torch 1.10.0
4. updated to torch 1.10.2
5. infiniband IPs updated

# usage - pip
1. clone
2. run `./createENV.sh`
3. submit `sbatch DDP_startscript_deep.sh`

# usage - conda
1. clone
2. run `./conda_torch.sh`
3. modify `DDP_startscript_deep.sh`\
comment out previous source\
`source /p/project/prcoe12/RAISE/envAI_deepv/bin/activate`\
uncomment:\
`source /p/project/prcoe12/RAISE/miniconda3_deepv/etc/profile.d/conda.sh`\
`conda activate`
4. submit `sbatch DDP_startscript_deep.sh`

# updates
1. with the new Stage2020, Conda is no longer needed! Simply use the envAI_deep as:\
`ml use $OTHERSTAGES`\
`ml Stages/2022 GCC OpenMPI Python cuDNN NCCL Python`\
`source /p/project/prcoe12/RAISE/envAI_deepv/bin/activate`
2. shared memory type performance increase is adapted, simply increase `--cpus-per-task`
3. migrated to OpenMPI (pscom issues) and updated to IB IPs
