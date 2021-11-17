# DL using DDP on deepv

# source
https://github.com/pytorch/pytorch#from-source

# current isues
1.

# to-do
1.

# done
1. CUDA is back!
2. connection issues are solved
3. updated to torch 1.10.0

# usage
add these commands to your batch script (on deepv):\
`module --force purge`\
`module use $OTHERSTAGES`\
`ml Stages/2020 GCC ParaStationMPI/5.4.7-1-mt cuDNN NCCL mpi-settings/CUDA`\
`source /p/project/prcoe12/RAISE/miniconda3_deepv/etc/profile.d/conda.sh`\
`conda activate`

# install (opt.)
1. clone
2. run `./install_pyDDP.sh`
3. run `./createENV.sh`
4. submit `sbatch DDP_startscript_deep.sh`

# updates
1. with the new Stage2020, Conda is no longer needed! Simply use the envAI_deep as:\
`ml GCC ParaStationMPI/5.4.9-1-mt Python cuDNN NCCL Python`\
`source /p/project/prcoe12/RAISE/envAI_deepv/bin/activate`
2. shared memory type performance increase is adapted, simply increase `--cpus-per-task`
