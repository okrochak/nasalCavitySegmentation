#!/bin/bash -x
echo "Load modules..."
module --force purge
module use $OTHERSTAGES
module load Stages/2023
module load GCCcore/.11.3.0
module load Python/3.10.4
module load GCC/11.3.0 ParaStationMPI/5.8.1-1
module load TensorFlow
module load VTK/9.2.5
module load scikit-image/0.19.3
module load scikit-learn/1.1.2
module load ParaView/5.11.0-EGL

SCRIPT_PATH=/p/scratch/cjhpc54/$USER/CT_to_STL_2023

source ${SCRIPT_PATH}/v_env_jw_2023/bin/activate

python ${SCRIPT_PATH}/1_seg.py
