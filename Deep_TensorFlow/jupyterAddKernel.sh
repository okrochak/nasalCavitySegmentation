#!/bin/bash
# -*- coding: utf-8 -*-
# author: EI
# version: 220408a
# adds jupyter to an existing python env
# usage: bash jupyterAddKernel.sh <env_location>

# get sys info
sysN="$(uname -n | cut -f2- -d.)"
cDir=$PWD
ENV_LOC=$cDir/$1
export TMPDIR=$PWD
echo "system:${sysN}"
echo "env location: $ENV_LOC"
echo

# warn if wrong bash command 
if [ -z "$1" ];then
  echo 'wrong usage: try: bash jupyterAddKernel.sh <env_location>'
  exit
fi

# set modules
ml --force purge
if [ "$sysN" = 'deepv' ] ; then
  ml use $OTHERSTAGES
  ml Stages/2022 GCC OpenMPI cuDNN NCCL Python CMake
elif [ "$sysN" = 'jureca' ] ; then
  ml Stages/2022 GCC OpenMPI Python NCCL cuDNN libaio CMake
else
  echo
  echo 'unknown system detected'
  echo 'canceling'
  echo
fi
echo "modules loaded"
echo

# kernel info
KERNEL_NAME=envAI_jk_${sysN}
KERNEL_SPECS_PREFIX=${HOME}/.local
KERNEL_SPECS_DIR=${KERNEL_SPECS_PREFIX}/share/jupyter/kernels

# get python version
pver="$(python --version 2>&1 | awk {'print $2'} | cut -f1-2 -d.)"
echo "python version is ${pver}"
echo

# environment that jupyter is built on 
if [ -z "${ENV_LOC}" ];then
  echo 'env does not exist'
  echo 'usage: bash jupyterAddKernel.sh env_location'
  exit
else
  source ${ENV_LOC}/bin/activate
  export PYTHONPATH=${VIRTUAL_ENV}/lib/python${pver}/site-packages:${PYTHONPATH}
fi

# create/Edit launch script for the Jupyter kernel
if [ -f "${VIRTUAL_ENV}/kernel.sh" ];then
  echo "kernel.sh exist!"
else
  echo '#!/bin/bash'"
  
# Load basic Python module
ml --force purge
ml use $OTHERSTAGES
ml Stages/2022 GCC OpenMPI cuDNN NCCL Python CMake

# Activate your Python virtual environment
source ${VIRTUAL_ENV}/bin/activate
    
# Ensure python packages installed in the virtual environment are always prefered
export PYTHONPATH=${VIRTUAL_ENV}/lib/python${pver}/site-packages:"'${PYTHONPATH}'"

exec python3 -m ipykernel "'$@' > ${VIRTUAL_ENV}/kernel.sh
  chmod +x ${VIRTUAL_ENV}/kernel.sh
  
  echo 'kernel.sh:'
  cat ${VIRTUAL_ENV}/kernel.sh # double check
fi

# create Jupyter kernel configuration directory and files
pip3 install --ignore-installed ipykernel --no-cache-dir
${VIRTUAL_ENV}/bin/python3 -m ipykernel install --name=${KERNEL_NAME} --prefix ${VIRTUAL_ENV}
VIRTUAL_ENV_KERNELS=${VIRTUAL_ENV}/share/jupyter/kernels

# adjust kernel.json file
mv ${VIRTUAL_ENV_KERNELS}/${KERNEL_NAME}/kernel.json ${VIRTUAL_ENV_KERNELS}/${KERNEL_NAME}/kernel.json.orig # backup
echo '{
  "argv": [
    "'${VIRTUAL_ENV}/kernel.sh'",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "'${KERNEL_NAME}'",
  "language": "python"
}' > ${VIRTUAL_ENV_KERNELS}/${KERNEL_NAME}/kernel.json

# create link to kernel specs
mkdir -p ${KERNEL_SPECS_DIR}
cd ${KERNEL_SPECS_DIR}
ln -s ${VIRTUAL_ENV_KERNELS}/${KERNEL_NAME} .

echo -e "\n\nThe new kernel '${KERNEL_NAME}' was added to your kernels in '${KERNEL_SPECS_DIR}/'\n"

#eof
