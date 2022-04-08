#!/bin/bash
# -*- coding: utf-8 -*-
# author: EI
# version: 220408a
# creates machine specific jupyter kernel

# get sys info
sysN="$(uname -n | cut -f2- -d.)"
cDir=$PWD
export TMPDIR=$PWD
echo "system:${sysN}"
echo

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
KERNEL_NAME=kernel_${sysN}
KERNEL_SPECS_PREFIX=${HOME}/.local
KERNEL_SPECS_DIR=${KERNEL_SPECS_PREFIX}/share/jupyter/kernels

# get python version
pver="$(python --version 2>&1 | awk {'print $2'} | cut -f1-2 -d.)"
echo "python version is ${pver}"
echo

# create and activate a virtual environment for the kernel
if [ -d "${cDir}/kernelAI_${sysN}" ];then
  echo 'env already existi:'

  source ${cDir}/kernelAI_${sysN}/bin/activate
  export PYTHONPATH=${VIRTUAL_ENV}/lib/python${pver}/site-packages:${PYTHONPATH}
else
  # create env
  python3 -m venv --system-site-packages ${cDir}/kernelAI_${sysN}

  # get headers for pip
  if [ -f "${cDir}/kernelAI_${sysN}/bin/pip3" ]; then
    echo 'pip already exist'
    echo
  else
    cp "$(which pip3)" $cDir/kernelAI_${sysN}/bin/
    ln -s $cDir/kernelAI_${sysN}/bin/pip3 $cDir/kernelAI_${sysN}/bin/pip${pver}
    var="#!$cDir/kernelAI_${sysN}/bin/python${pver}"
    sed -i "1s|.*|$var|" $cDir/kernelAI_${sysN}/bin/pip3
  fi

  # activate env
  source ${cDir}/kernelAI_${sysN}/bin/activate
  export PYTHONPATH=${VIRTUAL_ENV}/lib/python${pver}/site-packages:${PYTHONPATH}
fi
echo 'location of new venv:'
echo ${VIRTUAL_ENV} # double check
echo

# create/Edit launch script for the Jupyter kernel
if [ -f "${VIRTUAL_ENV}/kernel.sh" ];then
  echo "kernel.sh exist!"
else
  echo '#!/bin/bash'"
  
# Load basic Python module
ml GCC ParaStationMPI Python 

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

echo 'load this env as:
ml --force purge
ml use $OTHERSTAGES
ml Stages/2022 GCC OpenMPI cuDNN NCCL Python CMake
source ${cDir}/kernelAI_${sysN}/bin/activate'

#eof
