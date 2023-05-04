#!/bin/bash
ml --force purge

ml Stages/2023  GCC/11.3.0  OpenMPI/4.1.4 PyTorch/1.12.0-CUDA-11.7 torchvision/0.13.1-CUDA-11.7

python3 -m venv ray_tune_env

source ray_tune_env/bin/activate

pip3 install ray==2.4.0 ray[tune]==2.4.0
pip3 install python-dateutil pytz typing-extensions
pip3 install hpbandster ConfigSpace
deactivate