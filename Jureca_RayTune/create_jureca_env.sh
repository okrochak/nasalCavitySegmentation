#!/bin/bash
ml --force purge

ml Stages/2022  GCC/11.2.0  OpenMPI/4.1.2 PyTorch/1.11-CUDA-11.5 torchvision/0.12.0-CUDA-11.5

python3 -m venv ray_tune_env

source ray_tune_env/bin/activate

pip3 install ray ray[tune]

deactivate