ml --force purge
ml Stages/2022  GCC/11.2.0 CUDA/11.5 Python/3.9.6 PyTorch/1.11-CUDA-11.5 torchvision/0.12.0

## create vritual environment
python3 -m venv ddp_ray_env

source ddp_ray_env/bin/activate

# RAY TUNE 2.0 NOT WORKING
pip3 install ray==1.9.0 ray[tune]==1.9.0 ray[train]==1.9.0


# might be necessay, might be not
pip3 install requests
pip3 install pytz
pip3 install python-dateutil
