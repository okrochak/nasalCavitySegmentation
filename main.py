import os
import numpy as np

import time

from omegaconf import DictConfig
import hydra

from funcs import segmentation_step, preprocessing_step, CNN_A, CNN_B, CNN_C, final_step
from cnn.conv import get_net_2D, get_net_3D, getLargestCC

@hydra.main(version_base=None, config_path=os.getcwd(), config_name="config.yaml")
def main(config: DictConfig) -> None:

    t1 = time.time()

    '''1. Segmentation'''
    image, voxel_size = segmentation_step(config=config)
    
    '''2. Preprocessing'''
    # TODO: define this in config file
    # Kernel for convolution filter
    k = np.array([[-1, -1, -1], 
                  [-1, 10, -1], 
                  [-1, -1, -1]])
    
    X = preprocessing_step(config=config, image=image, k=k)

    '''3a. CNN-A'''

    A_norm, A_norm_res, model_A, segmentation = CNN_A(config=config, X=X)

    '''3b. CNN-B'''

    B_norm, B_norm_res, model_B, min_coord = CNN_B(config=config, X=X, segmentation=segmentation)

    t2 = time.time()
    print("step (i)")
    print("{:5.3f}s".format(t2 - t1))

    '''3c. CNN-C'''

    t3 = time.time()

    C_norm_res, model_C, inlet_left, inlet_right = CNN_C(config=config, X=X, segmentation=segmentation, min_coord=min_coord, voxel_size=voxel_size)

    t4 = time.time()
    print("step (iii) 1")
    print("{:5.3f}s".format(t4 - t3))

    t5 = time.time()

    '''4. Final step'''
    final_step(config=config, X=X, segmentation=segmentation, min_coord=min_coord, voxel_size=voxel_size)

    t6 = time.time()
    print("step (ii)")
    print("{:5.3f}s".format(t6 - t5))

if __name__ == "__main__":
    main()
