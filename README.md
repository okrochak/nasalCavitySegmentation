# Nasal cavity segmentation pipeline

This pipeline was originally developed my Dr. Mario Rüttgers and other collaborators, based on the following publication:  
Rüttgers, M., Waldmann, M., Schröder, W. et al.  
A machine-learning-based method for automatizing lattice-Boltzmann simulations of respiratory flows.  
Appl Intell 52, 9080–9100 (2022).  
https://doi.org/10.1007/s10489-021-02808-2 

## Prerequisites

This pipeline works with any modern Python installation. `uv` package manager is recommended for repository initialization.
Please note that if centerline based functionality of the pipeline is desired, VMTK package needs to be installed before running the pipeline. 
Please also make sure that the `vmtk` binary is included in the current path. 

## Execution

Please use `config.yaml` file to input the necessary variables before running the pipeline, such as the path to the input computed tomography files (in .dicom format).
Afterward, the pipeline can be executed by calling `uv run main.py` or `python main.py`. Several output files will be stored in the `output` folder, including the nasal cavity surface in `.stl` format. 
A sample dataset can be provided to test the pipeline by request to `o.krochak@fz-juelich.de`. 
