# Hyperparameter Optimization of Machine Learning Models with Ray Tune

For the optimization of the hyperparameters of neural networks (such as learning rate or batch size) or machine learning models in general, the Ray Tune library (current version supported is 2.4.0) can be used. The library features a smooth integration of PyTorch-based training scripts and enables two stages of parallelism:

- each training of a model with different hyperparameters (trial) can run in parallel on multiple GPUs (e.g. via PyTorch-DDP)
- several trials can run in parallel on an HPC machine (via Ray Tune itself)

For installation of Ray Tune, run the installation script

```bash
bash build_ray_env.py 
```

After installation, several example are available:

1. [Optimizing a ResNet18 on cifar-10 with AHSA or Random Search schedulers] 
2. [Optimizing a ResNet18 on cifar-10 with BOHB or Random Search schedulers]
3. [Optimizing a ResNet18 on cifar-10 with PBT or Random Search schedulers (including checkpointing)]


The ASHA scheduler is a variation of Random Search with early stopping of under-performing trials. The BOHB scheduler uses Bayesian Optimization in combination with early stopping, while the PBT scheduler uses evolutionary optimization and is well suited for optimizing non-stationary hyperparameters (such as learning rate schedules). 

The following parameters can be set for each script:

- num-samples: number of samples (trials) to evaluate
- max-iterations: for how long to train the trials at max
- par-workers: how many workers to allocate per trial
- scheduler: which scheduler to use
- data-dir: directory where the datasets are stored

To submit a job to the JURECA-DC-GPU machine, use the following command:

```bash
sbatch jureca_ray_startscript.sh 
```

For communication via the infiniband network it is important the specify the node ip-address in the startscript (whan launching Ray) in the following format:

```bash
--node-ip-address="$head_node"i
```

and 

```bash
--address "$head_node"i:"$port"
```

If multiple Ray instances run on the same machine, there might be problems if all use the same port value (7638), so it is advisable to change it to a different value in that case. 









