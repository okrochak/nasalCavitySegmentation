# benchmarks using JUBE
Old results, but still usable.\
Check new results from `Results` folder

## notes
benchmark using Synthetic Data for TBL\
check individual README files in each folder

# old results
1. Framework Comparison w/ NCCL (aka RCCL for AMD) on CTEAMD \
(conf: Dataset=TBL-small, Epoch=10, Learning Rate=0.01, Batch size=96):
![](bench_fw.png)
2. System Comparison w/ NCCL \
(conf: Dataset=MNISTx100, Epoch=10, Learning Rate=0.01, Batch Size=100):\
(note: AMD: CTEAMD // V100: DEEPEST // A100: JUWELS)
![](bench_system.png)
3. DDP/NCCL on JUWELS \
(conf: Dataset=TBL-small, Epoch=10, Learning Rate=0.01, Batch Size=100):
![](bench_id0_AT_juwels.png)
4. DDP/NCCL on JUWELS \
(conf: Dataset=TBL-small, Epoch=10, Learning Rate=scaled, Batch Size=100):
![](bench_id2_AT_juwels.png)
5. DDP/NCCL on JUWELS \
(conf: Dataset=MNISTx100, Epoch=10, Learning Rate=0.01, Batch Size=[10,100]):
![](bench_id0_MNIST_juwels.png)
6. DDP/MPI on DEEPEST \
(conf: Dataset=MNISTx100, Epoch=10, Learning Rate=0.01, Batch Size=[32,256]):
![](DDPPytorch/bench_id0_MNIST.png)
7. DDP/MPI on DEEPEST \
(conf: Dataset=TBL-small, Epoch=10, Learning Rate=0.01, Batch Size=[100,200]):
![](DDPPytorch/bench_id0_AT.png)
8. Horovod/MPI on DEEPEST \
(conf: Dataset=MNISTx100, Epoch=10, Learning Rate=0.01, Batch Size=[32,128]):
![](HoroPytorch/bench_id0_MNIST.png)
9. HeAT/MPI on DEEPEST \
(conf: Dataset=MNIST, Epoch=10, Learning Rate=0.01, Batch Size=[32,128]):
![](HeATPytorch/bench_id0.png)
