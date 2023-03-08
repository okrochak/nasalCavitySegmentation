# benchmark CNN with MNIST/RESNET/Autoencoder using JUBE
framework: Horovod\
backend: NCCL/MPI

# notes
benchmark using MNIST\
`jube run jobsys_mnist.xml`\
benchmark using Autoencoder for TBL\
`jube run jobsys_AT.xml`\
benchmark using ResNet\
`jube run jobsys_resnet.xml`\

# results
1. using MNIST:
![](bench_id0_MNIST.png)
2. using Autoencoder for TBL
![](bench_id0_AT.png)
