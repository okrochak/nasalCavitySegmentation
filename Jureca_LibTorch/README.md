# DL using LibTorch (C++ Torch)

# documentation 
https://github.com/pytorch/pytorch/blob/master/docs/libtorch.rst

# current isues
1. no distributed training 

# to-do
1. implement distributed training 

# done
1. as Python version is a wrapper, no performance difference
2. can simply be used alongisde a c++ code w/o Cpython
3. very limited compared to Python version (many classes/functions are missing) 

# usage
1. simply compile `mnist.cpp` using the `cmake` file as `bash compile.sh`
2. submit compiled `mnist` with `sbatch LibTorch_startscript.sh`
