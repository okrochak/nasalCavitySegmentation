# compile mnist.cpp with latest LibTorch

# load libraries
ml NVHPC/22.3 CMake/3.21.1 cuDNN/8.3.1.22-CUDA-11.5 Python/3.9.6

# get libtorch w/ gpu
wget https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.12.0%2Bcu116.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.12.0+cu116.zip
libtorch_dir=$PWD/libtorch

# compile mnist.cpp with libtorch w/ gpu to build folder
mkdir -p build
pushd build
cmake -DCMAKE_PREFIX_PATH=${libtorch_dir} -DDOWNLOAD_MNIST=ON ..
cmake --build . --config Release
mv mnist ..
popd

# eof
