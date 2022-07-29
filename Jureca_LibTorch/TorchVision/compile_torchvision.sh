# compile torchvision for dataloading (optional)

# load libraries
ml NVHPC/22.3 CMake/3.21.1 cuDNN/8.3.1.22-CUDA-11.5

# get libtorch w/ gpu
wget https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.12.0%2Bcu116.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.12.0+cu116.zip
libtorch_dir=$PWD/libtorch

# get png packages for torchvision
./compile_png.sh
libpng_dir=$PWD/libpng-1.6.37/build

# get jpeg packages 
./compile_png.sh
libjpeg_dir=$PWD/libjpeg/install

# current dir
m_dir=$PWD

# get torchvision
git clone https://github.com/pytorch/vision.git

# compile torchvision
pushd torchvision
rm -rf build
mkdir -p build
mkdir -p install
cd build
cmake -DCMAKE_PREFIX_PATH=${libtorch_dir} \
        -DWITH_CUDA=on \
        -DPNG_LIBRARY=${libpng_dir}/lib/libpng.so \
        -DPNG_PNG_INCLUDE_DIR=${libpng_dir}/include \
        -DJPEG_LIBRARY=${libjpeg_dir}/lib64/libjpeg.so \
        -DJPEG_INCLUDE_DIR=${libjpeg_dir}/include \
        -DCMAKE_INSTALL_PREFIX=../install \
        -DCMAKE_BUILD_TYPE=Release ..

make -j
make install
popd

# eof 
