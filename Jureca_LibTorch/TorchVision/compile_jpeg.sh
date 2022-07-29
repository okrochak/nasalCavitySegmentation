# load libraries
ml NVHPC/22.3 CMake/3.21.1 cuDNN/8.3.1.22-CUDA-11.5

git clone https://github.com/winlibs/libjpeg.git 
cd libjpeg

rm -rf build
mkdir -p build 
mkdir -p install
pushd build
cmake -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=Release ..
make -j
make install
popd
