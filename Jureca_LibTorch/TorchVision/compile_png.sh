# load libraries
ml NVHPC/22.3 CMake/3.21.1 cuDNN/8.3.1.22-CUDA-11.5

wget http://prdownloads.sourceforge.net/libpng/libpng-1.6.37.tar.gz?download
mv 'libpng-1.6.37.tar.gz?download'  libpng-1.6.37.tar.gz
tar xzf libpng-1.6.37.tar.gz

pushd libpng-1.6.37
rm -rf build
mkdir -p build
./configure --prefix=${PWD}/build
make
make install
popd
