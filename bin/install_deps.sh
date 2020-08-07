#!/bin/bash
#
# Script for installing project dependencies.
# Written using Amazon Linux 2 AMI of EC2.

set -ex

# Python 3
yum install -y python3 python3-pip git

# Install Python dependencies
pip3 install --user --no-cache-dir -r requirements.txt

# Install C++ dependencies
# g++ and OpenSSL
yum groupinstall -y "Development Tools"
yum install -y openssl-devel
mkdir tmp && cd tmp
# CMake https://cmake.org/install/
wget https://github.com/Kitware/CMake/releases/download/v3.18.1/cmake-3.18.1.tar.gz
tar -xzvf cmake-3.18.1.tar.gz
cd cmake-3.18.1/
./bootstrap && make && sudo make install
cd ..
bash
# nlohmann/json https://github.com/nlohmann/json
git clone https://github.com/nlohmann/json
cd json
mkdir build && cd build
cmake .. && make && sudo make install
cd ../..
# cpp-httplib
git clone https://github.com/yhirose/cpp-httplib.git
cd cpp-httplib
mkdir build && cd build
cmake .. && make && sudo make install
cd ../..
# LibTorch
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.6.0+cpu.zip
sudo mv libtorch /usr/local/
# cleanup
cd ..
rm -rf tmp
