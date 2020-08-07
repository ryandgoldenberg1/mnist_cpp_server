#!/bin/bash

set -ex

rm -rf build
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/usr/local/libtorch ..
cmake --build . --config Release
