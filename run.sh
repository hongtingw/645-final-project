#!/usr/bin/env bash
cd "$( dirname "${BASH_SOURCE[0]}" )"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
cd ..
./build/lenet_main