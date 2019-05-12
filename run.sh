#!/usr/bin/env bash
cd "$( dirname "${BASH_SOURCE[0]}" )"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
if [[ $? -eq 0 ]]; then
    cd ..
    ./build/lenet_inference opencv --logtostderr=1
else
    cd ..
fi