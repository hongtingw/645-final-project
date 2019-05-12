#!/usr/bin/env bash
cd "$( dirname "${BASH_SOURCE[0]}" )"
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
if [[ $? -eq 0 ]]; then
    cd ..
    GLOG_logtostderr=1 ./build/lenet_inference opencv
else
    cd ..
fi