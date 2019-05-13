#!/usr/bin/env bash
cd "$( dirname "${BASH_SOURCE[0]}" )"
mkdir -p build
cd build
cmake -DCMAKE_MODULE_PATH=/home/ubuntu/645-final-project/cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
if [[ $? -eq 0 ]]; then
    cd ..
    ./build/lenet_inference --logtostderr=1 --mat_op_impl=naive --pipeline_type=seq
    ./build/lenet_inference --logtostderr=1 --mat_op_impl=opencv --pipeline_type=seq
    ./build/lenet_inference --logtostderr=1 --mat_op_impl=opt --pipeline_type=seq
    ./build/lenet_inference --logtostderr=1 --mat_op_impl=naive --pipeline_type=prefetch
    ./build/lenet_inference --logtostderr=1 --mat_op_impl=opencv --pipeline_type=prefetch
    ./build/lenet_inference --logtostderr=1 --mat_op_impl=opt --pipeline_type=prefetch
else
    cd ..
fi