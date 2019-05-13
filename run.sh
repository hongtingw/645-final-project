#!/usr/bin/env bash
cd "$( dirname "${BASH_SOURCE[0]}" )"
mkdir -p build
cd build
cmake -DCMAKE_MODULE_PATH=/home/ubuntu/645-final-project/cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
if [[ $? -eq 0 ]]; then
    cd ..
    echo "Testing sequential pipeline..."
    ./build/lenet_inference --logtostderr=1 --mat_op_impl=naive --pipeline_type=seq
    ./build/lenet_inference --logtostderr=1 --mat_op_impl=opencv --pipeline_type=seq
    ./build/lenet_inference --logtostderr=1 --mat_op_impl=opt --pipeline_type=seq
    echo "Testing prefetching pipeline..."
    ./build/lenet_inference --logtostderr=1 --mat_op_impl=naive --pipeline_type=prefetch
    ./build/lenet_inference --logtostderr=1 --mat_op_impl=opencv --pipeline_type=prefetch
    ./build/lenet_inference --logtostderr=1 --mat_op_impl=opt --pipeline_type=prefetch
    echo "Testing sequential pipeline with 100us additional preprocessing time..."
    ./build/lenet_inference --logtostderr=1 --mat_op_impl=naive --pipeline_type=seq --additional_preprocessing_time=100
    ./build/lenet_inference --logtostderr=1 --mat_op_impl=opencv --pipeline_type=seq --additional_preprocessing_time=100
    ./build/lenet_inference --logtostderr=1 --mat_op_impl=opt --pipeline_type=seq --additional_preprocessing_time=100
    echo "Testing prefetching pipeline with 100us additional preprocessing time..."
    ./build/lenet_inference --logtostderr=1 --mat_op_impl=naive --pipeline_type=prefetch --additional_preprocessing_time=100
    ./build/lenet_inference --logtostderr=1 --mat_op_impl=opencv --pipeline_type=prefetch --additional_preprocessing_time=100
    ./build/lenet_inference --logtostderr=1 --mat_op_impl=opt --pipeline_type=prefetch --additional_preprocessing_time=100
else
    cd ..
fi