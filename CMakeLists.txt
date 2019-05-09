cmake_minimum_required(VERSION 3.13)
project(lenet_inference)

set(CMAKE_CXX_STANDARD 17)

add_library(mnist_reader MnistReader.cpp)
add_library(engine InferenceEngine.cpp)

add_executable(lenet_inference lenet_main.cpp)
target_link_libraries(lenet_inference engine mnist_reader)