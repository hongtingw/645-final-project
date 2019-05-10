cmake_minimum_required(VERSION 3.13)
project(lenet_inference)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

add_executable(lenet_inference lenet_main.cpp MnistReader.cpp InferenceEngine.cpp GreyscaleImage.cpp)