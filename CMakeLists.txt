cmake_minimum_required(VERSION 3.13)
project(lenet_inference)

add_subdirectory(json)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Enable SSE 4.2
if ((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    add_compile_options(-msse4.2)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    add_compile_options(/QxSSE4.2)
elseif ((CMAKE_CXX_COMPILER_ID MATCHES "MSVC") AND NOT CMAKE_CL_64)
    add_compile_options(/arch:SSE2) # VC doesn't support SSE4.2
endif ()

include("cmake/FindGlog.cmake")

find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)

message(STATUS ${GLOG_INCLUDE_DIRS})
include_directories(BEFORE ${GLOG_INCLUDE_DIRS})

add_executable(lenet_inference lenet_main.cpp MnistReader.cpp InferenceEngine.cpp
        blas/Blas.cpp
        layers/DenseLayer.cpp layers/FlattenLayer.cpp layers/ConvLayer.cpp)
target_link_libraries(lenet_inference PRIVATE ${OpenCV_LIBS} ${GLOG_LIBRARIES} nlohmann_json::nlohmann_json)