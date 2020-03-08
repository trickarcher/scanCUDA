cmake_minimum_required(VERSION 3.8)
project (scan CXX CUDA)

include_directories(include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_library(awkward_array_scan
        src/awkward_offset_arrays.cu
        src/awkward_offset_arrays.cpp)

target_compile_features(awkward_array_scan PUBLIC cxx_std_11)

add_executable(scan main.cu)

set_target_properties(scan PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(scan PRIVATE awkward_array_scan)


