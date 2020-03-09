cmake_minimum_required(VERSION 3.8)
project(scan CXX CUDA)

include_directories(include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_library(awkward_array_scan
        src/awkward_offset_arrays.cu
        src/awkward_offset_arrays.cpp)

target_compile_features(awkward_array_scan PUBLIC cxx_std_11)

add_executable(scan example.cpp)

set_target_properties(scan PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(scan PRIVATE awkward_array_scan)

################################
# GTest
################################
ADD_SUBDIRECTORY(ext/googletest)
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

################################
# Unit Tests
################################
# Add test cpp file
add_executable(tests tests/tests.cpp tests/scan_test.cpp)
# Link test executable against gtest & gtest_main
target_link_libraries(tests gtest gtest_main awkward_array_scan)
add_test(tests tests)


