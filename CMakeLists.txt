cmake_minimum_required(VERSION 3.10)
project(KayenAI)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add compiler flags for optimization and warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O2")

# Source directories
include_directories(${CMAKE_SOURCE_DIR})

# Tensor library
add_library(tensor
    src/tensor/tensor.cpp
)

# Tensor tests
add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor tensor)

# Enable testing
enable_testing()
add_test(NAME TensorTests COMMAND test_tensor)
