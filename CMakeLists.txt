cmake_minimum_required(VERSION 3.18)

project(hpps LANGUAGES CXX CUDA)

enable_language(CUDA)

set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(${PROJECT_NAME}
    test.cpp
    kernel.cu
    benchmark.cu
    config.h
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -ffast-math>
    $<$<COMPILE_LANGUAGE:CUDA>:-ffast-math -O3>
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)