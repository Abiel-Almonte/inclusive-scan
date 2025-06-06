cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(hpps)
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(${PROJECT_NAME} 
    main.cpp
    prefix_sum.cu
)

target_compile_options(${PROJECT_NAME} PRIVATE 
    -O3 -march=native -ffast-math
)