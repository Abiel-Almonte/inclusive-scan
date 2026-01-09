cmake_minimum_required(VERSION 3.20)
project(FastPrefixSum LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

add_executable(benchmark benchmark.cu)
add_executable(validate validate.cu)

target_include_directories(benchmark PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}" 
    "${CMAKE_CURRENT_SOURCE_DIR}/generated"
    "${CMAKE_CURRENT_SOURCE_DIR}/best_kernel"
)

target_compile_definitions(benchmark PRIVATE "KERNEL_HEADER_PATH=\"${KERNEL_HEADER_FILENAME}\"")

target_include_directories(validate PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(validate PRIVATE "KERNEL_HEADER_PATH=\"${KERNEL_HEADER_FILENAME}\"")

set_target_properties(benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(validate PROPERTIES CUDA_SEPARABLE_COMPILATION ON)