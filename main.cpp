#include <iostream>
#include <cuda_runtime.h>

extern "C" __global__ void block_wide_prefix_sum_naive(float* A, uint32_t n);

constexpr uint32_t BLOCK_SIZE= 64;

size_t get_alloc_size(uint32_t n_floats){
    size_t floats_per_cacheline= BLOCK_SIZE/sizeof(float);
    size_t blocks_to_alloc= (n_floats + floats_per_cacheline - 1) / floats_per_cacheline;
    return blocks_to_alloc*BLOCK_SIZE;
}

int main(int argc, char** argv){

    int size = std::atoi(argv[1]);
    std::cout << "Array size: " << size << std::endl;

    size_t alloc_size_bytes= get_alloc_size(size);
    float* A= static_cast<float*>(aligned_alloc(32, alloc_size_bytes));

    for (int i = 0; i < size; i++){
        A[i]= static_cast<float>(i);
    }

    float* A_device;
    cudaMalloc(&A_device, alloc_size_bytes);
    cudaMemcpy(A_device, A, alloc_size_bytes, cudaMemcpyHostToDevice);

    block_wide_prefix_sum_naive<<1, BLOCK_SIZE>>(A_device, size);
    cudaDeviceSynchronize();

    cudaMemcpy(A, A_device, alloc_size_bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++){
        std::cout << A[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(A_device);
    free(A);

    return 0;
}