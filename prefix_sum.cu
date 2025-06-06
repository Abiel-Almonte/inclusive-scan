#include <stdint.h>

constexpr uint32_t BLOCKDIM= 64;

__global__ void block_wide_prefix_sum_naive(float* A, uint32_t n){
    uint32_t index= threadIdx.y * BLOCKDIM  + threadIdx.x;
    
    if (index >= n){
        return;
    }

    float sum= 0.0f;
    for (uint32_t i= 0; i <= index; i++){
        sum+= A[i];
    }

    A[index]= sum;
}
extern "C" {
    void launch_block_wide_prefix_sum_naive(float* A, uint32_t n, size_t bytes){
        float* A_device;
        
        cudaMalloc(&A_device, bytes);
        cudaMemcpy(A_device, A, bytes, cudaMemcpyHostToDevice);

        block_wide_prefix_sum_naive<<<1, BLOCKDIM>>>(A_device, n);

        cudaDeviceSynchronize();

        cudaMemcpy(A, A_device, bytes, cudaMemcpyDeviceToHost);
        cudaFree(A_device);
    }
}