#include <stdint.h>

constexpr uint32_t BLOCKDIM= 64;

extern "C" {
    __global__ void block_wide_prefix_sum_naive(float* A, uint32_t n){
        uint32_t index= threadIdx.y * BLOCKDIM  + threadIdx.x;
        
        if (index >= n){
            return;
        }

        float sum= 0.0f;
        for (uint32_t i= 0; i <= index; i++){
            sum+= A[i];
        }

        A[index]= sum
    }
}