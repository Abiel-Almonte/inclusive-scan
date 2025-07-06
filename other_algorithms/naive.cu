#include <stdint.h>

__global__ void naive_prefix_sum(float* A, float* out, uint32_t n){
    uint32_t index= blockIdx.x * blockDim.x  + threadIdx.x;
    
    if (index >= n){
        return;
    }

    float sum= 0.0f;
    for (uint32_t i= 0; i < index; i++){
        sum+= A[i];
    }

    out[index] = sum;
}
