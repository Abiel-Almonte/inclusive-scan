#include <stdint.h>

__global__ void blelloch_scan(float* arr, float* out, uint32_t n){
    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + tid;
    uint32_t tree_index = tid + 1; //assuming zero indexed array

    extern __shared__ float scratch[];
    scratch[tid] = (gid < n)? arr[gid] : 0.0f;

    uint32_t s_max = blockDim.x;

    __syncthreads();

    for (int32_t s = 1;  s <= s_max; s*=2){
        uint32_t parent = tree_index*2*s - 1;
        if (parent >= blockDim.x){
            continue;
        };

        uint32_t left_child = parent - s;

        scratch[parent] += scratch[left_child];
        __syncthreads();
    }

    if (tid == 0){
        scratch[blockDim.x -1] = 0.0f;
    }

    __syncthreads();


    for (int32_t s = s_max ;  s >= 1 ; s/=2){
        uint32_t parent = tree_index*2*s - 1;
        if (parent >= blockDim.x){
            continue;
        }
        uint32_t left_child =parent - s;

        float temp = scratch[parent];
        scratch[parent] += scratch[left_child];
        scratch[left_child] = temp;
        __syncthreads();
    }

    if (gid < n){
        out[gid] = scratch[tid];
    }
}