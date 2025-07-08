#include <stdint.h>

__device__ float block_status[0]; //placeholder

__device__ inline float sequential_lookback(uint32_t tid, float block_prefix_sum) {
    float block_offset = 0.0f;

    if (tid == 0 && blockIdx.x > 0) {
        float accumulation = 0.0f;
#pragma unroll 1
        for (int prev = blockIdx.x - 1; prev >= 0; prev--) {
            while (isnan(block_status[prev])) {
                /* wait */
            }
            if (block_status[prev] < 0.0f) {
                accumulation -= block_status[prev];
            } else {
                accumulation += block_status[prev];
                break;
            }
        }
        block_offset = accumulation;
        block_status[blockIdx.x] = block_offset + block_prefix_sum; // prefix
        __threadfence();
    }

    return block_offset;
}