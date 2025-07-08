#include <stdint.h>
#include "config.h"

__device__ float block_status[MAXBLOCKS];
__device__ inline float ks_warp_scan(float parent, uint32_t lane, uint32_t wid, float warp_offsets[]);
__device__ inline void ks_cross_warp_scan(uint32_t wid, uint32_t lane, float warp_sums[]);
__device__ inline void blelloch_shmem_scan(uint32_t tid, uint32_t n_warps, float warp_sums[]);
__device__ inline float sequential_lookback(uint32_t tid, float block_prefix_sum);
__device__ inline float parallel_lookback(uint32_t tid, uint32_t lane, float block_prefix_sum);


/*
 * Single pass device wide parallel inclusive scan aligned for 32x
 * 
 * Kogge-Stone warp scan and cross warp scan
 * Merrill & Garland's decoupled lookback
 */
extern "C" __global__ void single_pass_scan(float* A, float* B, uint32_t N) {
    uint32_t tid = threadIdx.x;
    uint16_t bdim = blockDim.x;
    uint32_t gid = blockIdx.x * bdim + tid;

    if (tid == 0) {
        block_status[blockIdx.x] = NAN; // invalid
    }
    __syncthreads();

    uint32_t wid = tid / WARPSIZE;
    uint32_t lane = tid % WARPSIZE;
    uint32_t n_warps = (bdim + WARPSIZE - 1) / WARPSIZE;

    float x = __ldg(&A[gid]);
    
    __shared__ float warp_sums[WARPSIZE];
    x = ks_warp_scan(x, lane, wid, warp_sums); 
    __syncthreads();
    ks_cross_warp_scan(wid, lane, warp_sums);
    
    __syncthreads();
    
    if (wid > 0){
        x +=  warp_sums[wid -1];
    }
    
    __shared__ float block_prefix_sum;
    if (tid == 0) {
        if (blockIdx.x == 0){
            block_status[blockIdx.x] = block_prefix_sum;
        }else {
            block_status[blockIdx.x] = -block_prefix_sum;
        }
    }
    __syncthreads();

    float block_offset = sequential_lookback(tid, block_prefix_sum);

    __shared__ float shared_block_offset;
    if (tid == 0) {
        shared_block_offset = block_offset;
    }
    __syncthreads();

    B[gid] = x + shared_block_offset;
}

__device__ inline float ks_warp_scan(float parent, uint32_t lane, uint32_t wid, float warp_offsets[]) {
#pragma unroll 1
    for (int delta = 1; delta < WARPSIZE; delta <<= 1) {
        float child = __shfl_up_sync(FULLMASK, parent, delta);
        if (lane >= delta) {
            parent += child;
        }
    }

    if (lane == WARPSIZE - 1){
        warp_offsets[wid] = parent;
    }

    return parent;
}

__device__ inline void ks_cross_warp_scan(uint32_t wid, uint32_t lane, float warp_sums[]){
    if (wid == 0){
        float warp_sum = warp_sums[lane];
        for (int delta = 1; delta < WARPSIZE; delta <<=1){
            float value= __shfl_up_sync(FULLMASK, warp_sum, delta);
            if(lane >= delta){
                warp_sum += value;
            }
        }
        warp_sums[lane] = warp_sum;
    }
}


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
                accumulation += block_status[prev] ;
                break;
            }
        }
        block_offset = accumulation;
        block_status[blockIdx.x] = block_offset + block_prefix_sum; // prefix
        __threadfence();
    }

    return block_offset;
}

__device__ inline float parallel_lookback(uint32_t tid, uint32_t lane, float block_prefix_sum){
    return 0.0f;
}

