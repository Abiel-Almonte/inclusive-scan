#include <stdint.h>
#include "config.h"

__device__ float block_status[MAXBLOCKS];
__device__ inline float
ks_block_scan(float x, uint32_t wid, uint32_t lane, uint32_t n_warps, uint32_t n_lanes, uint32_t mask);
__device__ inline float logarithmic_lookback(uint32_t tid, float block_prefix_sum);
__device__ inline float parallel_lookback(uint32_t tid, uint32_t lane, float block_prefix_sum);

/*
 * Single pass device wide parallel inclusive scan
 *
 * Kogge-Stone warp scan and cross warp reduction
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
    uint32_t remaining = N - bdim * blockIdx.x;
    uint32_t n_lanes = (remaining >= wid*WARPSIZE)? (remaining - wid*WARPSIZE)? WARPSIZE : remaining - wid*WARPSIZE : 0u;

    uint32_t mask = (1u << n_lanes) - 1u;

    float x = (gid < N) ? __ldg(&A[gid]) : 0.0f;
    x = ks_block_scan(x, wid, lane, n_warps, n_lanes, mask);

    __shared__ float block_prefix_sum;
    if (tid == 0) {
        if (blockIdx.x == 0) {
            block_status[blockIdx.x] = block_prefix_sum;
        } else {
            block_status[blockIdx.x] = -block_prefix_sum;
        }
    }
    __syncthreads();

    float block_offset = logarithmic_lookback(tid, block_prefix_sum);

    __shared__ float shared_block_offset;
    if (tid == 0) {
        shared_block_offset = block_offset;
    }
    __syncthreads();

    if (gid < N) {
        B[gid] = x + shared_block_offset;
    }
}

__device__ inline float
ks_block_scan(float x, uint32_t wid, uint32_t lane, uint32_t n_warps, uint32_t n_lanes, uint32_t mask) {
    __shared__ float warp_sums[WARPSIZE];

#pragma unroll 1
    for (int delta = 1; delta < n_lanes; delta <<= 1) {
        float value = __shfl_up_sync(mask, x, delta);
        if (lane >= delta) {
            x += value;
        }
    }

    if (lane == n_lanes - 1 && wid < n_warps) {
        warp_sums[wid] = x;
    }
    __syncthreads();

    if (wid == 0) {
        float warp_sum = warp_sums[lane];

#pragma unroll 1
        for (int delta = 1; delta < n_lanes; delta <<= 1) {
            float value = __shfl_up_sync(mask, warp_sum, delta);
            if (lane >= delta) {
                warp_sum += value;
            }
        }
        warp_sums[lane] = warp_sum;
    }
    __syncthreads();

    float warp_sum = 0.0f;
    if (wid > 0 && wid < n_warps) {
        warp_sum = warp_sums[wid - 1];
    }

    return x + __shfl_up_sync(mask, warp_sum, 1);
}

__device__ inline float logarithmic_lookback(uint32_t tid, float block_prefix_sum) { //similar to ks addr
    float block_offset = 0.0f;

    if (tid == 0 && blockIdx.x > 0) {
        float accumulation = 0.0f;
        int prev = blockIdx.x - 1;
        int jump = 1;

#pragma unroll 1
        while(prev >= 0){
            while (isnan(block_status[prev])) {}

            float value = block_status[prev];

            if (block_status[prev] < 0.0f) {
                accumulation -= value;
            } else {
                accumulation += value;
                break;
            }

            jump <<= 1;
            prev -= jump;
        }

        block_offset = accumulation;
        block_status[blockIdx.x] = block_offset + block_prefix_sum; // prefix
        __threadfence();
    }

    return block_offset;
}

__device__ inline float parallel_lookback(uint32_t tid, uint32_t lane, float block_prefix_sum) {
    return 0.0f;
}
