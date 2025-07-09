#include <stdint.h>
#include "config.h"

__device__ inline float
ks_block_scan(float x, uint32_t wid, uint32_t lane, uint32_t n_warps, uint32_t n_lanes, uint32_t mask);

/*
 * Single pass device wide parallel inclusive scan
 *
 * Kogge-Stone warp scan and cross warp reduction
 * Merrill & Garland's decoupled lookback
 */
extern "C" __global__ void single_pass_scan(float* A, float* B, uint32_t N) {
    const uint32_t tid = threadIdx.x;
    const uint32_t bdim = blockDim.x;
    const uint32_t gid = blockIdx.x * bdim + tid;
    const uint32_t wid = tid / WARPSIZE;
    const uint32_t lane = tid & (WARPSIZE - 1);
    const uint32_t n_warps = (bdim + WARPSIZE - 1) / WARPSIZE;
    const uint32_t remaining = (N > blockIdx.x * bdim) ? N - blockIdx.x * bdim : 0;
    const uint32_t rem = (remaining > wid * WARPSIZE) ? remaining - wid * WARPSIZE : 0;
    const uint32_t n_lanes = (rem > WARPSIZE) ? WARPSIZE : rem;
    const uint32_t mask = (1u << n_lanes) - 1u;

    float x = (gid < N) ? __ldg(&A[gid]) : 0.0f;
    x = ks_block_scan(x, wid, lane, n_warps, n_lanes, mask);

    if (gid < N) {
        B[gid] = x;
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
