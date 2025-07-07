#include <stdint.h>
#include "config.h"

__device__ float block_status[MAXBLOCKS];
__device__ __forceinline__ float ks_warp_scan(float parent, uint32_t mask, uint32_t lane, uint32_t active_lanes, uint32_t wid, float warp_offsets[]);
__device__ __forceinline__ float make_warp_scan_exclusive(float inclusive_sum, uint32_t mask, uint32_t lane);
__device__ __forceinline__ void blelloch_cross_warp_upsweep(uint32_t tid, uint32_t n_warps, float warp_sums[]);
__device__ __forceinline__ void blelloch_cross_warp_downsweep(uint32_t tid, uint32_t n_warps, float warp_sums[]);
__device__ __forceinline__ float sequential_lookback(uint32_t tid, float block_prefix_sum);
__device__ __forceinline__ float parallel_lookback(uint32_t tid, uint32_t lane, float block_prefix_sum);
// flag << value;

/*
 * Single pass device wide parallel prefix sum
 *
 * Kogge-Stone warp scan
 * Blelloch block scan
 * Merrill & Garland's decoupled lookback
 */
extern "C" __global__ void single_pass_scan(float* A, float* B, uint32_t N) {
    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + tid;

    if (tid == 0) {
        block_status[blockIdx.x] = NAN; // invalid
    }
    __syncthreads();

    uint32_t wid = tid / WARPSIZE;
    uint32_t lane = tid % WARPSIZE;
    uint32_t n_warps = (blockDim.x + WARPSIZE - 1) / WARPSIZE;

    uint32_t remaining = N - blockIdx.x * blockDim.x;
    uint32_t active_lanes =
        (remaining > wid * WARPSIZE) ? ((remaining >= WARPSIZE) ? WARPSIZE : remaining - wid * WARPSIZE) : 0;

    uint32_t mask = (active_lanes > 0) ? (1u << active_lanes) - 1u : 0;

    float x = (gid < N && active_lanes > 0) ? A[gid] : 0.0f;
    float original_x = x;
    
    extern __shared__ float warp_sums[];

    x = ks_warp_scan(x, mask, lane, active_lanes, wid, warp_sums);
    __syncthreads();

    blelloch_cross_warp_upsweep(tid, n_warps, warp_sums);

    if (tid == 0) {
        warp_sums[n_warps - 1] = 0.0f;
    }
    __syncthreads();

    blelloch_cross_warp_downsweep(tid, n_warps, warp_sums);
    x += warp_sums[wid];

    __shared__ float block_prefix_sum;
    if (tid == blockDim.x - 1) {
        block_prefix_sum = x + original_x;

        if (blockIdx.x == 0) {
            block_status[0] = block_prefix_sum;
        } else {
            block_status[blockIdx.x] = -block_prefix_sum; // aggregate
        }
    }
    __syncthreads();

    float block_offset = sequential_lookback(tid, block_prefix_sum);

    __shared__ float shared_block_offset;
    if (tid == 0) {
        shared_block_offset = block_offset;
    }
    __syncthreads();

    if (gid < N && active_lanes > 0) {
        B[gid] = x + shared_block_offset;
    }
}

__device__ __forceinline__ float ks_warp_scan(float parent, uint32_t mask, uint32_t lane, uint32_t active_lanes, uint32_t wid, float warp_offsets[]) {
#pragma unroll 1
    for (int delta = 1; delta < active_lanes; delta <<= 1) {
        float child = __shfl_up_sync(mask, parent, delta);
        if (lane >= delta && lane < active_lanes) {
            parent += child;
        }
    }

    if (lane == active_lanes - 1){
        warp_offsets[wid] = parent;
    }

    parent = __shfl_up_sync(mask, parent, 1);
    if (lane == 0){
        parent = 0.0f;
    }

    return parent;
}

__device__ __forceinline__ void blelloch_cross_warp_upsweep(uint32_t tid, uint32_t n_warps, float warp_sums[]) {
    uint32_t tree_index = tid + 1;

#pragma unroll 1
    for (int delta = 1; delta <= n_warps / 2; delta <<= 1) {
        if (tid < n_warps) {
            uint32_t parent = tree_index * 2 * delta - 1;
            if (parent < n_warps) {
                warp_sums[parent] += warp_sums[parent - delta];
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void blelloch_cross_warp_downsweep(uint32_t tid, uint32_t n_warps, float warp_sums[]) {
    uint32_t tree_index = tid + 1;

#pragma unroll 1
    for (int delta = n_warps >> 1; delta >= 1; delta >>= 1) {
        if (tid < n_warps) {
            uint32_t parent = tree_index * 2 * delta - 1;
            if (parent < n_warps) {
                uint32_t child = parent - delta;
                float temp = warp_sums[parent];
                warp_sums[parent] += warp_sums[child];
                warp_sums[child] = temp;
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ float sequential_lookback(uint32_t tid, float block_prefix_sum) {
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

__device__ __forceinline__ float parallel_lookback(uint32_t tid, uint32_t lane, float block_prefix_sum){
    return 0.0f;
}

