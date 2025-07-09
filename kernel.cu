#include <stdint.h>
#include <math.h>
#include "config.h"

__device__ inline float
ks_block_scan(float x, uint32_t wid, uint32_t lane, uint32_t n_warps, uint32_t n_lanes, uint32_t mask);

volatile __device__ int64_t block_status[MAXBLOCKS];
__device__ inline int64_t pack_status(int64_t flag, float value);
__device__ inline int64_t unpack_flag(int64_t status);
__device__ inline float unpack_value(int64_t status);
__device__ inline float decoupled_lookback();

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

    if(tid == 0){
        block_status[blockIdx.x] = pack_status(INVALID, 0.0f);
    }
    __syncthreads();

    float x = (gid < N) ? __ldg(&A[gid]) : 0.0f;
    x = ks_block_scan(x, wid, lane, n_warps, n_lanes, mask);

    if (wid == n_warps - 1 && lane == n_lanes - 1){
        if (blockIdx.x == 0){
            block_status[blockIdx.x] = pack_status(PREFIX, x);
        }
        else{
            block_status[blockIdx.x] = pack_status(AGGREGATE, x);
        }
    }
    __syncthreads();

    __shared__ float block_offset;
    if (tid == 0){
        if (blockIdx.x == 0) {
            block_offset = 0.0f;
        } else {
            block_offset = decoupled_lookback();
        }
    }
    __syncthreads();

    if (gid < N) {
        B[gid] = x + block_offset;
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
        for (int delta = 1; delta < n_warps; delta <<= 1) { // Note: n_warps
            float value = __shfl_up_sync(FULLMASK, warp_sum, delta);
            if (lane >= delta) {
                warp_sum += value;
            }
        }
        warp_sums[lane] = warp_sum;
    }
    __syncthreads();

    float warp_prefix = (wid > 0) ? warp_sums[wid - 1] : 0.0f;

    return x + warp_prefix;
}

__device__ inline int64_t pack_status(int64_t flag, float value){
    return (flag << 32) | __float_as_int(value);
}
__device__ inline int64_t unpack_flag(int64_t status){
    return status >> 32;
}
__device__ inline float unpack_value(int64_t status){
    return __int_as_float(status & 0xFFFFFFFF);
}

__device__ inline float decoupled_lookback(){// this cannot be called by blockIdx.x == 0
    int base = blockIdx.x;
    float block_offset = 0.0f;
    
    if (base > 0){
        int prev = base - 1;
        float accumulation = 0.0f;

        while(prev >= 0){
            int64_t status;
            do{
                status = block_status[prev];
            } while (unpack_flag(status) == INVALID);

            float value = unpack_value(status);
            if (unpack_flag(status) == PREFIX){
                accumulation += value;
                break;
            }
            else{
                accumulation += value;
            }

            prev--;
        }

        block_offset = accumulation;

        float my_aggregate = unpack_value(block_status[base]);
        block_status[base] = pack_status(PREFIX, block_offset + my_aggregate);
        __threadfence_system();
    }

    return block_offset;
}


__global__ void test_packing_kernel(int* d_result) {
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;

    if (tid == 0 && bid == 0) {
        // Test case
        const int64_t original_flag = PREFIX;
        const float original_value = 3.14f;
        const float epsilon = 1e-6f;

        // 1. Pack the values
        int64_t packed_data = pack_status(original_flag, original_value);

        // 2. Unpack the values
        int64_t unpacked_flag_val = unpack_flag(packed_data);
        float unpacked_value_val = unpack_value(packed_data);

        // 3. Verify
        bool flag_ok = (unpacked_flag_val == original_flag);
        bool value_ok = (fabsf(unpacked_value_val - original_value) < epsilon);

        if (flag_ok && value_ok) {
            *d_result = 1; // Success
        } else {
            *d_result = 0; // Failure
        }
    }
} 