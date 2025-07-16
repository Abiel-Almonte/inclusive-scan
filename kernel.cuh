




#include <stdint.h>

#define BLOCKDIM 512
#define WARPSIZE 32
#define FULLMASK 0xffffffff
#define VEC_SIZE 4
#define VECS_SIZE 1

#define INVALID 0
#define AGGREGATE 1
#define PREFIX 2


__device__ float ks_block_scan_4x(float4& vec, uint32_t wid, uint32_t lane, uint32_t n_warps);

__device__ void parallel_lookback(volatile uint64_t* temp_storage, float* shared_prefix_sum);
__device__ __forceinline__ float warp_reduce_sum(float val);
__device__ __forceinline__ uint64_t pack_status(uint32_t flag, float sum);
__device__ __forceinline__ void unpack_status(uint64_t status, uint32_t& flag, float& sum);
__device__ __forceinline__ float4 float4_add(float s, float4 v);



extern "C" __global__ void single_pass_scan_4x(const float4* A, float4* B, volatile uint64_t* temp_storage, uint32_t N_vec) {
    __shared__ float shared_prefix_sum;

    const uint32_t tid = threadIdx.x;
    const uint32_t gid = blockIdx.x * BLOCKDIM + tid;
    const uint32_t wid = tid / WARPSIZE;
    const uint32_t lane = tid & (WARPSIZE - 1);
    const uint32_t n_warps = BLOCKDIM / WARPSIZE;

    if (tid == 0) {
        shared_prefix_sum = 0.0f;
        temp_storage[blockIdx.x] = pack_status(INVALID, 0.0f);
        __threadfence();
    }

    float4 x = (gid < N_vec) ? A[gid] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float block_sum = ks_block_scan_4x(x, wid, lane, n_warps);

    if (tid == 0) {
        uint32_t flag = (blockIdx.x == 0) ? PREFIX : AGGREGATE;
        temp_storage[blockIdx.x] = pack_status(flag, block_sum);
        __threadfence();
    }


    if (blockIdx.x > 0) {
        if (wid == 0) {
            parallel_lookback(temp_storage, &shared_prefix_sum);
        }
        __syncthreads();

        if (tid == 0) {
            float final_sum = block_sum + shared_prefix_sum;
            temp_storage[blockIdx.x] = pack_status(PREFIX, final_sum);
            __threadfence();
        }
    }
    __syncthreads();

    float4 final_val = float4_add(shared_prefix_sum, x);
    if (gid < N_vec) {
        B[gid] = final_val;
    }
}




__device__ float ks_block_scan_4x(float4& vec, uint32_t wid, uint32_t lane, uint32_t n_warps) {
    __shared__ float warp_sums[WARPSIZE];
    vec.y += vec.x;
    vec.z += vec.y;
    vec.w += vec.z;
    float thread_sum = vec.w;

#pragma unroll 4
    for (uint32_t delta = 1; delta < WARPSIZE; delta <<= 1) {
        float neighbor_sum = __shfl_up_sync(FULLMASK, thread_sum, delta);
        if (lane >= delta) {
            thread_sum += neighbor_sum;
        }
    }

    float last_warp_offset = __shfl_up_sync(FULLMASK, thread_sum, 1);
    if (lane == 0){
        last_warp_offset = 0.0f;
    }

    if (lane == WARPSIZE - 1) {
        warp_sums[wid] = thread_sum;
    }

    __syncthreads();
    if (wid == 0) {
        float warp_sum = (lane < n_warps) ? warp_sums[lane] : 0.0f;

#pragma unroll 32
        for (uint32_t delta = 1; delta < n_warps; delta <<= 1) {
            float neighbor_warp_sum = __shfl_up_sync(FULLMASK, warp_sum, delta);
            if (lane >= delta) {
                warp_sum += neighbor_warp_sum;
            }
        }
        if (lane < n_warps) {
            warp_sums[lane] = warp_sum;
        }
    }
    __syncthreads();
    float rest_warp_offset = (wid > 0) ? warp_sums[wid - 1] : 0.0f;
    vec = float4_add(rest_warp_offset +  last_warp_offset, vec);
    return warp_sums[n_warps - 1];
}




__device__ void parallel_lookback(volatile uint64_t* temp_storage, float* shared_prefix_sum) {
    const uint32_t lane = threadIdx.x;
    float prefix = 0.0f;
    int base = blockIdx.x - 1;

    while (base >= 0) {
        int prev = base - lane;
        uint64_t status = (prev >= 0) ? temp_storage[prev] : pack_status(PREFIX, 0.0f);

        float prev_sum;
        uint32_t prev_flag;
        unpack_status(status, prev_flag, prev_sum);

        uint32_t ready_ballot = __ballot_sync(FULLMASK, prev_flag != INVALID || prev < 0);
        if (ready_ballot != FULLMASK) {
            continue;
        }

        uint32_t prefix_ballot = __ballot_sync(FULLMASK, prev_flag == PREFIX);

        float aggregate = 0.0f;
        if (prefix_ballot > 0) {
            uint32_t first_prefix_lane = __ffs(prefix_ballot) - 1;

            if (lane <= first_prefix_lane) {
                aggregate = prev_sum;
            }
        } else {
            if (prev >= 0) {
                aggregate = prev_sum;
            }
        }

        float accumulation = warp_reduce_sum(aggregate);
        if (lane == 0)
            prefix += accumulation;

        if (prefix_ballot > 0) {
            if (lane == 0) {
                *shared_prefix_sum = prefix;
            }
            break;
        }

        base -= WARPSIZE;
    }

    if (base < 0 && lane == 0) {
        *shared_prefix_sum = prefix;
    }
}

__device__ __forceinline__ uint64_t pack_status(uint32_t flag, float sum) {
    uint32_t sum_bits = __float_as_uint(sum);
    return ((uint64_t) flag << 32) | sum_bits;
}

__device__ __forceinline__ void unpack_status(uint64_t status, uint32_t& flag, float& sum) {
    flag = (uint32_t) (status >> 32);
    sum = __uint_as_float((uint32_t) status);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (uint32_t delta = WARPSIZE / 2; delta > 0; delta /= 2) {
        val += __shfl_down_sync(FULLMASK, val, delta);
    }
    return val;
}

__device__ __forceinline__ float4 float4_add(float s, float4 v) {
    return make_float4(s + v.x, s + v.y, s + v.z, s + v.w);
}
