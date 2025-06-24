#include <stdint.h>

constexpr uint32_t BLOCKDIM= 128;

__global__ void naive_prefix_sum(float* A, float* out, uint32_t n){
    uint32_t index= blockIdx.x * BLOCKDIM  + threadIdx.x;
    
    if (index >= n){
        return;
    }

    float sum= 0.0f;
    for (uint32_t i= 0; i <= index; i++){
        sum+= A[i];
    }

    out[index] = sum;
}


/*
n_0 n_1 ... n_(s_dim) | n_(s+1) n_(s+2) ... n_(s_dim + s_dim) | ... 
T_s T_s ... T_s       | T_(s+1) T_(s+1)     T_(s+1)           | ... 
    n_0 ... n_0
        ... n_1
            .
            .
            .
            = T_s

T_s are dependant on the Total of each Section (parition).

first pass does prefix sum on each section.
second pass does prefix sum on the totals
finally we add the prefix sum of totals to correct sections


*/

__global__ void scan_blocks(float* A, uint32_t n, float* d_block_totals){
    uint32_t gid= blockIdx.x * BLOCKDIM + threadIdx.x;
    uint32_t tid= threadIdx.x;

    if (gid >= n){
        return;
    }
    
    __shared__ float shared_data[BLOCKDIM];
    shared_data[tid] = A[gid];

    __syncthreads();

    float sum= 0.0f;
    for (uint32_t i = 0; i <= tid; i++){
        sum += shared_data[i];
    }

    A[gid]= sum;

    if (tid == BLOCKDIM - 1) {
        d_block_totals[blockIdx.x] = A[gid];
    }
}

__global__ void scan_block_totals(float* A, uint32_t n, float* block_totals, float* block_scanned_totals){
    uint32_t tid = threadIdx.x;

    float sum = 0.0f;
    for (uint32_t i = 0; i < tid; i++){
        sum += block_totals[i];
    }

    block_scanned_totals[tid]= sum; 

}

__global__ void add_scanned_totals(float* A, uint32_t n, float* d_block_scanned_totals){
    uint32_t gid = blockIdx.x * BLOCKDIM + threadIdx.x;
    uint32_t bid = blockIdx.x;

    if (bid > 0 && gid < n){
        A[gid] += d_block_scanned_totals[bid];
    }

}

void two_pass_block_wide_prefix_sum(float* A, uint32_t n){
    uint32_t N_BLOCKS = (n + BLOCKDIM - 1) / BLOCKDIM;
    float* d_block_totals;
    float* d_block_scanned_totals;
    
    cudaMalloc(&d_block_totals, N_BLOCKS*sizeof(float));
    cudaMalloc(&d_block_scanned_totals, N_BLOCKS*sizeof(float));
    
    scan_blocks<<<N_BLOCKS, BLOCKDIM>>>(A, n, d_block_totals);
    scan_block_totals<<<1, N_BLOCKS>>>(A,n, d_block_totals, d_block_scanned_totals);
    add_scanned_totals<<<N_BLOCKS, BLOCKDIM>>>(A, n, d_block_scanned_totals);

    cudaDeviceSynchronize();

    cudaFree(d_block_totals);
    cudaFree(d_block_scanned_totals);
}


extern "C" {

    void launch_naive_bw_prefix_sum(float* A_host, uint32_t n, size_t bytes){
        uint32_t N_BLOCKS = (n + BLOCKDIM - 1) / BLOCKDIM;
        float* A_device;
        float* A_out_device;
        cudaMalloc(&A_device, bytes);
        cudaMalloc(&A_out_device, bytes);

        cudaMemcpy(A_device, A_host, bytes, cudaMemcpyHostToDevice);
        naive_prefix_sum<<<N_BLOCKS, BLOCKDIM>>>(A_device, A_out_device, n);
        cudaDeviceSynchronize();
        cudaMemcpy(A_host, A_out_device, bytes, cudaMemcpyDeviceToHost);

        cudaFree(A_device);
        cudaFree(A_out_device);
    }

    void launch_2pass_bw_prefix_sum(float* A_host, uint32_t n, size_t bytes){
        float* A_device;

        cudaMalloc(&A_device, bytes);
        cudaMemcpy(A_device, A_host, bytes, cudaMemcpyHostToDevice);

        two_pass_block_wide_prefix_sum(A_device, n);

        cudaMemcpy(A_host, A_device, bytes, cudaMemcpyDeviceToHost);
        cudaFree(A_device);
    }
    
}