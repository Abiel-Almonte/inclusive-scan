#include <algorithm>
#include <cmath>
#include <cub/cub.cuh>
#include <iostream>
#include <numeric>
#include <vector>

#include "config.h"

#define CHECK_CUDA(call)                                                                                     \
    do {                                                                                                     \
        cudaError_t _err = call;                                                                             \
        if (_err != cudaSuccess) {                                                                           \
            std::cerr << "CUDA error: " << cudaGetErrorString(_err) << " at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                                                                          \
            exit(1);                                                                                         \
        }                                                                                                    \
    } while (0)

extern "C" {

__global__ void single_pass_scan(float* A, float* B, uint32_t N);

void compare_with_cub(float* h_input, uint32_t n, int reps) {
    size_t bytes = n * sizeof(float);

    float *d_input, *d_output_mine, *d_output_cub;
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output_mine, bytes));
    CHECK_CUDA(cudaMalloc(&d_output_cub, bytes));
    CHECK_CUDA(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, d_input, d_output_cub, n);
    void* d_temp_storage = nullptr;
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::vector<float> times_mine, times_cub;
    uint32_t N_BLOCKS = (n + BLOCKDIM - 1) / BLOCKDIM;
    size_t num_warps = (BLOCKDIM + 32 - 1) / 32;
    size_t SHMEM_BYTES = num_warps * sizeof(float);

    for (int i = 0; i < reps; i++) {
        CHECK_CUDA(cudaEventRecord(start));
        single_pass_scan<<<N_BLOCKS, BLOCKDIM, SHMEM_BYTES>>>(d_input, d_output_mine, n);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
        times_mine.push_back(elapsed);
    }

    for (int i = 0; i < reps; i++) {
        CHECK_CUDA(cudaEventRecord(start));
        cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_input, d_output_cub, n);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float elapsed;
        CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, stop));
        times_cub.push_back(elapsed);
    }

    std::sort(times_mine.begin(), times_mine.end());
    std::sort(times_cub.begin(), times_cub.end());
    float median_mine = times_mine[reps / 2];
    float median_cub = times_cub[reps / 2];

    std::cout << "\n=== Performance Comparison (n=" << n << ") ===" << std::endl;

    std::cout << "Your implementation:\n";
    std::cout << "  Array size: " << n << " elements\n";
    std::cout << "  Median time: " << median_mine << " ms\n";
    std::cout << "  Throughput: " << (n / median_mine / 1e6) << " billion elements/sec\n";
    std::cout << "  Bandwidth: " << (2 * bytes / median_mine / 1e6) << " GB/s\n";

    std::cout << "CUB implementation:\n";
    std::cout << "  Array size: " << n << " elements\n";
    std::cout << "  Median time: " << median_cub << " ms\n";
    std::cout << "  Throughput: " << (n / median_cub / 1e6) << " billion elements/sec\n";
    std::cout << "  Bandwidth: " << (2 * bytes / median_cub / 1e6) << " GB/s\n";

    std::cout << "Ratio (yours/CUB): " << median_mine / median_cub << "x\n";

    std::vector<float> h_mine(n), h_cub(n);
    CHECK_CUDA(cudaMemcpy(h_mine.data(), d_output_mine, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_cub.data(), d_output_cub, bytes, cudaMemcpyDeviceToHost));

    bool match = true;
    float abs_tol = 1e-3f;
    float rel_tol = 1e-6f;
    for (size_t i = 0; i < n; ++i) {
        float diff = std::abs(h_mine[i] - h_cub[i]);
        float max_val = std::max(std::abs(h_mine[i]), std::abs(h_cub[i]));
        if (diff > abs_tol && diff > rel_tol * max_val) {
            std::cout << "Mismatch at " << i << ": " << h_mine[i] << " vs " << h_cub[i] << " (diff: " << diff << ")"
                      << std::endl;
            match = false;
        }
    }
    if (match)
        std::cout << "\nResults match CUB" << std::endl;
    else
        std::cout << "Mismatch found!\n";

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output_mine));
    CHECK_CUDA(cudaFree(d_output_cub));
    CHECK_CUDA(cudaFree(d_temp_storage));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}
}