#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include KERNEL_HEADER_PATH

#define VEC_SIZE 4

void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

bool validate(const std::vector<float>& our_result, const std::vector<float>& cub_result, int n, float& max_rel_err) {
    max_rel_err = 0.0f;
    bool passed = true;
    for (int i = 0; i < n; ++i) {
        float ref = cub_result[i];
        float res = our_result[i];
        float rel_err = (ref != 0) ? fabsf((res - ref) / ref) : fabsf(res - ref);
        if (rel_err > 1e-5f) {
            max_rel_err = rel_err;
            passed = false;
        }
    }
    return passed;
}

int main(int argc, char** argv) {
    int N;
    if (argc > 1) {
        N = atoi(argv[1]);
    } else {
        N = 1 << 28;
    }

    std::cout << "Benchmarking for N = " << N << std::endl;

    const int N_vec = (N + VEC_SIZE - 1) / VEC_SIZE;
    const size_t n_bytes = (size_t) N * sizeof(float);

    std::vector<float> h_A(N);
    for (int i = 0; i < N; ++i)
        h_A[i] = (rand() % 100) / 100.0f;

    float *d_A, *d_B;
    checkCudaErrors(cudaMalloc((void**) &d_A, n_bytes));
    checkCudaErrors(cudaMalloc((void**) &d_B, n_bytes));
    checkCudaErrors(cudaMemcpy(d_A, h_A.data(), n_bytes, cudaMemcpyHostToDevice));

    uint32_t gridDim = (N_vec + BLOCKDIM - 1) / BLOCKDIM;
    uint64_t* d_temp_storage_mine;
    checkCudaErrors(cudaMalloc((void**) &d_temp_storage_mine, gridDim * sizeof(uint64_t)));
    checkCudaErrors(cudaMemset(d_temp_storage_mine, 0, gridDim * sizeof(uint64_t)));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Warm-up run
    single_pass_scan_4x<<<gridDim, BLOCKDIM>>>((const float4*) d_A, (float4*) d_B, d_temp_storage_mine, N_vec);

    std::vector<float> timings;
    for (int i = 0; i < 101; ++i) {
        checkCudaErrors(cudaEventRecord(start));
        checkCudaErrors(cudaMemset(d_temp_storage_mine, 0, gridDim * sizeof(uint64_t)));
        single_pass_scan_4x<<<gridDim, BLOCKDIM>>>((const float4*) d_A, (float4*) d_B, d_temp_storage_mine, N_vec);
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        float ms;
        checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));
        timings.push_back(ms);
    }
    std::sort(timings.begin(), timings.end());
    double median_ms = timings[timings.size() / 2];

    double bandwidth = (double) n_bytes * 2 / (median_ms / 1000.0) / 1e9;
    std::cout << "My Kernel Performance: " << bandwidth << " GB/s" << std::endl;

    std::vector<float> h_B(N);
    std::vector<float> h_B_cub(N);
    checkCudaErrors(cudaMemcpy(h_B.data(), d_B, n_bytes, cudaMemcpyDeviceToHost));

    void* d_temp_storage_cub = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage_cub, temp_storage_bytes, d_A, d_B, N);
    checkCudaErrors(cudaMalloc(&d_temp_storage_cub, temp_storage_bytes));

    cub::DeviceScan::InclusiveSum(d_temp_storage_cub, temp_storage_bytes, d_A, d_B, N);

    timings.clear();
    for (int i = 0; i < 101; ++i) {
        checkCudaErrors(cudaEventRecord(start));
        cub::DeviceScan::InclusiveSum(d_temp_storage_cub, temp_storage_bytes, d_A, d_B, N);
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        float ms;
        checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));
        timings.push_back(ms);
    }
    std::sort(timings.begin(), timings.end());
    median_ms = timings[timings.size() / 2];

    double cub_bandwidth = (double) n_bytes * 2 / (median_ms / 1000.0) / 1e9;
    checkCudaErrors(cudaMemcpy(h_B_cub.data(), d_B, n_bytes, cudaMemcpyDeviceToHost));
    std::cout << "CUB Kernel Performance: " << cub_bandwidth << " GB/s" << std::endl;

    float max_rel_err;
    bool passed = validate(h_B, h_B_cub, N, max_rel_err);
    if (passed) {
        std::cout << "Validation: [PASS]" << std::endl;
    } else {
        std::cout << "Validation: [FAIL] (Max Rel Err: " << std::scientific << max_rel_err << ")" << std::endl;
    }

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_temp_storage_mine);
    cudaFree(d_temp_storage_cub);

    return 0;
}