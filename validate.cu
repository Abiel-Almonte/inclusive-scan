#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include KERNEL_HEADER_PATH

#define VEC_SIZE 4
#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_RESET "\x1b[0m"

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
            return false;
        }
    }
    return passed;
}

bool run_test(int N) {
    std::cout << "Testing N = " << std::left << std::setw(12) << N;

    if (N == 0) {
        std::cout << ANSI_COLOR_GREEN << "[PASS]" << ANSI_COLOR_RESET << " (Edge case: N=0)" << std::endl;
        return true;
    }

    std::vector<float> h_A(N);
    std::vector<float> h_B(N, 0.0f);
    std::vector<float> h_B_cub(N, 0.0f);

    for (int i = 0; i < N; ++i) {
        h_A[i] = (rand() % 100) / 100.0f;
    }

    float *d_A, *d_B, *d_B_cub;
    checkCudaErrors(cudaMalloc((void**) &d_A, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**) &d_B, N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**) &d_B_cub, N * sizeof(float)));

    checkCudaErrors(cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    uint32_t N_vec = (N + VEC_SIZE - 1) / VEC_SIZE;
    uint32_t gridDim = (N_vec + BLOCKDIM - 1) / BLOCKDIM;

    uint64_t* d_payload;
    checkCudaErrors(cudaMalloc((void**) &d_payload, gridDim * sizeof(uint64_t)));
    checkCudaErrors(cudaMemset(d_payload, 0, gridDim * sizeof(uint64_t)));

    single_pass_scan_4x<<<gridDim, BLOCKDIM>>>((const float4*) d_A, (float4*) d_B, d_payload, N_vec);
    checkCudaErrors(cudaGetLastError());

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_A, d_B_cub, N);
    checkCudaErrors(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_A, d_B_cub, N);

    checkCudaErrors(cudaMemcpy(h_B.data(), d_B, N * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_B_cub.data(), d_B_cub, N * sizeof(float), cudaMemcpyDeviceToHost));

    float max_rel_err;
    bool passed = validate(h_B, h_B_cub, N, max_rel_err);

    if (passed) {
        std::cout << ANSI_COLOR_GREEN << "[PASS]" << ANSI_COLOR_RESET << std::endl;
    } else {
        std::cout << ANSI_COLOR_RED << "[FAIL]" << ANSI_COLOR_RESET << " (Max Rel Err: " << max_rel_err << ")"
                  << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_B_cub);
    cudaFree(d_payload);
    cudaFree(d_temp_storage);

    return passed;
}

int main() {
    std::vector<int> test_sizes = {
        0,
        1,
        2,
        3,
        4,
        5,
        15,
        16,
        17,
        31,
        32,
        33,
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        256,
        257,
        511,
        512,
        513,
        1023,
        1024,
        1025,
        2047,
        2048,
        2049,
        4095,
        4096,
        4097,
        (1 << 14) - 1,
        (1 << 14),
        (1 << 14) + 1,
        (1 << 16) - 1,
        (1 << 16),
        (1 << 16) + 1,
        (1 << 20) - 1,
        (1 << 20),
        (1 << 20) + 1,
        1000,
        5000,
        10000,
        50000,
        100000,
    };

    int passed_count = 0;
    int total_tests = test_sizes.size();

    for (int n : test_sizes) {
        if (run_test(n)) {
            passed_count++;
        }
    }

    std::cout << "\n----------------------------------------\n";
    std::cout << "Validation Summary:\n";
    if (passed_count == total_tests) {
        std::cout << ANSI_COLOR_GREEN << "All " << total_tests << " tests passed!" << ANSI_COLOR_RESET << std::endl;
    } else {
        std::cout << ANSI_COLOR_RED << passed_count << " / " << total_tests << " tests passed." << ANSI_COLOR_RESET
                  << std::endl;
    }
    std::cout << "----------------------------------------\n";

    return (passed_count == total_tests) ? 0 : 1;
}