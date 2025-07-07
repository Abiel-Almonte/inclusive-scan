#include <iostream>
#include <stdint.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>

constexpr long long TARGET_SEC_NS = 10'000'000'000;

extern "C" {
    void compare_with_cub(float* h_input, uint32_t n, int reps);
}

constexpr uint32_t BLOCK_SIZE= 64;

size_t get_alloc_size(uint32_t n_floats){
    constexpr size_t CACHE_LINE_SIZE = 64;
    size_t bytes_needed = n_floats * sizeof(float);
    return ((bytes_needed + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE) * CACHE_LINE_SIZE;
}

int main(int argc, char** argv){

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <array_size>" << std::endl;
        return EXIT_FAILURE;
    }

    int power_of_two = std::atoi(argv[1]);
    int size = 1 << power_of_two;

    std::cout << "Array size: " << size << std::endl;

    size_t alloc_size_bytes= get_alloc_size(size);
    float* A= static_cast<float*>(aligned_alloc(32, alloc_size_bytes));

    for (int i = 0; i < size; i++) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    compare_with_cub(A, size, 100);
 
    free(A);
    return 0;
}