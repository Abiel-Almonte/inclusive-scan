#include <iostream>
#include <stdint.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>

constexpr long long TARGET_SEC_NS = 1'000'000'000;

extern "C" {
    //inclusive sum
    void launch_naive_bw_prefix_sum(float* A_host, uint32_t n, size_t bytes);
    void launch_2pass_bw_prefix_sum(float* A_host, uint32_t n, size_t bytes);
}

constexpr uint32_t BLOCK_SIZE= 64;

size_t get_alloc_size(uint32_t n_floats){
    size_t floats_per_cacheline= BLOCK_SIZE/sizeof(float);
    size_t blocks_to_alloc= (n_floats + floats_per_cacheline - 1) / floats_per_cacheline;
    return blocks_to_alloc*BLOCK_SIZE;
}

int main(int argc, char** argv){

    int size = std::atoi(argv[1]);
    std::cout << "Array size: " << size << std::endl;

    size_t alloc_size_bytes= get_alloc_size(size);
    float* A_naive= static_cast<float*>(aligned_alloc(32, alloc_size_bytes));
    float* A_rival= static_cast<float*>(aligned_alloc(32, alloc_size_bytes));

    std::vector<double> timings(50);
    
    for (int i = 0; i < 50; i++){ 
        for (int j = 0; j < size; j++){
            A_rival[j]= static_cast<float>(j);
        }

        auto start= std::chrono::high_resolution_clock::now();
        launch_naive_bw_prefix_sum(A_naive, size, alloc_size_bytes);
        auto end = std::chrono::high_resolution_clock::now();

        double latency_ns= (end - start).count();
        timings[i]= latency_ns;
    }

    std::sort(timings.begin(), timings.end());
    double avg_latency_per_rep= timings.at(24);

    long long reps= static_cast<long long>(TARGET_SEC_NS/ avg_latency_per_rep);
    std::vector<double> timings_naive(reps);
    std::vector<double> timings_rival(reps);
    
    for (long long i = 0; i < reps; i++){
        for (int j = 0; j < size; j++){
            A_rival[j]= static_cast<float>(j);
            A_naive[j]= static_cast<float>(j);
        }

        auto start= std::chrono::high_resolution_clock::now();
        launch_naive_bw_prefix_sum(A_naive, size, alloc_size_bytes);
        auto end = std::chrono::high_resolution_clock::now();

        double latency_ns= (end - start).count();
        timings_naive[i]= latency_ns;

        start = std::chrono::high_resolution_clock::now();
        launch_2pass_bw_prefix_sum(A_rival, size, alloc_size_bytes);
        end = std::chrono::high_resolution_clock::now();

        latency_ns= (end - start).count();
        timings_rival[i]= latency_ns;

        if (i < 1){
            for (int j = 0; j < size; j++){
                if (std::fabs(A_naive[j] - A_rival[j]) > 1e-4f){
                    std::cout << "Sanity Check: Failed" << std::endl;
                    free(A_naive);
                    free(A_rival);
                    exit(EXIT_FAILURE);
                }
            }
            std::cout << "Sanity Check: OK" << std::endl;
        }
    }

    free(A_naive);
    free(A_rival);

    std::sort(timings_naive.begin(), timings_naive.end());
    std::sort(timings_rival.begin(), timings_rival.end());

    long long p95_index= reps*95/ 100;
    long long p50_index= reps*50/ 100;

    double p95_latency_naive = timings_naive[p95_index];
    double p50_latency_naive = timings_naive[p50_index];

    double p95_latency_rival = timings_rival[p95_index];
    double p50_latency_rival = timings_rival[p50_index];

    std::cout << "Median Latency: " << std::endl;
    std::cout << "naive: " << p50_latency_naive* 10e-6 << " ms" << std::endl;
    std::cout << "rival: " << p50_latency_rival* 10e-6 << " ms" << std::endl;

    std::cout << "P95 Latency: " << std::endl;
    std::cout << "naive: " << p95_latency_naive* 10e-6 << " ms" << std::endl;
    std::cout << "rival: " << p95_latency_rival* 10e-6 << " ms" << std::endl;


    return 0;
}